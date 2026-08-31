"""Sport-neutral Bradley-Terry power ratings.

Generalized out of src/cfb/rankings.py, which fits college football. Nothing here is
college-specific: the model needs only (home, away, home_won, neutral_site) per game,
and the FBS/FCS term that used to be hardwired is now an optional covariate that a
single-division sport simply does not pass.

Why Bradley-Terry rather than Elo: Elo is path-dependent. It walks games in order,
regresses at season boundaries and weights late results more than early ones, so two
teams with identical resumes can rate differently purely because of scheduling order.
That is fine for forecasting and wrong for "who is actually best". This fits one global
maximum-likelihood solve over every game at once, so shuffling the schedule changes
nothing.

Design matrix, one row per game:
    +1 in the home team's column, -1 in the away team's column,
    +1 in a home_field column (0 at neutral sites),
    optionally +/-1 in a division column for cross-division games
    target = home team won

Coefficients are strengths on a log-odds scale, so the difference between two teams
converts straight to a win probability:
    P(A beats B on a neutral field) = sigmoid(strength_A - strength_B)

Ridge regularization is not optional. Unbeaten and winless teams produce infinite
maximum-likelihood estimates (complete separation); the penalty shrinks them toward
average, which is also the honest answer, since going undefeated against weak
opposition is weaker evidence than the raw record suggests.

Rank uncertainty comes from a bootstrap over games. That is what answers "how close are
they": a team whose rank swings between 3 and 19 across resamples is not meaningfully
separated from its neighbours however confident the point estimate looks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Convert log-odds strengths to a familiar Elo-like point scale (400 per 10x odds).
ELO_SCALE = 400.0 / np.log(10.0)

# The division term is deliberately under-penalized relative to team terms. Only a few
# percent of college games cross divisions, so plain ridge would shrink this toward zero
# and let FCS teams drift onto the FBS scale — the failure mode where a team that never
# played an FBS opponent lands in the top 10. Scaling the column up by this factor (and
# dividing the coefficient back out) weakens its effective penalty by roughly f**2.
DIV_SCALE = 12.0

REQUIRED_COLUMNS = (
    "season", "week", "home_team_name", "away_team_name", "home_won", "neutral_site",
)


def validate(games: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in games.columns]
    if missing:
        raise ValueError(f"games frame is missing required columns: {missing}")


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def team_divisions(games: pd.DataFrame) -> dict[str, str]:
    """Division per team, when the sport has divisions. Empty dict otherwise.

    Prefers the per-side `home_division`/`away_division` columns. Divisions belong to a
    (team, season) pair, not a team — programs move up, and Sacramento State went FCS
    to FBS between 2025 and 2026. A frame covering several seasons therefore has to say
    which side of each game was in which division, and the caller decides whose season
    it is by choosing what to pass in.
    """
    has_sides = {"home_division", "away_division"} <= set(games.columns)
    if not has_sides and "division" not in games.columns:
        return {}

    out: dict[str, str] = {}
    for g in games.itertuples(index=False):
        if has_sides:
            home = _clean(getattr(g, "home_division", None))
            away = _clean(getattr(g, "away_division", None))
        else:
            home = away = _clean(getattr(g, "division", None))
        if home is not None:
            out.setdefault(g.home_team_name, home)
        if away is not None:
            out.setdefault(g.away_team_name, away)
    return out


def _design(games: pd.DataFrame, teams: list[str], divisions: dict[str, str],
            major: str | None) -> tuple[csr_matrix, np.ndarray]:
    """Build the sparse design matrix and target.

    `major` names the stronger division (e.g. "fbs"); pass None for a single-division
    sport, which drops the division column entirely.
    """
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(games), len(teams)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    div_col = np.zeros(n)

    # Per-game divisions when the frame carries them, so a team that changed division
    # between seasons is treated correctly in each season's games rather than being
    # retroactively assigned one division for its whole history.
    has_sides = {"home_division", "away_division"} <= set(games.columns)

    for r, g in enumerate(games.itertuples(index=False)):
        rows += [r, r]
        cols += [idx[g.home_team_name], idx[g.away_team_name]]
        vals += [1.0, -1.0]
        if major is not None:
            if has_sides:
                home_div = _clean(getattr(g, "home_division", None))
                away_div = _clean(getattr(g, "away_division", None))
            else:
                home_div = divisions.get(g.home_team_name)
                away_div = divisions.get(g.away_team_name)
            h = 1.0 if home_div == major else 0.0
            a = 1.0 if away_div == major else 0.0
            div_col[r] = (h - a) * DIV_SCALE

    x_teams = csr_matrix((vals, (rows, cols)), shape=(n, k))
    # Home field only applies where the game actually had a home team.
    home_col = csr_matrix(
        (1.0 - games["neutral_site"].to_numpy(dtype=float)).reshape(-1, 1)
    )
    blocks = [x_teams, home_col]
    if major is not None:
        blocks.append(csr_matrix(div_col.reshape(-1, 1)))

    x = hstack(blocks, format="csr")
    y = games["home_won"].to_numpy(dtype=int)
    return x, y


def _solve(games: pd.DataFrame, teams: list[str], divisions: dict[str, str],
           major: str | None, C: float, weights: np.ndarray | None
           ) -> tuple[pd.Series, float, float]:
    x, y = _design(games, teams, divisions, major)
    model = LogisticRegression(C=C, fit_intercept=False, solver="lbfgs", max_iter=5000)
    model.fit(x, y, sample_weight=weights)

    coef = model.coef_[0]
    within = pd.Series(coef[:len(teams)] * ELO_SCALE, index=teams)
    home_adv = float(coef[len(teams)] * ELO_SCALE)
    div_gap = (
        float(coef[len(teams) + 1] * DIV_SCALE * ELO_SCALE) if major is not None else 0.0
    )

    # Put both ladders on one scale by adding each team's division baseline back on.
    if major is not None:
        baseline = pd.Series(
            [div_gap if divisions.get(t) == major else 0.0 for t in teams], index=teams
        )
        within = within + baseline

    strengths = within - within.mean()
    return strengths.sort_values(ascending=False), home_adv, div_gap


def fit_ratings(games: pd.DataFrame, C: float = 2.0,
                divisions: dict[str, str] | None = None,
                major: str | None = None) -> tuple[pd.Series, float, float]:
    """Ratings in Elo-like points, home-field points, and the division gap."""
    validate(games)
    divisions = divisions if divisions is not None else team_divisions(games)
    teams = sorted(set(games["home_team_name"]) | set(games["away_team_name"]))
    return _solve(games, teams, divisions, major, C, None)


def prior_weight(week: int, w0: float, tau: float) -> float:
    """How much one last-season game counts, given how deep into this season we are."""
    return float(w0 * np.exp(-max(week, 0) / tau))


def fit_with_prior(current: pd.DataFrame, prior: pd.DataFrame | None, week: int,
                   C: float = 2.0, w0: float = 1.0, tau: float = 8.0,
                   major: str | None = None) -> tuple[pd.Series, float, float]:
    """Fit this season's games so far, anchored by last season's at a decayed weight.

    Rather than blending two rating vectors after the fact, last season's GAMES are
    refit alongside this season's at a sample weight below 1. That keeps strength of
    schedule intact instead of collapsing each team's season into one number, and it is
    a single solve rather than two plus a merge rule.

    `current` should already be filtered to games played before `week`.
    """
    has_prior = prior is not None and not prior.empty
    has_current = current is not None and not current.empty

    if not has_prior:
        return fit_ratings(current, C=C, major=major)
    if not has_current:
        # Preseason: last year is all the evidence there is, so use it undecayed.
        return fit_ratings(prior, C=C, major=major)

    w = prior_weight(week, w0, tau)
    combined = pd.concat([prior, current], ignore_index=True)
    weights = np.concatenate([np.full(len(prior), w), np.ones(len(current))])

    validate(combined)
    divisions = team_divisions(combined)
    teams = sorted(set(combined["home_team_name"]) | set(combined["away_team_name"]))
    return _solve(combined, teams, divisions, major, C, weights)


def bootstrap_ranks(games: pd.DataFrame, n_boot: int = 200, C: float = 2.0,
                    seed: int = 42, divisions: dict[str, str] | None = None,
                    major: str | None = None) -> pd.DataFrame:
    """Resample games with replacement and refit, giving a rank distribution per team."""
    rng = np.random.default_rng(seed)
    divisions = divisions if divisions is not None else team_divisions(games)
    ranks: dict[str, list[int]] = {}

    for b in range(n_boot):
        sample = games.iloc[rng.integers(0, len(games), len(games))]
        try:
            s, _, _ = fit_ratings(sample, C=C, divisions=divisions, major=major)
        except Exception:
            continue
        for rank, team in enumerate(s.index, start=1):
            ranks.setdefault(team, []).append(rank)
        if (b + 1) % 50 == 0:
            logger.info("bootstrap %d/%d", b + 1, n_boot)

    if not ranks:
        return pd.DataFrame(columns=["rank_p05", "rank_p50", "rank_p95", "n_boot"])

    return pd.DataFrame({
        "team": list(ranks),
        "rank_p05": [int(np.percentile(v, 5)) for v in ranks.values()],
        "rank_p50": [int(np.percentile(v, 50)) for v in ranks.values()],
        "rank_p95": [int(np.percentile(v, 95)) for v in ranks.values()],
        "n_boot": [len(v) for v in ranks.values()],
    }).set_index("team")


def win_prob(strength_a: float, strength_b: float) -> float:
    """Neutral-field win probability from two Elo-scale strengths."""
    return 1.0 / (1.0 + 10 ** ((strength_b - strength_a) / 400.0))


def score_games(strengths: pd.Series, home_adv: float, test: pd.DataFrame):
    """Log loss of a rating set on a set of games. (None, 0) if degenerate."""
    from sklearn.metrics import log_loss

    ps, ys = [], []
    for g in test.itertuples(index=False):
        if g.home_team_name not in strengths.index or g.away_team_name not in strengths.index:
            continue
        diff = strengths[g.home_team_name] - strengths[g.away_team_name]
        if not g.neutral_site:
            diff += home_adv
        ps.append(win_prob(diff, 0.0))
        ys.append(g.home_won)
    if len(ys) < 10 or len(set(ys)) < 2:
        return None, 0
    return float(log_loss(ys, np.clip(ps, 1e-6, 1 - 1e-6))), len(ys)


def transitivity_check(games: pd.DataFrame, C: float = 2.0,
                       major: str | None = None, top_n: int = 25) -> dict:
    """Test the premise a ranked list makes: do higher teams actually beat lower ones?

    A rating always *predicts* the higher-rated team wins, so the claim is only
    interesting measured against real results. This counts games won by the lower-rated
    side, which is the honest measure of how well any linear order can describe a
    season — and the number to quote when someone treats a ranking as a ground truth.

    Ported from the original college implementation and generalized; the upset rate is
    exactly where football and baseball diverge.
    """
    strengths, _, _ = fit_ratings(games, C=C, major=major)
    rank = {team: i for i, team in enumerate(strengths.index, start=1)}

    upsets = total = top_beaten_by_outsider = 0
    for g in games.itertuples(index=False):
        home_rank, away_rank = rank.get(g.home_team_name), rank.get(g.away_team_name)
        if home_rank is None or away_rank is None:
            continue
        total += 1
        winner_rank = home_rank if g.home_won else away_rank
        loser_rank = away_rank if g.home_won else home_rank
        if winner_rank > loser_rank:
            upsets += 1
            if loser_rank <= top_n and winner_rank > top_n:
                top_beaten_by_outsider += 1

    return {
        "games": total,
        "upsets": upsets,
        "upset_rate": round(upsets / total, 4) if total else None,
        "top_beaten_by_outsider": top_beaten_by_outsider,
    }
