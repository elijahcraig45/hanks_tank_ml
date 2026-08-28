"""A "true" college football top 25 — Bradley-Terry strength ratings.

Why not Elo: Elo is path-dependent. It walks games in order, regresses toward the mean
at season boundaries, and weights late results more than early ones. Two teams with
identical resumes can end up rated differently purely because of scheduling order.
That is fine for forecasting and wrong for "who is actually best".

This fits a Bradley-Terry model instead: one global maximum-likelihood solve over every
game in the season at once. Each team gets a single strength parameter, home-field is a
separate covariate, and the result is order-independent — shuffle the schedule and you
get the same ratings.

Design matrix, one row per game:
    +1 in the home team's column, -1 in the away team's column,
    +1 in a home_field column (0 for neutral sites)
    target = home team won

The fitted coefficients are team strengths on a log-odds scale, so the difference
between any two teams converts directly to a win probability with no extra assumptions:
    P(A beats B on a neutral field) = sigmoid(strength_A - strength_B)

FBS and FCS are fitted TOGETHER. The schedule graph is weakly connected — most teams
never play most other teams — and cross-division games are the edges that tie the two
populations to a common scale. Fitting them separately would make the two ladders
incomparable.

Ridge regularization is not optional here. With ~380 teams and ~12 games each, unbeaten
and winless teams produce infinite maximum-likelihood estimates (complete separation).
The penalty shrinks them toward average, which is also the honest answer: going 12-0
against weak opposition is weaker evidence than the raw record suggests.

Rank uncertainty comes from a bootstrap over games — resample the season, refit, and
record where each team lands. That is what answers "how close are they": a team whose
rank swings between 3 and 19 across resamples is not meaningfully separated from its
neighbours, however confident the point estimate looks.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from espn_data import load_games  # noqa: E402

logger = logging.getLogger(__name__)

# Convert log-odds strengths to a familiar Elo-like point scale (400 per 10x odds).
ELO_SCALE = 400.0 / np.log(10.0)

# Ridge strength, chosen by 5-fold CV on held-out game outcomes (see cv_ridge()).
# 2.0 minimizes log loss at 0.5128 / 74.3% accuracy. Stronger penalties looked
# "safer" but were flattening genuine separation: at C=0.35 every team from #2 to #25
# collapsed into a 68-point band of coin flips, which is a modelling artifact rather
# than a fact about the season.
DEFAULT_C = 2.0


# The division term is deliberately under-penalized relative to team terms. Only ~7.5%
# of games cross divisions, so ridge would shrink this coefficient toward zero and let
# FCS teams drift onto the FBS scale — which is exactly the failure mode where a team
# that never played an FBS opponent lands in the top 10. Scaling the column up by this
# factor (and dividing the coefficient back out) weakens its effective penalty ~f^2.
DIV_SCALE = 12.0


def team_divisions(games: pd.DataFrame) -> dict[str, str]:
    d = {}
    for g in games.itertuples(index=False):
        d[g.home_team_name] = g.division
        d[g.away_team_name] = g.division
    return d


def _design(games: pd.DataFrame, teams: list[str],
            divisions: dict[str, str]) -> tuple[csr_matrix, np.ndarray]:
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(games), len(teams)

    rows, cols, vals = [], [], []
    div_col = np.zeros(n)
    for r, g in enumerate(games.itertuples(index=False)):
        rows += [r, r]
        cols += [idx[g.home_team_name], idx[g.away_team_name]]
        vals += [1.0, -1.0]
        # +1 when the home team is the FBS side of a cross-division game, -1 when away.
        h = 1.0 if divisions.get(g.home_team_name) == "fbs" else 0.0
        a = 1.0 if divisions.get(g.away_team_name) == "fbs" else 0.0
        div_col[r] = (h - a) * DIV_SCALE

    X_teams = csr_matrix((vals, (rows, cols)), shape=(n, k))
    # Home-field only applies when the game actually had a home team.
    home_col = csr_matrix((1.0 - games["neutral_site"].to_numpy(dtype=float)).reshape(-1, 1))
    X = hstack([X_teams, home_col, csr_matrix(div_col.reshape(-1, 1))], format="csr")
    y = games["home_won"].to_numpy(dtype=int)
    return X, y


def fit_ratings(games: pd.DataFrame, C: float = DEFAULT_C,
                divisions: dict[str, str] | None = None
                ) -> tuple[pd.Series, float, float]:
    """Return (ratings in Elo-like points, home-field points, FBS-over-FCS points).

    Team coefficients are strengths *within* their division; the returned rating adds
    the fitted division baseline back on, so FBS and FCS sit on one comparable scale.
    """
    divisions = divisions or team_divisions(games)
    teams = sorted(set(games["home_team_name"]) | set(games["away_team_name"]))
    X, y = _design(games, teams, divisions)

    model = LogisticRegression(
        C=C, fit_intercept=False, solver="lbfgs", max_iter=5000,
    )
    model.fit(X, y)

    coef = model.coef_[0]
    within = pd.Series(coef[:len(teams)] * ELO_SCALE, index=teams)
    home_adv = float(coef[len(teams)] * ELO_SCALE)
    div_gap = float(coef[len(teams) + 1] * DIV_SCALE * ELO_SCALE)

    # Put both ladders on one scale by adding each team's division baseline.
    baseline = pd.Series(
        [div_gap if divisions.get(t) == "fbs" else 0.0 for t in teams], index=teams
    )
    strengths = within + baseline
    strengths = strengths - strengths.mean()
    return strengths.sort_values(ascending=False), home_adv, div_gap


# --------------------------------------------------------------------------
# Carrying last season forward as a decaying prior
# --------------------------------------------------------------------------
#
# A single college season cannot identify 315 teams: ~12 games each on a schedule graph
# where most teams never meet. In week 3 a pure current-season fit is mostly noise.
# Last season is the natural prior — same programs, and its strength of schedule is
# already encoded in real games.
#
# Rather than blending two rating vectors after the fact, last season's GAMES are refit
# alongside this season's with a sample weight below 1. That keeps strength of schedule
# intact instead of collapsing each team's season into a single number, and it is one
# solve rather than two plus a merge rule.
#
#     w(week) = PRIOR_W0 * exp(-week / PRIOR_TAU)
#
# Both constants were tuned by walk-forward log loss across FOUR seasons (2022-2025),
# not one — the 2025-only optimum was tau=12, which degraded on the other three. Roster
# turnover is handled by the decay itself rather than by discounting week 0, which is
# right: in the preseason last year is all the evidence there is.
#
#   week 0 -> 1.00   week 4 -> 0.61   week 8 -> 0.37   week 12 -> 0.22
#
# Setting tau=40 (i.e. effectively no decay, just pool both seasons) scores 0.5500
# against 0.5420 here, so the decay is doing real work and not just decoration.
PRIOR_W0 = 1.0
PRIOR_TAU = 8.0


def prior_weight(week: int, w0: float = PRIOR_W0, tau: float = PRIOR_TAU) -> float:
    """How much one last-season game counts, given how deep into this season we are."""
    return float(w0 * np.exp(-max(week, 0) / tau))


def fit_with_prior(current: pd.DataFrame, prior: pd.DataFrame | None, week: int,
                   C: float = DEFAULT_C, w0: float = PRIOR_W0, tau: float = PRIOR_TAU,
                   ) -> tuple[pd.Series, float, float]:
    """Fit on this season's games so far, anchored by last season's at a decayed weight.

    `current` should already be filtered to games played before `week`.
    """
    if prior is None or prior.empty:
        return fit_ratings(current, C=C)
    if current is None or current.empty:
        return fit_ratings(prior, C=C)

    w = prior_weight(week, w0, tau)
    combined = pd.concat([prior, current], ignore_index=True)
    weights = np.concatenate([np.full(len(prior), w), np.ones(len(current))])

    divisions = team_divisions(combined)
    teams = sorted(set(combined["home_team_name"]) | set(combined["away_team_name"]))
    X, y = _design(combined, teams, divisions)

    model = LogisticRegression(C=C, fit_intercept=False, solver="lbfgs", max_iter=5000)
    model.fit(X, y, sample_weight=weights)

    coef = model.coef_[0]
    within = pd.Series(coef[:len(teams)] * ELO_SCALE, index=teams)
    home_adv = float(coef[len(teams)] * ELO_SCALE)
    div_gap = float(coef[len(teams) + 1] * DIV_SCALE * ELO_SCALE)

    baseline = pd.Series(
        [div_gap if divisions.get(t) == "fbs" else 0.0 for t in teams], index=teams
    )
    strengths = (within + baseline)
    strengths = strengths - strengths.mean()
    return strengths.sort_values(ascending=False), home_adv, div_gap


def _score_week(strengths: pd.Series, home_adv: float, test: pd.DataFrame):
    """Log loss of a rating set on one week's games. None if the week is degenerate."""
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
    return log_loss(ys, np.clip(ps, 1e-6, 1 - 1e-6)), len(ys)


def validate_prior(season: int = 2025, prior_season: int | None = None,
                   C: float = DEFAULT_C, w0: float = PRIOR_W0,
                   tau: float = PRIOR_TAU) -> pd.DataFrame:
    """Week-by-week honest test: does the prior actually predict better?

    For each week, fit on everything strictly before it and score that week's games.
    Three fits are compared — current-season-only, prior-blended, and prior-only. The
    last is the "just use last year" baseline the blend has to beat to be worth having.
    """
    games = load_games()
    cur_all = games[games["season"] == season]
    prior = games[games["season"] == (prior_season or season - 1)]
    if prior.empty:
        raise SystemExit(f"no cached games for prior season {prior_season or season - 1}")

    s_p, ha_p, _ = fit_ratings(prior, C=C)  # prior-only fit is constant across weeks

    rows = []
    for week in sorted(cur_all["week"].unique()):
        test = cur_all[cur_all["week"] == week]
        seen = cur_all[cur_all["week"] < week]
        if len(test) < 15:
            continue

        entry = {"week": int(week), "prior_w": round(prior_weight(week, w0, tau), 3)}
        entry["prior_only"], n = _score_week(s_p, ha_p, test)
        entry["games"] = n

        if len(seen) >= 40:
            s_c, ha_c, _ = fit_ratings(seen, C=C)
            entry["current_only"], _ = _score_week(s_c, ha_c, test)
        else:
            entry["current_only"] = None

        s_b, ha_b, _ = fit_with_prior(seen, prior, week, C=C, w0=w0, tau=tau)
        entry["blended"], _ = _score_week(s_b, ha_b, test)
        rows.append(entry)

    return pd.DataFrame(rows)


def bootstrap_ranks(games: pd.DataFrame, n_boot: int = 200, C: float = DEFAULT_C,
                    seed: int = 42, divisions: dict[str, str] | None = None) -> pd.DataFrame:
    """Resample games with replacement and refit, to get a rank distribution per team."""
    rng = np.random.default_rng(seed)
    divisions = divisions or team_divisions(games)
    ranks: dict[str, list[int]] = {}

    for b in range(n_boot):
        sample = games.iloc[rng.integers(0, len(games), len(games))]
        try:
            s, _, _ = fit_ratings(sample, C=C, divisions=divisions)
        except Exception:
            continue
        for rank, team in enumerate(s.index, start=1):
            ranks.setdefault(team, []).append(rank)
        if (b + 1) % 50 == 0:
            logger.info("bootstrap %d/%d", b + 1, n_boot)

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


def build_top25(season: int = 2025, n_boot: int = 200, C: float = DEFAULT_C,
                week: int | None = None, use_prior: bool = True
                ) -> tuple[pd.DataFrame, dict]:
    """Top 25 as of `week` (default: end of season).

    With use_prior, last season's games are refit alongside at a decaying weight, which
    is what makes an early-season ranking meaningful instead of noise. Validated
    week-by-week across 2022-2025: blended log loss 0.5420 vs 0.5866 current-only.
    """
    games = load_games()
    season_games = games[games["season"] == season].copy()
    if season_games.empty:
        raise SystemExit(f"no games cached for {season}")

    if week is not None:
        season_games = season_games[season_games["week"] < week]
        effective_week = week
    else:
        effective_week = int(games[games["season"] == season]["week"].max())

    prior = games[games["season"] == season - 1] if use_prior else None
    if prior is not None and prior.empty:
        prior = None

    if prior is not None:
        strengths, home_adv, div_gap = fit_with_prior(
            season_games, prior, effective_week, C=C)
        # Bootstrap resamples the actual evidence set, prior included, so the rank
        # ranges reflect the same information the point estimate used.
        boot_pool = pd.concat([prior, season_games], ignore_index=True)
    else:
        strengths, home_adv, div_gap = fit_ratings(season_games, C=C)
        boot_pool = season_games

    divisions = team_divisions(pd.concat(
        [p for p in (prior, season_games) if p is not None], ignore_index=True))
    boot = bootstrap_ranks(boot_pool, n_boot=n_boot, C=C, divisions=divisions)

    division_of = team_divisions(season_games) or divisions

    record = {}
    for g in season_games.itertuples(index=False):
        w, l = ((g.home_team_name, g.away_team_name) if g.home_won
                else (g.away_team_name, g.home_team_name))
        record.setdefault(w, [0, 0])[0] += 1
        record.setdefault(l, [0, 0])[1] += 1

    top = strengths.head(26)  # 26 so #25 has a "beat the next team" comparison
    rows = []
    for rank, (team, rating) in enumerate(top.items(), start=1):
        nxt = top.iloc[rank] if rank < len(top) else None
        w, l = record.get(team, [0, 0])
        rows.append({
            "rank": rank,
            "team": team,
            "division": division_of.get(team),
            "record": f"{w}-{l}",
            "rating": round(float(rating), 1),
            "gap_to_next": round(float(rating - nxt), 1) if nxt is not None else None,
            "p_beat_next": round(win_prob(rating, nxt), 3) if nxt is not None else None,
            "p_beat_no26": round(win_prob(rating, float(top.iloc[-1])), 3),
            "rank_p05": int(boot.loc[team, "rank_p05"]) if team in boot.index else None,
            "rank_p50": int(boot.loc[team, "rank_p50"]) if team in boot.index else None,
            "rank_p95": int(boot.loc[team, "rank_p95"]) if team in boot.index else None,
        })

    meta = {
        "season": season,
        "games": len(season_games),
        "teams": len(strengths),
        "home_field_points": round(home_adv, 1),
        "fbs_over_fcs_points": round(div_gap, 1),
        "ridge_C": C,
        "n_boot": n_boot,
        "week": effective_week,
        "prior_weight": round(prior_weight(effective_week), 3) if prior is not None else 0.0,
    }
    return pd.DataFrame(rows).head(25), meta


def transitivity_check(season: int = 2025, C: float = DEFAULT_C) -> dict:
    """Test the premise: do these teams actually beat everyone below them?

    A rating always *predicts* the higher team wins, so the claim is only interesting
    against real results. This counts the games where a lower-rated team beat a
    higher-rated one, which is the honest measure of how well any linear order can
    describe a season.
    """
    games = load_games()
    s = games[games["season"] == season].copy()
    strengths, _, _ = fit_ratings(s, C=C)
    rank = {t: i for i, t in enumerate(strengths.index, start=1)}

    upsets = total = top25_losses_to_outside = 0
    for g in s.itertuples(index=False):
        rh, ra = rank.get(g.home_team_name), rank.get(g.away_team_name)
        if rh is None or ra is None:
            continue
        total += 1
        winner_rank = rh if g.home_won else ra
        loser_rank = ra if g.home_won else rh
        if winner_rank > loser_rank:
            upsets += 1
            if loser_rank <= 25 and winner_rank > 25:
                top25_losses_to_outside += 1

    return {
        "games": total,
        "upsets": upsets,
        "upset_rate": round(upsets / total, 4) if total else None,
        "top25_beaten_by_outsider": top25_losses_to_outside,
    }


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--week", type=int, default=None,
                    help="rank as of this week (default: end of season)")
    ap.add_argument("--no-prior", action="store_true",
                    help="ignore last season entirely")
    ap.add_argument("--write-bq", action="store_true")
    args = ap.parse_args()

    table, meta = build_top25(args.season, n_boot=args.boot,
                              week=args.week, use_prior=not args.no_prior)

    print()
    print(f"TRUE TOP 25 — {meta['season']}  (Bradley-Terry, {meta['games']} games, "
          f"{meta['teams']} teams, FBS+FCS fitted together)")
    print(f"home-field {meta['home_field_points']} pts | "
          f"FBS-over-FCS {meta['fbs_over_fcs_points']} pts | "
          f"last season weighted {meta['prior_weight']:.2f} at week {meta['week']}")
    print("=" * 100)
    print(f"{'#':>3} {'team':<32} {'div':<4} {'rec':>6} {'rating':>7} "
          f"{'gap':>6} {'P(beat next)':>12} {'P(beat #26)':>11}  {'rank 5-95%':>11}")
    print("-" * 100)
    for r in table.itertuples(index=False):
        rng = f"{r.rank_p05}-{r.rank_p95}" if r.rank_p05 is not None else "—"
        print(f"{r.rank:>3} {r.team[:32]:<32} {r.division:<4} {r.record:>6} "
              f"{r.rating:>7.1f} {(r.gap_to_next or 0):>6.1f} "
              f"{(r.p_beat_next or 0):>12.1%} {r.p_beat_no26:>11.1%}  {rng:>11}")

    check = transitivity_check(args.season)
    print()
    print("Does the premise hold? — 'these 25 would beat anyone below them'")
    print(f"  games with both teams rated : {check['games']}")
    print(f"  won by the LOWER-rated team : {check['upsets']} ({check['upset_rate']:.1%})")
    print(f"  top-25 team beaten by a team outside the top 25: "
          f"{check['top25_beaten_by_outsider']}")

    if args.write_bq:
        import cfb_config
        from backfill_cfb import ensure_datasets, load

        out = table.copy()
        out["season"] = meta["season"]
        out["as_of_week"] = meta["week"]
        out["prior_weight"] = meta["prior_weight"]
        out["home_field_points"] = meta["home_field_points"]
        out["fbs_over_fcs_points"] = meta["fbs_over_fcs_points"]
        ensure_datasets()
        load(out, cfb_config.CTX.season_dataset, "power_rankings")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
