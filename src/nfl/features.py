"""NFL v1 feature construction.

Every feature is computed causally: a game's row sees only games that finished before
it. The whole table is built in one chronological pass so there is no way for a future
result to leak backwards.

Elo and Pythagorean are ported from the MLB builders (build_v8_features.py) with
football constants. Everything else is the point-differential/streak/context block
with NFL-appropriate windows — MLB's 10g/30g spans half an NFL season, so 3g/8g.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from config import (
    ELO_HOME_BONUS,
    ELO_K,
    ELO_SEASON_REGRESSION,
    ELO_START,
    PYTHAG_EXPONENT,
)

logger = logging.getLogger(__name__)

SHORT_WINDOW = 3
LONG_WINDOW = 8


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Probability A beats B. Sport-agnostic; ported verbatim from the MLB builder."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def pythagorean_win_pct(scored: float, allowed: float, exponent: float = PYTHAG_EXPONENT) -> float:
    if scored <= 0 and allowed <= 0:
        return 0.5
    denom = scored ** exponent + allowed ** exponent
    if denom == 0:
        return 0.5
    return scored ** exponent / denom


EPA_METRICS = (
    "off_epa_play", "def_epa_play", "off_pass_epa", "off_rush_epa",
    "def_pass_epa", "def_rush_epa", "off_success_rate", "def_success_rate",
    "off_explosive_rate", "def_explosive_rate",
)


class EloParams:
    """Per-sport Elo settings. NFL defaults come from config; CFB overrides them —
    blowouts are routine in college, so margin-of-victory damping is required or
    ratings run away, and roster turnover forces heavier season regression."""

    __slots__ = ("start", "k", "home_bonus", "season_regression",
                 "pythag_exponent", "mov_damping")

    def __init__(self, start=ELO_START, k=ELO_K, home_bonus=ELO_HOME_BONUS,
                 season_regression=ELO_SEASON_REGRESSION,
                 pythag_exponent=PYTHAG_EXPONENT, mov_damping=False):
        self.start = start
        self.k = k
        self.home_bonus = home_bonus
        self.season_regression = season_regression
        self.pythag_exponent = pythag_exponent
        self.mov_damping = mov_damping


DEFAULT_ELO = EloParams()


def mov_multiplier(margin: float, elo_diff: float) -> float:
    """FiveThirtyEight-style margin-of-victory multiplier with autocorrelation control.
    Keeps a 60-point college blowout from moving a rating six times as far as a
    10-point win, and damps the favourite's gain so ratings stay bounded."""
    return np.log(abs(margin) + 1.0) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))


class _TeamState:
    """Rolling per-team state, updated only after a game is consumed."""

    __slots__ = ("elo", "pf", "pa", "results", "margins", "streak",
                 "last_date", "games", "epa")

    def __init__(self) -> None:
        self.elo = ELO_START
        self.pf = 0.0          # season points for
        self.pa = 0.0          # season points against
        self.results = deque(maxlen=LONG_WINDOW)   # 1/0 most recent last
        self.margins = deque(maxlen=LONG_WINDOW)   # point differential
        self.streak = 0        # +n winning, -n losing
        self.last_date = None
        self.games = 0
        # One deque per EPA metric; carries across the season boundary is reset below.
        self.epa = {m: deque(maxlen=LONG_WINDOW) for m in EPA_METRICS}

    def new_season(self, regression: float = ELO_SEASON_REGRESSION,
                   start: float = ELO_START) -> None:
        self.elo = start + (self.elo - start) * (1.0 - regression)
        self.pf = 0.0
        self.pa = 0.0
        self.results.clear()
        self.margins.clear()
        self.streak = 0
        self.games = 0
        self.last_date = None
        for dq in self.epa.values():
            dq.clear()

    def _mean(self, dq: deque, n: int) -> float:
        if not dq:
            return 0.0
        window = list(dq)[-n:]
        return float(np.mean(window))

    def snapshot(self, prefix: str) -> dict:
        pyth = pythagorean_win_pct(self.pf, self.pa) if self.games else 0.5
        win_pct = float(np.mean(self.results)) if self.results else 0.5
        return {
            f"{prefix}_elo": self.elo,
            f"{prefix}_pythag_season": pyth,
            f"{prefix}_point_diff_3g": self._mean(self.margins, SHORT_WINDOW),
            f"{prefix}_point_diff_8g": self._mean(self.margins, LONG_WINDOW),
            f"{prefix}_points_scored_pg": self.pf / self.games if self.games else 0.0,
            f"{prefix}_points_allowed_pg": self.pa / self.games if self.games else 0.0,
            f"{prefix}_win_pct_season": win_pct,
            f"{prefix}_current_streak": self.streak,
            f"{prefix}_streak_magnitude": abs(self.streak),
            f"{prefix}_on_winning_streak": int(self.streak >= 2),
            f"{prefix}_on_losing_streak": int(self.streak <= -2),
            f"{prefix}_games_played": self.games,
            **self._epa_snapshot(prefix),
        }

    def _epa_snapshot(self, prefix: str) -> dict:
        """Rolling EPA over the last 3 and 8 games. 0.0 when a team has no history yet
        (start of the first season, or expansion) — EPA is centred near zero, so that
        is a neutral prior rather than an invented value."""
        out = {}
        for m in EPA_METRICS:
            dq = self.epa[m]
            out[f"{prefix}_{m}_3g"] = self._mean(dq, SHORT_WINDOW)
            out[f"{prefix}_{m}_8g"] = self._mean(dq, LONG_WINDOW)
        return out

    def push_epa(self, metrics: dict) -> None:
        for m in EPA_METRICS:
            v = metrics.get(m)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                self.epa[m].append(float(v))

    def update(self, scored: float, allowed: float, won: int) -> None:
        self.pf += scored
        self.pa += allowed
        self.results.append(won)
        self.margins.append(scored - allowed)
        if won:
            self.streak = self.streak + 1 if self.streak >= 0 else 1
        else:
            self.streak = self.streak - 1 if self.streak <= 0 else -1
        self.games += 1


def build_features(games: pd.DataFrame, epa: pd.DataFrame | None = None,
                   elo: "EloParams | None" = None) -> pd.DataFrame:
    """One chronological pass producing a causal feature row per game.

    `epa` is the team-week aggregate from epa.build_team_week_epa(). When supplied,
    each team's rolling EPA is folded in — read before the game is emitted, updated
    only after, exactly like the score-based state.
    """
    elo = elo or DEFAULT_ELO
    games = games.sort_values(["season", "week", "game_date"]).reset_index(drop=True)

    epa_lookup: dict[tuple, dict] = {}
    if epa is not None and len(epa):
        for rec in epa.to_dict("records"):
            epa_lookup[(rec["season"], rec["week"], rec["team"])] = rec

    state: dict[str, _TeamState] = defaultdict(_TeamState)
    h2h: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=6))
    current_season = None
    rows = []

    for g in games.itertuples(index=False):
        if g.season != current_season:
            for st in state.values():
                st.new_season(elo.season_regression, elo.start)
            current_season = g.season

        home, away = g.home_team, g.away_team
        hs, as_ = state[home], state[away]

        row = {
            "game_id": g.game_id,
            "season": g.season,
            "week": g.week,
            "game_date": g.game_date,
            "home_team": home,
            "away_team": away,
            "home_won": g.home_won,
        }
        row.update(hs.snapshot("home"))
        row.update(as_.snapshot("away"))

        # --- Elo ---
        h_elo = hs.elo + elo.home_bonus
        row["elo_differential"] = h_elo - as_.elo
        row["elo_home_win_prob"] = elo_expected(h_elo, as_.elo)

        # --- differentials ---
        row["pythag_differential"] = row["home_pythag_season"] - row["away_pythag_season"]
        row["point_diff_differential"] = row["home_point_diff_8g"] - row["away_point_diff_8g"]

        # EPA differentials — the matchup framing the model actually needs:
        # my offence against your defence, both directions.
        if epa_lookup:
            row["net_epa_8g"] = (
                (row["home_off_epa_play_8g"] - row["home_def_epa_play_8g"])
                - (row["away_off_epa_play_8g"] - row["away_def_epa_play_8g"])
            )
            row["net_epa_3g"] = (
                (row["home_off_epa_play_3g"] - row["home_def_epa_play_3g"])
                - (row["away_off_epa_play_3g"] - row["away_def_epa_play_3g"])
            )
            row["home_off_vs_away_def_8g"] = (
                row["home_off_epa_play_8g"] - row["away_def_epa_play_8g"]
            )
            row["away_off_vs_home_def_8g"] = (
                row["away_off_epa_play_8g"] - row["home_def_epa_play_8g"]
            )
            row["success_rate_differential"] = (
                row["home_off_success_rate_8g"] - row["away_off_success_rate_8g"]
            )
            row["explosive_differential"] = (
                row["home_off_explosive_rate_8g"] - row["away_off_explosive_rate_8g"]
            )
        row["point_diff_short_differential"] = row["home_point_diff_3g"] - row["away_point_diff_3g"]
        row["win_pct_diff"] = row["home_win_pct_season"] - row["away_win_pct_season"]
        row["streak_differential"] = row["home_current_streak"] - row["away_current_streak"]
        row["points_allowed_differential"] = (
            row["away_points_allowed_pg"] - row["home_points_allowed_pg"]
        )

        # --- head to head (last 6 meetings, ~3 years for division rivals) ---
        key = tuple(sorted((home, away)))
        prior = list(h2h[key])
        if prior:
            wins = sum(1 for w in prior if w == home)
            row["h2h_win_pct"] = wins / len(prior)
            row["h2h_games"] = len(prior)
        else:
            row["h2h_win_pct"] = 0.5
            row["h2h_games"] = 0

        # --- schedule context (free from nflverse) ---
        row["home_rest"] = float(getattr(g, "home_rest", np.nan) or np.nan)
        row["away_rest"] = float(getattr(g, "away_rest", np.nan) or np.nan)
        row["rest_advantage"] = row["home_rest"] - row["away_rest"]
        row["home_short_week"] = int(row["home_rest"] <= 4)
        row["away_short_week"] = int(row["away_rest"] <= 4)
        row["home_off_bye"] = int(row["home_rest"] >= 13)
        row["away_off_bye"] = int(row["away_rest"] >= 13)
        row["is_divisional"] = int(getattr(g, "div_game", 0) or 0)
        row["is_playoff"] = int(g.game_type != "REG")
        row["is_neutral_site"] = int(str(getattr(g, "location", "Home")) != "Home")

        # market columns kept alongside but excluded from the `pure` feature set
        row["spread_line"] = getattr(g, "spread_line", np.nan)
        row["total_line"] = getattr(g, "total_line", np.nan)

        rows.append(row)

        # --- consume the result (strictly after the row is emitted) ---
        # Scheduled-but-unplayed games carry no result: emit the feature row so the
        # week can be predicted, but leave team state untouched.
        if pd.isna(getattr(g, "home_won", np.nan)):
            continue

        home_pts, away_pts = g.home_score, g.away_score
        hs.update(home_pts, away_pts, g.home_won)
        as_.update(away_pts, home_pts, 1 - g.home_won)
        h2h[key].append(home if g.home_won else away)

        if epa_lookup:
            hm = epa_lookup.get((g.season, g.week, home))
            am = epa_lookup.get((g.season, g.week, away))
            if hm:
                hs.push_epa(hm)
            if am:
                as_.push_epa(am)

        exp = elo_expected(h_elo, as_.elo)
        shift = elo.k * (g.home_won - exp)
        if elo.mov_damping:
            shift *= mov_multiplier(home_pts - away_pts, h_elo - as_.elo)
        hs.elo += shift
        as_.elo -= shift

    out = pd.DataFrame(rows)
    logger.info("built %d feature rows, %d columns", len(out), out.shape[1])
    return out


# Feature sets. `PURE` deliberately excludes every market-derived column.
NON_FEATURE = {
    "game_id", "season", "week", "game_date", "home_team", "away_team", "home_won",
    "spread_line", "total_line",
}


def feature_columns(df: pd.DataFrame, include_market: bool = False) -> list[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE]
    if include_market:
        cols += ["spread_line", "total_line"]
    return cols
