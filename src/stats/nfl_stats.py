"""NFL player season stats from nflverse, plus leaders derived from them.

Unlike college — where no public ESPN endpoint returns a full per-player season table —
the NFL has nflverse, so leaders here are computed from the same rows the player lookup
serves. That is deliberate: a leaderboard that disagrees with the table beneath it is
worse than no leaderboard, and deriving both from one source makes that impossible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

try:  # package layout locally, flat module tree in Cloud Functions
    from rankings.http import get_bytes as _get_bytes, cache_dir as _cache_dir
except ImportError:  # pragma: no cover
    from http_transport import get_bytes as _get_bytes, cache_dir as _cache_dir

logger = logging.getLogger(__name__)

# Writable wherever this runs; /workspace is read-only in Cloud Functions.
CACHE = _cache_dir("stats")

RELEASE = "https://github.com/nflverse/nflverse-data/releases/download/stats_player"

IDENTITY = [
    "player_id", "player_display_name", "position", "position_group",
    "recent_team", "season", "season_type", "games", "headshot_url",
]

# Leaders worth a tab.
#
# `higher_is_better` is not decoration: interceptions thrown is the one category where
# leading is bad, and sorting it like yards would crown the league's worst quarterback.
#
# The qualifier gates on VOLUME, never on the ranked stat itself. Gating passing EPA on
# passing EPA would silently exclude every below-average passer; gating interceptions
# thrown on interceptions thrown is incoherent. Attempts and targets are what make a
# rate or a negative stat comparable.
LEADER_CATEGORIES = [
    # (column, label, higher_is_better, qualifier_column, qualifier_min)
    ("passing_yards", "Passing yards", True, "attempts", 100),
    ("passing_tds", "Passing TDs", True, "attempts", 100),
    ("passing_epa", "Passing EPA", True, "attempts", 150),
    # Ranked ascending, so the label has to say "fewest" — "Interceptions thrown"
    # with a 1 at the top reads as the opposite of what it means.
    ("passing_interceptions", "Fewest interceptions", False, "attempts", 150),
    ("rushing_yards", "Rushing yards", True, "carries", 40),
    ("rushing_tds", "Rushing TDs", True, "carries", 20),
    ("receiving_yards", "Receiving yards", True, "targets", 20),
    ("receptions", "Receptions", True, "targets", 20),
    ("receiving_tds", "Receiving TDs", True, "targets", 10),
    ("def_sacks", "Sacks", True, None, 0),
    ("def_tackles_solo", "Solo tackles", True, None, 0),
    ("def_interceptions", "Interceptions caught", True, None, 0),
    ("def_pass_defended", "Passes defended", True, None, 0),
]


def _curl(url: str, timeout: int = 90) -> bytes:
    """Delegates to the shared transport; see rankings.http for why."""
    return _get_bytes(url, timeout)


def fetch_player_stats(season: int, refresh: bool = False) -> pd.DataFrame:
    """Season-level regular-season player stats for one year.

    Returns an empty frame where the season has not been published yet. nflverse only
    creates the release asset once a season's first games are played, so a 404 in
    September is the normal pre-season state, not a failure — and reporting it as one
    every week would bury a real outage in expected noise.
    """
    cached = CACHE / f"nfl_player_stats_{season}.csv"
    if refresh or not cached.exists():
        try:
            cached.write_bytes(_curl(f"{RELEASE}/stats_player_reg_{season}.csv"))
        except Exception as exc:
            if cached.exists():
                logger.warning("nflverse fetch failed (%s); using cached copy", exc)
            elif "404" in str(exc):
                logger.info("nflverse has not published %d player stats yet", season)
                return pd.DataFrame()
            else:
                raise

    df = pd.read_csv(cached, low_memory=False)
    df["season"] = season
    logger.info("NFL player stats %d: %d players, %d columns",
                season, len(df), len(df.columns))
    return df


def leaders(stats: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    """Long-form leaderboard: one row per (category, rank)."""
    rows = []
    for column, label, higher_is_better, qualifier, minimum in LEADER_CATEGORIES:
        if column not in stats.columns:
            logger.debug("skipping %s: not in this feed", column)
            continue

        sub = stats[stats[column].notna()].copy()
        if qualifier and qualifier in sub.columns and minimum:
            sub = sub[pd.to_numeric(sub[qualifier], errors="coerce").fillna(0) >= minimum]
        # A zero in a counting stat is not an achievement; it would otherwise fill the
        # bottom of any short board.
        sub = sub[sub[column] != 0]
        if sub.empty:
            logger.warning("no qualifiers for %s", column)
            continue

        sub = sub.sort_values(column, ascending=not higher_is_better).head(top)
        for rank, r in enumerate(sub.itertuples(index=False), start=1):
            value = float(getattr(r, column))
            rows.append({
                "season": int(getattr(r, "season")),
                "sport": "nfl",
                "category": column,
                "category_label": label,
                "higher_is_better": higher_is_better,
                "rank": rank,
                "value": value,
                "display_value": f"{int(value)}" if value.is_integer() else f"{value:.2f}",
                "player_id": getattr(r, "player_id", None),
                "player_name": getattr(r, "player_display_name", None),
                "position": getattr(r, "position", None),
                "team": getattr(r, "recent_team", None),
            })

    df = pd.DataFrame(rows)
    for column in ("player_id", "player_name", "position", "team",
                   "category", "category_label", "display_value"):
        if column in df.columns:
            df[column] = df[column].astype("string")
    logger.info("NFL leaders: %d rows across %d categories",
                len(df), df["category"].nunique() if not df.empty else 0)
    return df
