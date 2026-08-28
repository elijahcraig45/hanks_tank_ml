"""nflverse ingestion with a local raw cache.

nflverse serves parquet from GitHub release assets with no SLA, and this sits inside a
weekly cron. Every successful pull is cached to disk so a GitHub outage degrades to
last-good rather than failing the run. In production the same cache lands in GCS.

Returns pandas, not polars: nflreadpy is polars-native but the training harness copied
from the MLB side is pandas-based, so the conversion boundary lives here, once.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import DATA_DIR, FIRST_SEASON, RAW_CACHE

logger = logging.getLogger(__name__)

SCHEDULES_CACHE = RAW_CACHE / "schedules.parquet"


def _fetch_schedules() -> pd.DataFrame:
    import nflreadpy as nfl

    df = nfl.load_schedules().to_pandas()
    logger.info("fetched %d schedule rows from nflverse", len(df))
    return df


def load_schedules(use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    """Full nflverse schedule table, cached locally.

    Falls back to the cache if the network fetch fails — the whole point of the cache.
    """
    RAW_CACHE.mkdir(parents=True, exist_ok=True)

    if refresh or not (use_cache and SCHEDULES_CACHE.exists()):
        try:
            df = _fetch_schedules()
            df.to_parquet(SCHEDULES_CACHE, index=False)
            return df
        except Exception as exc:
            if SCHEDULES_CACHE.exists():
                logger.warning("nflverse fetch failed (%s); using cached copy", exc)
                return pd.read_parquet(SCHEDULES_CACHE)
            raise

    return pd.read_parquet(SCHEDULES_CACHE)


def completed_games(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Decided games only, with the target column attached.

    `result` is home_score - away_score. NFL ties (result == 0) are dropped: they are
    ~0.2% of games and a binary home_won target has nowhere to put them.
    """
    if df is None:
        df = load_schedules()

    played = df[df["result"].notna()].copy()
    decided = played[played["result"] != 0].copy()

    dropped = len(played) - len(decided)
    if dropped:
        logger.info("dropped %d tie games", dropped)

    decided["home_won"] = (decided["result"] > 0).astype(int)
    decided["game_date"] = pd.to_datetime(decided["gameday"])
    decided = decided[decided["season"] >= FIRST_SEASON]

    return decided.sort_values(["season", "week", "game_date"]).reset_index(drop=True)
