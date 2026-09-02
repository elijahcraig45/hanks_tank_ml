"""Backfill NFL history into BigQuery.

Loads three tables:
  nfl_historical.games          one row per completed game, 1999-present
  nfl_historical.team_week_epa  per-team, per-week EPA aggregates (2006+)
  nfl_historical.teams          reference: abbreviations, colours, logos

Safe to re-run — every table is WRITE_TRUNCATE.

Usage:
  python backfill_nfl_history.py
  python backfill_nfl_history.py --skip-epa
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bq_io import ensure_dataset, load_table, replace_seasons  # noqa: E402
from config import CTX, FIRST_SEASON  # noqa: E402
from data import completed_games, load_schedules  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GAME_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
    "away_team", "away_score", "home_team", "home_score", "result", "total",
    "overtime", "home_rest", "away_rest", "div_game", "roof", "surface", "temp",
    "wind", "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "stadium", "stadium_id", "location", "home_won", "game_date",
]


def backfill_games() -> int:
    games = completed_games()
    cols = [c for c in GAME_COLUMNS if c in games.columns]
    df = games[cols].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    return load_table(
        df, CTX.hist_dataset, "games",
        partition_field="game_date",
        cluster_fields=["season", "home_team"],
    )


def backfill_epa(seasons: list[int] | None = None) -> int:
    """Load per-team, per-week EPA for `seasons`, defaulting to the full history.

    `seasons` exists because the weekly path must not rebuild the archive. Play-by-play
    is the heaviest thing this repo loads, and the cache it was relying on lives in
    /tmp — empty on every Cloud Function cold start. So the scheduled ingest was
    rebuilding twenty seasons of play-by-play on every run and dying at the 2GB limit:
    `Memory limit of 1953 MiB exceeded with 2018 MiB used`. It had never once written
    the current season.

    Two other things were wrong with the old version and are fixed here. The season
    range was hardcoded `range(2006, 2026)`, which excludes 2026 — so even a run that
    survived would not have added the current season. And the write truncated, so a
    single-season refresh would have replaced the whole table with one season.
    """
    import polars as pl
    from epa import EPA_CACHE, build_team_week_epa

    full_history = seasons is None
    if full_history:
        # Every season the model trains on, through the current one. Not a hardcoded
        # end year — that is how 2026 came to be excluded.
        seasons = list(range(2006, CTX.season + 1))

    if EPA_CACHE.exists():
        cached = pl.read_parquet(EPA_CACHE).to_pandas()
        df = cached[cached["season"].isin(seasons)]
        missing = sorted(set(seasons) - set(df["season"].unique().tolist()))
        if missing:
            logger.info("EPA cache missing %s; building those", missing)
            built = build_team_week_epa(missing).to_pandas()
            if not built.empty:
                df = pd.concat([df, built], ignore_index=True)
    else:
        df = build_team_week_epa(seasons).to_pandas()

    if df.empty:
        logger.warning("no EPA rows for %s (season may not have started)", seasons)
        return 0

    # A full rebuild may replace the table; a per-season refresh must not.
    if full_history:
        return load_table(df, CTX.hist_dataset, "team_week_epa",
                          cluster_fields=["season", "team"])
    return replace_seasons(df, CTX.hist_dataset, "team_week_epa",
                           cluster_fields=["season", "team"])


def backfill_teams() -> int:
    import nflreadpy as nfl

    df = nfl.load_teams().to_pandas()
    return load_table(df, CTX.hist_dataset, "teams")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-epa", action="store_true")
    args = ap.parse_args()

    ensure_dataset(CTX.hist_dataset)
    ensure_dataset(CTX.season_dataset)

    n_games = backfill_games()
    n_teams = backfill_teams()
    n_epa = 0 if args.skip_epa else backfill_epa()

    print(f"\nbackfill complete: {n_games} games (from {FIRST_SEASON}), "
          f"{n_teams} teams, {n_epa} team-week EPA rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
