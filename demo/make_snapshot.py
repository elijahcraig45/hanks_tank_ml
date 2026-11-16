"""Rebuild demo/data/nfl_games_1999_2025.csv.gz, the only input the demo reads.

    python demo/make_snapshot.py                                   # fresh pull (needs nflreadpy and network)
    python demo/make_snapshot.py --source data/nfl/raw/schedules.parquet

The demo never runs this. It is here so anyone can check where the snapshot came from: the nflverse schedules
table (https://github.com/nflverse/nflverse-data, CC BY 4.0), cut to completed 1999-2025 games and the columns the
NFL features use. The committed file was made with --source from the pipeline's cached pull of 2026-11-03.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "data" / "nfl_games_1999_2025.csv.gz"
LAST_SEASON = 2025
COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday", "away_team", "away_score", "home_team", "home_score",
    "location", "result", "away_rest", "home_rest", "spread_line", "total_line", "div_game",
]


def load(source: str | None) -> pd.DataFrame:
    if source:
        return pd.read_parquet(source)
    import nflreadpy

    return nflreadpy.load_schedules().to_pandas()


def snapshot(df: pd.DataFrame) -> pd.DataFrame:
    played = df[(df.season <= LAST_SEASON) & df.result.notna()]
    # Keep nflverse's row order. XGBoost subsamples rows, so reordering games that share a date moves its log loss
    # by about 0.001 even though every feature is identical.
    return played[COLUMNS].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", help="a cached nflverse schedules parquet instead of a fresh pull")
    snap = snapshot(load(ap.parse_args().source))
    snap.to_csv(OUT, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"wrote {OUT.name}: {len(snap):,} games, {snap.season.min()}-{snap.season.max()}")


if __name__ == "__main__":
    main()
