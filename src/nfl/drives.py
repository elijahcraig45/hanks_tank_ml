"""One row per drive, extracted from nflverse play-by-play, for the drive simulator.

Stored in `nfl_historical.drives` (DDL: scripts/gcp/football/create_drive_sim_tables.sql).
Why a table: the simulator fits on drives, and play-by-play is the heaviest load in this
repo. Rebuilding it in a cold Cloud Function is what killed the old EPA ingest
(`Memory limit of 1953 MiB exceeded`). So:

  * weekly: `mode=ingest` calls ingest_season(current season) — one season of pbp, the
    same pull the EPA step already makes — and replaces only that season's rows;
  * once: history 2008-2025 is loaded from local pbp parquet with
        python src/nfl/drives.py --pbp-dir <dir with pbp_YYYY.parquet> --seasons 2008-2025 --write
    (without --write it only writes a local parquet / prints counts).

The extraction is the research extractor (research/football_2026_09/drive_sim/nfl_drives.py)
verbatim, so the live model trains on the same rows the backtest did.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)

TABLE = "drives"
PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "home_team", "away_team", "posteam", "defteam",
    "qtr", "game_half", "half_seconds_remaining", "play_type", "yardline_100", "fixed_drive",
    "fixed_drive_result", "total_home_score", "total_away_score", "play_id",
]
RES = {"Touchdown": "TD", "Field goal": "FG", "Missed field goal": "MFG", "Punt": "PUNT",
       "Turnover": "TO", "Opp touchdown": "OTD", "Turnover on downs": "TOD", "Safety": "SAF",
       "End of half": "EOH"}
SCRIMMAGE = ["run", "pass", "punt", "field_goal", "no_play", "qb_kneel", "qb_spike"]


def extract_drives(pbp: "pl.DataFrame") -> "pl.DataFrame":
    """Plays -> drives for one or more seasons. Empty in, empty out."""
    import polars as pl  # lazy: the predict path reads drives without polars

    if pbp is None or pbp.height == 0:
        return pl.DataFrame()
    p = pbp.select(PBP_COLUMNS)
    p = p.filter(pl.col("fixed_drive").is_not_null() & pl.col("posteam").is_not_null()).sort(
        ["game_id", "play_id"])
    p = p.with_columns([
        pl.col("total_home_score").shift(1).over("game_id").fill_null(0).alias("hs0"),
        pl.col("total_away_score").shift(1).over("game_id").fill_null(0).alias("as0")])
    scrim = pl.col("play_type").is_in(SCRIMMAGE)
    d = p.group_by(["game_id", "fixed_drive"], maintain_order=True).agg([
        pl.col("season").first(), pl.col("week").first(), pl.col("season_type").first(),
        pl.col("home_team").first(), pl.col("away_team").first(),
        pl.col("posteam").first(), pl.col("defteam").first(), pl.col("qtr").first(),
        pl.col("game_half").first(), pl.col("fixed_drive_result").first().alias("res"),
        pl.col("half_seconds_remaining").first().alias("hsr0"),
        pl.col("yardline_100").filter(scrim).first().alias("yl"),
        pl.col("hs0").first(), pl.col("as0").first(),
        pl.col("total_home_score").last().alias("hs1"), pl.col("total_away_score").last().alias("as1"),
        pl.col("yardline_100").filter(scrim).last().alias("yl_end"),
    ])
    d = d.with_columns(pl.col("res").replace_strict(RES, default=None).alias("res"))
    d = d.with_columns([
        (pl.col("posteam") == pl.col("home_team")).alias("off_home"),
        (pl.col("hs1") - pl.col("hs0")).alias("dh"), (pl.col("as1") - pl.col("as0")).alias("da")])
    d = d.with_columns([
        pl.when(pl.col("off_home")).then(pl.col("dh")).otherwise(pl.col("da")).alias("off_pts"),
        pl.when(pl.col("off_home")).then(pl.col("da")).otherwise(pl.col("dh")).alias("def_pts"),
        pl.when(pl.col("off_home")).then(pl.col("hs0") - pl.col("as0"))
          .otherwise(pl.col("as0") - pl.col("hs0")).alias("sd0"),
    ])
    d = d.with_columns([
        pl.col("hsr0").shift(-1).over(["game_id", "game_half"]).alias("hsr_next"),
        pl.col("yl").shift(-1).over(["game_id", "game_half"]).alias("yl_next"),
    ])
    d = d.with_columns((pl.col("hsr0") - pl.col("hsr_next").fill_null(0)).clip(0, 1800).alias("dur"))
    return d.with_columns([pl.col(c).cast(pl.Float64) for c in (
        "fixed_drive", "qtr", "hsr0", "yl", "hs0", "as0", "hs1", "as1", "yl_end", "dh", "da",
        "off_pts", "def_pts", "sd0", "hsr_next", "yl_next", "dur")]
        + [pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)])


def ingest_season(season: int) -> int:
    """Weekly path: one season of pbp -> that season's drives, replacing only it."""
    import nflreadpy as nfl
    from bq_io import replace_seasons
    from config import CTX

    pbp = nfl.load_pbp(seasons=[season])
    d = extract_drives(pbp)
    del pbp
    if d.height == 0:
        logger.info("no drives yet for %d (season not started)", season)
        return 0
    return replace_seasons(d.to_pandas(), CTX.hist_dataset, TABLE,
                           cluster_fields=["season", "posteam"])


def load_drives(first_season: int, source: str = "auto"):
    """Drives for seasons >= first_season: BigQuery, or a local parquet
    (env NFL_DRIVES_PARQUET) when source is "auto"/"cache" and BigQuery has nothing."""
    import os

    import pandas as pd

    if source in ("auto", "bq"):
        try:
            from bq_io import client
            from config import CTX
            from google.cloud import bigquery

            cfg = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("s", "INT64", int(first_season))])
            df = client().query(
                f"SELECT * FROM `{CTX.project}.{CTX.hist_dataset}.{TABLE}` WHERE season >= @s",
                job_config=cfg).to_dataframe()
            if not df.empty:
                return df
        except Exception as exc:
            if source == "bq":
                raise
            logger.warning("drives BigQuery read failed (%s); trying local parquet", exc)
    path = os.environ.get("NFL_DRIVES_PARQUET")
    if path and Path(path).exists():
        df = pd.read_parquet(path)
        return df[df.season >= first_season]
    return pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description="backfill nfl_historical.drives from local pbp")
    ap.add_argument("--pbp-dir", required=True)
    ap.add_argument("--seasons", default="2008-2025")
    ap.add_argument("--out", help="also write a local parquet here")
    ap.add_argument("--write", action="store_true", help="replace these seasons in BigQuery")
    a = ap.parse_args()
    import polars as pl

    lo, hi = (int(x) for x in a.seasons.split("-"))
    frames = []
    for s in range(lo, hi + 1):
        f = Path(a.pbp_dir) / f"pbp_{s}.parquet"
        d = extract_drives(pl.read_parquet(f, columns=PBP_COLUMNS))
        frames.append(d)
        print(s, d.height, flush=True)
    d = pl.concat(frames, how="diagonal_relaxed")
    if a.out:
        d.write_parquet(a.out)
    if a.write:
        from bq_io import replace_seasons
        from config import CTX

        n = replace_seasons(d.to_pandas(), CTX.hist_dataset, TABLE,
                            cluster_fields=["season", "posteam"])
        print("wrote", n)
    print("total drives", d.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
