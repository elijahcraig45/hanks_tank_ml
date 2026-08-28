"""Backfill college football history and predictions into BigQuery.

Loads:
  cfb_historical.games           completed FBS + FCS games, tagged by division
  cfb_season.game_predictions    week-by-week out-of-sample predictions per division

FBS and FCS share both tables and are separated by the `division` column. Models are
trained per division; Elo is pooled across both (see pipeline.py for why).

Usage:
  python backfill_cfb.py                      # games + predictions for HOLDOUT_SEASON
  python backfill_cfb.py --season 2024
  python backfill_cfb.py --games-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nfl"))

import cfb_config  # noqa: E402
from espn_data import load_games  # noqa: E402
from pipeline import backfill_division, baselines, build  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _bq():
    """Import the NFL bq helpers but point them at the CFB project/datasets."""
    from google.cloud import bigquery

    return bigquery.Client(project=cfb_config.CTX.project)


def ensure_datasets() -> None:
    from google.cloud import bigquery

    c = _bq()
    for ds in (cfb_config.CTX.hist_dataset, cfb_config.CTX.season_dataset):
        ref = bigquery.Dataset(f"{cfb_config.CTX.project}.{ds}")
        ref.location = "US"
        try:
            c.get_dataset(ref)
        except Exception:
            c.create_dataset(ref)
            logger.info("created dataset %s", ds)


def load(df: pd.DataFrame, dataset: str, table: str,
         partition_field: str | None = None,
         cluster_fields: list[str] | None = None,
         write_disposition: str = "WRITE_TRUNCATE") -> int:
    from google.cloud import bigquery

    if df.empty:
        logger.warning("%s.%s: nothing to load", dataset, table)
        return 0

    cfg = bigquery.LoadJobConfig(write_disposition=write_disposition, autodetect=True)
    if partition_field:
        cfg.time_partitioning = bigquery.TimePartitioning(field=partition_field)
    if cluster_fields:
        cfg.clustering_fields = cluster_fields

    table_id = f"{cfb_config.CTX.project}.{dataset}.{table}"
    _bq().load_table_from_dataframe(df, table_id, job_config=cfg).result()
    logger.info("loaded %d rows -> %s", len(df), table_id)
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=cfb_config.HOLDOUT_SEASON)
    ap.add_argument("--games-only", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    games = load_games()
    if games.empty:
        raise SystemExit("no cached CFB games — run data.fetch_history() first")

    logger.info("games: %d rows, %d seasons, divisions=%s",
                len(games), games["season"].nunique(),
                sorted(games["division"].unique()))

    if not args.no_write:
        ensure_datasets()
        g = games.copy()
        g["game_date"] = pd.to_datetime(g["game_date"])
        load(g, cfb_config.CTX.hist_dataset, "games",
             partition_field="game_date", cluster_fields=["season", "division"])

    if args.games_only:
        return 0

    feats = build(games)
    frames = []
    print()
    for division in ("fbs", "fcs"):
        base = baselines(feats, division, args.season)
        if not base:
            logger.warning("no %s games for %d", division, args.season)
            continue
        rows = backfill_division(feats, division, args.season)
        if rows.empty:
            continue
        frames.append(rows)
        acc = rows["prediction_correct"].mean()
        print(f"{division.upper()} {args.season}: {len(rows)} predictions")
        print(f"  model        {acc:.2%}")
        print(f"  always-home  {base['always_home']:.2%}")
        print(f"  Elo only     {base['elo_only']:.2%}   <- the bar")
        print(rows.groupby("confidence_tier")["prediction_correct"]
              .agg(["count", "mean"]).to_string())
        print()

    if frames and not args.no_write:
        allrows = pd.concat(frames, ignore_index=True)
        load(allrows, cfb_config.CTX.season_dataset, "game_predictions",
             partition_field="game_date", cluster_fields=["season", "division"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
