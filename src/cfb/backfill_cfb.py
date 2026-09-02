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
         write_disposition: str = "WRITE_APPEND") -> int:
    """Append a frame to a BigQuery table.

    Defaults to WRITE_APPEND, not WRITE_TRUNCATE. A truncating default is how the
    weekly ingest wiped 2021-2024 out of cfb_historical.games: on a cold Cloud Function
    container the /tmp parquet cache is empty, so the frame held only the current season
    and the load replaced the entire table with it. A caller that means to replace
    something has to say which seasons, via replace_seasons().
    """
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


def _season_counts(client, table_id: str) -> dict[int, int]:
    """Rows per season already in BigQuery. Empty dict if the table does not exist."""
    try:
        rows = client.query(
            f"SELECT season, COUNT(*) AS n FROM `{table_id}` GROUP BY season"
        ).result()
    except Exception as exc:  # a table that has never been created is expected here
        logger.info("%s: no existing rows (%s)", table_id, str(exc)[:120])
        return {}
    return {int(r["season"]): int(r["n"]) for r in rows}


def _guard_shrink(df: pd.DataFrame, existing: dict[int, int], table_id: str) -> None:
    """Refuse a write that would shrink a season already present in BigQuery.

    The cold-cache signature: the parquet cache is empty or ESPN is unreachable, the
    fetch returns a fraction of a season, and the write replaces a complete season with
    it. Nothing downstream can distinguish that from a genuinely short season, so it has
    to fail loudly here. Seasons under 100 rows are still in progress and have nothing
    worth protecting.
    """
    incoming = df.groupby("season").size().to_dict()
    for season, have in sorted(existing.items()):
        if have < 100:
            continue
        got = int(incoming.get(season, 0))
        if got and got < 0.9 * have:
            raise RuntimeError(
                f"{table_id}: refusing to write {got} rows for season {season} over "
                f"{have} existing rows. That is the cold-cache signature — check the "
                f"ESPN fetch or the parquet cache actually returned a full season."
            )


def replace_seasons(df: pd.DataFrame, dataset: str, table: str,
                    partition_field: str | None = None,
                    cluster_fields: list[str] | None = None) -> int:
    """Replace exactly the seasons present in `df`, leaving every other season alone.

    The only safe way to refresh a season into a table that also holds history.
    WRITE_TRUNCATE here replaces the whole table with whatever the caller happened to
    fetch, which is what destroyed 2021-2024.
    """
    from google.cloud import bigquery

    if df.empty:
        logger.warning("%s.%s: nothing to load", dataset, table)
        return 0

    client = _bq()
    table_id = f"{cfb_config.CTX.project}.{dataset}.{table}"
    seasons = sorted(int(s) for s in df["season"].unique())

    _guard_shrink(df, _season_counts(client, table_id), table_id)

    try:
        client.query(
            f"DELETE FROM `{table_id}` WHERE season IN UNNEST(@seasons)",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("seasons", "INT64", seasons)
                ]
            ),
        ).result()
        logger.info("%s: cleared seasons %s", table_id, seasons)
    except Exception as exc:
        # First run: the table does not exist yet, so there is nothing to delete.
        logger.info("%s: pre-delete skipped (%s)", table_id, str(exc)[:120])

    return load(df, dataset, table, partition_field=partition_field,
                cluster_fields=cluster_fields, write_disposition="WRITE_APPEND")


def replace_game_ids(df: pd.DataFrame, dataset: str, table: str,
                     partition_field: str | None = None,
                     cluster_fields: list[str] | None = None) -> int:
    """Replace exactly the games present in `df`, leaving every other row alone.

    Predictions need game-level scoping, not season-level: the table carries rows for
    upcoming games as well as completed ones, and a backfill only ever regenerates the
    completed ones. Deleting a whole season would therefore drop the predictions for
    unplayed weeks, which nothing can rebuild — they have to have been written before
    kickoff to mean anything.
    """
    from google.cloud import bigquery

    if df.empty:
        logger.warning("%s.%s: nothing to load", dataset, table)
        return 0

    client = _bq()
    table_id = f"{cfb_config.CTX.project}.{dataset}.{table}"
    ids = df["game_id"].astype(str).tolist()

    try:
        client.query(
            f"DELETE FROM `{table_id}` WHERE game_id IN UNNEST(@ids)",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("ids", "STRING", ids)
                ]
            ),
        ).result()
        logger.info("%s: cleared %d game ids", table_id, len(ids))
    except Exception as exc:
        logger.info("%s: pre-delete skipped (%s)", table_id, str(exc)[:120])

    return load(df, dataset, table, partition_field=partition_field,
                cluster_fields=cluster_fields, write_disposition="WRITE_APPEND")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=cfb_config.HOLDOUT_SEASON)
    ap.add_argument("--games-only", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    games = load_games()
    if games.empty:
        raise SystemExit("no cached CFB games — run espn_data.fetch_history() first")

    logger.info("games: %d rows, %d seasons, divisions=%s",
                len(games), games["season"].nunique(),
                sorted(games["division"].unique()))

    if not args.no_write:
        ensure_datasets()
        g = games.copy()
        g["game_date"] = pd.to_datetime(g["game_date"])
        replace_seasons(g, cfb_config.CTX.hist_dataset, "games",
                        partition_field="game_date",
                        cluster_fields=["season", "division"])

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
        replace_game_ids(allrows, cfb_config.CTX.season_dataset, "game_predictions",
                         partition_field="game_date",
                         cluster_fields=["season", "division"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
