"""BigQuery IO for the NFL pipeline.

Ports the delete-then-insert idempotency pattern from the MLB pipeline
(season_2026_pipeline.py `_load_to_bq`/`_delete_date`), with one change: the delete key
is (season, week) rather than a date. NFL results settle across Thursday->Monday, so a
date-keyed overwrite would leave half a week's rows orphaned.

Uses load jobs rather than streaming inserts — the MLB side learned the hard way that
streaming buffers conflict with partition overwrites.
"""

from __future__ import annotations

import logging

import pandas as pd
from google.cloud import bigquery

from config import CTX

logger = logging.getLogger(__name__)


def client() -> bigquery.Client:
    return bigquery.Client(project=CTX.project)


def ensure_dataset(dataset: str, location: str = "US") -> None:
    c = client()
    ref = bigquery.Dataset(f"{CTX.project}.{dataset}")
    ref.location = location
    try:
        c.get_dataset(ref)
        logger.info("dataset %s exists", dataset)
    except Exception:
        c.create_dataset(ref)
        logger.info("created dataset %s", dataset)


def load_table(
    df: pd.DataFrame,
    dataset: str,
    table: str,
    write_disposition: str = "WRITE_TRUNCATE",
    partition_field: str | None = None,
    cluster_fields: list[str] | None = None,
) -> int:
    """Load a dataframe, replacing the table by default."""
    if df.empty:
        logger.warning("%s.%s: nothing to load", dataset, table)
        return 0

    c = client()
    table_id = f"{CTX.project}.{dataset}.{table}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
    )
    if partition_field:
        job_config.time_partitioning = bigquery.TimePartitioning(field=partition_field)
    if cluster_fields:
        job_config.clustering_fields = cluster_fields

    job = c.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    logger.info("loaded %d rows -> %s", len(df), table_id)
    return len(df)


def replace_seasons(
    df: pd.DataFrame,
    dataset: str,
    table: str,
    partition_field: str | None = None,
    cluster_fields: list[str] | None = None,
) -> int:
    """Replace exactly the seasons present in `df`, leaving every other season alone.

    The safe way to refresh one season into a table that also holds history.
    load_table defaults to WRITE_TRUNCATE, which replaces the whole table with
    whatever the caller happened to build — fine for a table rebuilt in full from a
    single source, and destructive for anything refreshed a season at a time.
    """
    if df.empty:
        logger.warning("%s.%s: nothing to load", dataset, table)
        return 0

    seasons = sorted(int(x) for x in df["season"].unique())
    table_id = f"{CTX.project}.{dataset}.{table}"

    try:
        client().query(
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

    return load_table(df, dataset, table, write_disposition="WRITE_APPEND",
                      partition_field=partition_field,
                      cluster_fields=cluster_fields)


def delete_week(dataset: str, table: str, season: int, week: int) -> None:
    """Idempotency for weekly reruns: clear the (season, week) slice before insert."""
    c = client()
    sql = f"""
        DELETE FROM `{CTX.project}.{dataset}.{table}`
        WHERE season = @season AND week = @week
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("season", "INT64", season),
        bigquery.ScalarQueryParameter("week", "INT64", week),
    ])
    try:
        c.query(sql, job_config=cfg).result()
        logger.info("cleared %s.%s for %d wk%d", dataset, table, season, week)
    except Exception as exc:
        # Table may not exist yet on a first run — that's fine.
        logger.debug("delete_week skipped (%s)", exc)


def upsert_week(df: pd.DataFrame, dataset: str, table: str, season: int, week: int) -> int:
    delete_week(dataset, table, season, week)
    return load_table(df, dataset, table, write_disposition="WRITE_APPEND")


def query(sql: str) -> pd.DataFrame:
    return client().query(sql).to_dataframe()
