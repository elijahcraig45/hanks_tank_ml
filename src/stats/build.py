"""Build and publish football stat tables.

    python -m stats.build --sport nfl --season 2025 --dry-run
    python -m stats.build --sport cfb --season 2025 --write-bq

Coverage is uneven by design, because the feeds are:

  NFL   full per-player season table (nflverse) + leaders derived from it
  CFB   team season stats incl. opponent splits (ESPN) + league leaders (ESPN core)

College has no per-player season table because no public ESPN endpoint returns one —
the sortable athlete endpoint returns "-" for the very stat it sorts on. Rather than
fake it, the college player experience is leaders-only, and the UI says so.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stats import cfb_advanced, cfb_stats, cfbd, nfl_stats  # noqa: E402

logger = logging.getLogger(__name__)

DATASETS = {"nfl": ("NFL_DATASET", "nfl_season"), "cfb": ("CFB_DATASET", "cfb_season")}


def build(sport: str, season: int,
          providers: tuple[str, ...] = ("espn", "cfbd")) -> dict[str, pd.DataFrame]:
    """Every table this sport can supply, keyed by BigQuery table name.

    `providers` selects which feeds to pull. It exists so the Cloud Function can run
    them as separate invocations: the ESPN work rides the weekly ingest, while the
    CollegeFootballData work — a dozen HTTP calls, an 85-column flatten and a
    14,000-row pivot — gets its own job rather than being chained onto an invocation
    that already spends its 540 seconds on games, scoring and a bootstrap fit.

    Ordering matters for one table. Both providers can produce `stat_leaders`, and the
    write is delete-season-then-append, so whichever runs later wins. CFBD's version is
    the better one (it has player ids, positions, direction and volume floors), and
    running it second means a CFBD failure leaves ESPN's version standing rather than
    no leaderboard at all.
    """
    if sport == "nfl":
        players = nfl_stats.fetch_player_stats(season)
        if players.empty:
            # Pre-season: nflverse has not published this year yet. write_bq skips
            # empty frames, so last season's tables stay untouched.
            return {}
        return {
            "player_season_stats": players,
            "stat_leaders": nfl_stats.leaders(players),
        }
    if sport == "cfb":
        tables: dict[str, pd.DataFrame] = {}

    if sport == "cfb" and "espn" in providers:
        tables |= {
            # ESPN: conventional box-score totals with an opponent split. Kept as the
            # source for this table because it is what the site already renders, and
            # because mixing two providers' column vocabularies into one table would
            # put it at the mercy of either one's field rename.
            "team_season_stats": cfb_stats.fetch_team_stats(season),
        }

        # CollegeFootballData: everything ESPN cannot do. Each is optional — a missing
        # key or an uncovered tier must cost only its own table, never the whole run,
        # so failures are recorded and stepped over.
    if sport == "cfb" and "cfbd" in providers:
        if cfbd.has_api_key():
            for name, fetch in (
                ("team_game_advanced", cfb_advanced.fetch_team_game_advanced),
                ("team_season_advanced", cfb_advanced.fetch_team_season_advanced),
                ("team_season_epa", cfb_advanced.fetch_team_season_epa),
                ("betting_lines", cfb_advanced.fetch_lines),
            ):
                try:
                    frame = fetch(season)
                    if not frame.empty:
                        tables[name] = frame
                except Exception as exc:
                    logger.warning("cfb %s skipped: %s", name, str(exc)[:200])

            # Players and the leaderboard derived from them, so the board can never
            # disagree with the table beneath it.
            try:
                players = cfb_advanced.fetch_player_season(season)
                if not players.empty:
                    tables["player_season_stats"] = players
                    leaders = cfb_advanced.leaders_from_players(players, season)
                    if not leaders.empty:
                        tables["stat_leaders"] = leaders
            except Exception as exc:
                logger.warning("cfb players skipped: %s", str(exc)[:200])

    if sport == "cfb":
        # Falls back to ESPN's leaders only where the richer version was not produced —
        # either because CFBD was not asked for, or because it failed. ESPN's lacks
        # player_id, position, direction and any volume qualifier.
        if "stat_leaders" not in tables and "espn" in providers:
            tables["stat_leaders"] = cfb_stats.fetch_leaders(season)

        return tables
    raise ValueError(f"unknown sport: {sport}")


# Per-table partition and cluster spec. BigQuery fixes a table's clustering at
# creation, so a later append that asks for clustering the table was not created with is
# rejected outright — these have to be right on the first load and never change after.
# Tables under ~10k rows are deliberately absent: clustering buys nothing measurable
# there and only adds a way to fail.
# Deliberately empty. Verified 2026-09-01: cfb_season.team_season_stats,
# nfl_season.player_season_stats and both stat_leaders tables were all created
# unclustered and unpartitioned, so adding a spec for any of them here would make the
# next weekly append fail. Add an entry only for a table being created fresh.
TABLE_LAYOUT: dict[str, dict] = {}


_DDL_TYPES = {"STRING": "STRING", "FLOAT": "FLOAT64", "FLOAT64": "FLOAT64", "INTEGER": "INT64", "INT64": "INT64", "BOOLEAN": "BOOL", "BOOL": "BOOL",
              "TIMESTAMP": "TIMESTAMP", "DATE": "DATE", "DATETIME": "DATETIME", "TIME": "TIME", "NUMERIC": "NUMERIC", "BIGNUMERIC": "BIGNUMERIC", "BYTES": "BYTES"}


def _load_config(bigquery, layout: dict | None = None):
    cfg = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        # Without this, the first append that carries a new column is rejected.
        # These tables do gain columns: the feeds widen, and a pivoted player table
        # gains one per new (category, statType) pair the source starts publishing.
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    layout = layout or {}
    if layout.get("partition"):
        cfg.time_partitioning = bigquery.TimePartitioning(field=layout["partition"])
    if layout.get("cluster"):
        cfg.clustering_fields = layout["cluster"]
    return cfg


def _has_fraction(col: pd.Series) -> bool:
    """True if a numeric column holds a value that is not a whole number (an integer column cannot store it)."""
    if not pd.api.types.is_float_dtype(col):
        return False
    vals = col.dropna()
    return bool(len(vals) and (vals % 1 != 0).any())


def _widen_integer_columns(client, target, table_id: str, df: pd.DataFrame) -> list[str]:
    """Widen INTEGER columns to FLOAT64 where the feed now sends fractions.

    These tables are created by the first load, which types a column from that week's values: a percentage that happened to be 100.0 for every team became INT64 in
    BigQuery, and the first week it was 71.43 the load failed with 'Float value 71.428570 was truncated converting to int64' (CFB team stats, 2026-10-05). INT64 -> FLOAT64
    loses nothing below 2^53 and is the only direction the data can ever need, so it is done here, before the scratch copy is made from the target.
    """
    widened = []
    for f in target.schema:
        if f.field_type in ("INTEGER", "INT64") and f.name in df.columns and _has_fraction(df[f.name]):
            client.query(f"ALTER TABLE `{table_id}` ALTER COLUMN `{f.name}` SET DATA TYPE FLOAT64").result()
            widened.append(f.name)
    if widened:
        logger.warning("%s: widened %d INT64 column(s) to FLOAT64 for fractional values: %s", table_id, len(widened), ", ".join(widened))
    return widened


def _replace_season(client, bigquery, table_id: str, df: pd.DataFrame, season: int, layout: dict | None = None) -> None:
    """Replace one season's rows so that a failure anywhere leaves the table exactly as it was.

    This used to DELETE the season and then load the new rows. A load that failed (a pandas dtype BigQuery would not convert, in September 2026) had already deleted the
    season, and the site served an empty player table for weeks. Now the new rows are loaded into a scratch copy of the table first (same schema, same load options, so any
    conversion or load error happens BEFORE the target is touched), and the swap is one transaction: if the INSERT fails, the DELETE is rolled back.
    """
    from google.api_core.exceptions import NotFound

    try:
        target = client.get_table(table_id)
    except NotFound:
        # First load: there is nothing to protect, and this is the only time a table's partitioning and clustering can be chosen.
        client.load_table_from_dataframe(df, table_id, job_config=_load_config(bigquery, layout)).result()
        return

    stage_id = f"{table_id}__stage"
    try:
        if _widen_integer_columns(client, target, table_id, df):
            target = client.get_table(table_id)  # the schema changed: the scratch copy and the column comparison below must see the new one
        client.query(f"CREATE OR REPLACE TABLE `{stage_id}` LIKE `{table_id}` "
                     f"OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY))").result()
        client.load_table_from_dataframe(df, stage_id, job_config=_load_config(bigquery)).result()
        stage = client.get_table(stage_id)
        have = {f.name for f in target.schema}
        for f in stage.schema:  # a column the feed added: add it to the target first (DDL cannot run inside the transaction)
            if f.name not in have:
                client.query(f"ALTER TABLE `{table_id}` ADD COLUMN IF NOT EXISTS `{f.name}` {_DDL_TYPES.get(f.field_type, 'STRING')}").result()
        cols = ", ".join(f"`{f.name}`" for f in stage.schema)
        client.query(
            "BEGIN TRANSACTION;\n"
            f"DELETE FROM `{table_id}` WHERE season = {int(season)};\n"
            f"INSERT INTO `{table_id}` ({cols}) SELECT {cols} FROM `{stage_id}`;\n"
            "COMMIT TRANSACTION;").result()
    finally:
        try:
            client.delete_table(stage_id, not_found_ok=True)
        except Exception as exc:  # noqa: BLE001  the scratch table expires on its own after a day
            logger.info("%s: could not drop the scratch table (%s)", stage_id, str(exc)[:120])


def write_bq(sport: str, season: int, tables: dict[str, pd.DataFrame]) -> list[dict]:
    from google.cloud import bigquery

    env, default = DATASETS[sport]
    project = os.environ.get("GCP_PROJECT", "hankstank")
    dataset = os.environ.get(env, default)
    client = bigquery.Client(project=project)

    out = []
    for name, df in tables.items():
        if df.empty:
            continue
        table_id = f"{project}.{dataset}.{name}"
        _replace_season(client, bigquery, table_id, df, season, TABLE_LAYOUT.get(name))
        out.append({"table": table_id, "rows": len(df)})
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", required=True, choices=sorted(DATASETS))
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-bq", action="store_true")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--providers", type=str, default="espn,cfbd",
                    help="comma list: espn, cfbd")
    args = ap.parse_args()

    providers = tuple(p.strip() for p in args.providers.split(",") if p.strip())
    tables = build(args.sport, args.season, providers=providers)

    print()
    print(f"STAT TABLES — {args.sport.upper()} {args.season}")
    print("=" * 72)
    for name, df in tables.items():
        print(f"  {name:<22} {len(df):>6} rows  {len(df.columns):>4} cols")
        if args.out_dir:
            path = Path(args.out_dir) / f"{args.sport}_{name}_{args.season}.csv"
            df.to_csv(path, index=False)
            print(f"{'':<24} -> {path}")

    if "stat_leaders" in tables and not tables["stat_leaders"].empty:
        leaders = tables["stat_leaders"]
        print()
        print("  leader categories:")
        for label, group in leaders.groupby("category_label", sort=False):
            best = group.sort_values("rank").iloc[0]
            print(f"    {label:<22} {best.player_name} ({best.team}) {best.display_value}")

    if args.write_bq and not args.dry_run:
        for result in write_bq(args.sport, args.season, tables):
            print(f"\nwrote {result['rows']} rows to {result['table']}")
    else:
        env, default = DATASETS[args.sport]
        print(f"\n[dry run] would write to {os.environ.get(env, default)}.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
