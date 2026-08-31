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

from stats import cfb_stats, nfl_stats  # noqa: E402

logger = logging.getLogger(__name__)

DATASETS = {"nfl": ("NFL_DATASET", "nfl_season"), "cfb": ("CFB_DATASET", "cfb_season")}


def build(sport: str, season: int) -> dict[str, pd.DataFrame]:
    """Every table this sport can supply, keyed by BigQuery table name."""
    if sport == "nfl":
        players = nfl_stats.fetch_player_stats(season)
        return {
            "player_season_stats": players,
            "stat_leaders": nfl_stats.leaders(players),
        }
    if sport == "cfb":
        return {
            "team_season_stats": cfb_stats.fetch_team_stats(season),
            "stat_leaders": cfb_stats.fetch_leaders(season),
        }
    raise ValueError(f"unknown sport: {sport}")


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
        # Replace this season's slice so re-runs are idempotent.
        try:
            client.query(
                f"DELETE FROM `{table_id}` WHERE season = {season}"
            ).result()
        except Exception:
            pass  # table does not exist yet; the load creates it
        client.load_table_from_dataframe(
            df, table_id,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
        ).result()
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
    args = ap.parse_args()

    tables = build(args.sport, args.season)

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
