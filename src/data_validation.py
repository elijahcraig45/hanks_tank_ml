#!/usr/bin/env python3
"""
Data Validation for mlb_2026_season dataset.

Checks:
  - Row counts per table
  - Duplicate detection (games by game_pk, statcast by game_pk+pitcher+batter+description)
  - Freshness (most recent snapshot_date / game_date vs today)
  - Referential integrity (team_ids consistent across tables)
  - Score sanity (no negative scores, reasonable innings)

Usage:
    python data_validation.py --year 2026
    python data_validation.py --year 2026 --fix-duplicates
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
DATASET = "mlb_2026_season"


class DataValidator:
    def __init__(self, fix_duplicates: bool = False):
        self.bq = bigquery.Client(project=PROJECT)
        self.fix_duplicates = fix_duplicates
        self.warnings = []
        self.errors = []

    def _query_scalar(self, sql: str):
        rows = list(self.bq.query(sql).result())
        return rows[0][0] if rows else None

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------
    def check_row_counts(self):
        logger.info("Checking row counts...")
        tables = ["teams", "team_stats", "player_stats", "standings",
                   "games", "rosters", "transactions", "statcast_pitches"]
        for t in tables:
            count = self._query_scalar(
                f"SELECT COUNT(*) FROM `{PROJECT}.{DATASET}.{t}`"
            )
            logger.info("  %-20s %s rows", t, f"{count:,}" if count else "0")
            if count == 0:
                self.warnings.append(f"{t} is empty")

    def check_game_duplicates(self):
        logger.info("Checking game duplicates...")
        dups = self._query_scalar(f"""
            SELECT COUNT(*) FROM (
                SELECT game_pk, game_date, COUNT(*) c
                FROM `{PROJECT}.{DATASET}.games`
                GROUP BY game_pk, game_date HAVING c > 1
            )
        """)
        if dups and dups > 0:
            msg = f"games has {dups} duplicate game_pk+date combos"
            self.errors.append(msg)
            if self.fix_duplicates:
                logger.info("  Fixing game duplicates...")
                self.bq.query(f"""
                    DELETE FROM `{PROJECT}.{DATASET}.games`
                    WHERE STRUCT(game_pk, game_date, synced_at) NOT IN (
                        SELECT STRUCT(game_pk, game_date, MAX(synced_at))
                        FROM `{PROJECT}.{DATASET}.games`
                        GROUP BY game_pk, game_date
                    )
                """).result()
                logger.info("  ✓ duplicates resolved")
        else:
            logger.info("  ✓ no game duplicates")

    def check_statcast_duplicates(self):
        logger.info("Checking statcast duplicates...")
        dups = self._query_scalar(f"""
            SELECT COUNT(*) FROM (
                SELECT game_pk, pitcher, batter, game_date, description, COUNT(*) c
                FROM `{PROJECT}.{DATASET}.statcast_pitches`
                GROUP BY game_pk, pitcher, batter, game_date, description HAVING c > 1
            )
        """)
        if dups and dups > 0:
            msg = f"statcast_pitches has {dups} duplicate groups"
            self.warnings.append(msg)
        else:
            logger.info("  ✓ no statcast duplicates")

    def check_freshness(self):
        logger.info("Checking data freshness...")
        yesterday = (date.today() - timedelta(days=1)).isoformat()

        latest_game = self._query_scalar(
            f"SELECT MAX(game_date) FROM `{PROJECT}.{DATASET}.games`"
        )
        if latest_game:
            latest_str = latest_game.isoformat() if hasattr(latest_game, "isoformat") else str(latest_game)
            logger.info("  latest game_date: %s", latest_str)
            if latest_str < yesterday:
                self.warnings.append(f"games not fresh — latest is {latest_str}")
        else:
            self.warnings.append("no games data at all")

        latest_standings = self._query_scalar(
            f"SELECT MAX(snapshot_date) FROM `{PROJECT}.{DATASET}.standings`"
        )
        if latest_standings:
            ls = latest_standings.isoformat() if hasattr(latest_standings, "isoformat") else str(latest_standings)
            logger.info("  latest standings snapshot: %s", ls)

    def check_score_sanity(self):
        logger.info("Checking score sanity...")
        bad_scores = self._query_scalar(f"""
            SELECT COUNT(*) FROM `{PROJECT}.{DATASET}.games`
            WHERE (home_score < 0 OR away_score < 0)
              AND status = 'Final'
        """)
        if bad_scores and bad_scores > 0:
            self.errors.append(f"{bad_scores} games with negative scores")
        else:
            logger.info("  ✓ scores OK")

        bad_innings = self._query_scalar(f"""
            SELECT COUNT(*) FROM `{PROJECT}.{DATASET}.games`
            WHERE innings > 25 AND status = 'Final'
        """)
        if bad_innings and bad_innings > 0:
            self.warnings.append(f"{bad_innings} games with >25 innings")

    def check_team_consistency(self):
        logger.info("Checking team ID consistency...")
        orphans = self._query_scalar(f"""
            SELECT COUNT(DISTINCT g.home_team_id)
            FROM `{PROJECT}.{DATASET}.games` g
            LEFT JOIN `{PROJECT}.{DATASET}.teams` t ON g.home_team_id = t.team_id
            WHERE t.team_id IS NULL AND g.home_team_id IS NOT NULL
        """)
        if orphans and orphans > 0:
            self.warnings.append(f"{orphans} home_team_ids in games not in teams table")
        else:
            logger.info("  ✓ team IDs consistent")

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------
    def run(self) -> int:
        self.check_row_counts()
        self.check_game_duplicates()
        self.check_statcast_duplicates()
        self.check_freshness()
        self.check_score_sanity()
        self.check_team_consistency()

        logger.info("=" * 60)
        if self.errors:
            logger.error("CRITICAL ERRORS: %d", len(self.errors))
            for e in self.errors:
                logger.error("  ✗ %s", e)
        if self.warnings:
            logger.warning("WARNINGS: %d", len(self.warnings))
            for w in self.warnings:
                logger.warning("  ⚠ %s", w)
        if not self.errors and not self.warnings:
            logger.info("✅ All checks passed")

        if self.errors:
            return 1
        if self.warnings:
            return 2
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--fix-duplicates", action="store_true")
    args = parser.parse_args()

    v = DataValidator(fix_duplicates=args.fix_duplicates)
    code = v.run()
    sys.exit(code)


if __name__ == "__main__":
    main()
