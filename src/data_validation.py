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
                self._dedupe_games()
        else:
            logger.info("  ✓ no game duplicates")

    def _dedupe_games(self):
        """Collapse duplicate game_pk+game_date rows, keeping the freshest.

        The previous implementation was a single DELETE keyed on
        STRUCT(game_pk, game_date, synced_at) NOT IN (... MAX(synced_at) ...).
        That silently never deleted anything in the case this table actually
        produces: the collector inserts a skeleton row when a game is scheduled
        (synced_at NULL, no score) and inserts the completed row afterwards
        rather than updating in place. `STRUCT(.., NULL) NOT IN (..)` evaluates
        to NULL, not TRUE, so the skeleton row survived every run and the
        validator reported CRITICAL forever.

        Ranking with NULLS LAST fixes the NULL case, and rewriting the affected
        keys through a transaction also handles two fully identical rows, which
        no DELETE predicate can tell apart. Only duplicated keys are touched,
        and the table is never recreated, so partitioning and clustering on
        `games` are preserved.
        """
        table = f"{PROJECT}.{DATASET}.games"
        try:
            self.bq.query(f"""
                BEGIN TRANSACTION;

                CREATE TEMP TABLE _games_dedup AS
                SELECT * EXCEPT(_rn) FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY game_pk, game_date
                        ORDER BY synced_at DESC NULLS LAST
                    ) AS _rn
                    FROM `{table}`
                    WHERE STRUCT(game_pk, game_date) IN (
                        SELECT STRUCT(game_pk, game_date) FROM `{table}`
                        GROUP BY game_pk, game_date HAVING COUNT(*) > 1
                    )
                )
                WHERE _rn = 1;

                DELETE FROM `{table}`
                WHERE STRUCT(game_pk, game_date) IN (
                    SELECT STRUCT(game_pk, game_date) FROM _games_dedup
                );

                INSERT INTO `{table}`
                SELECT * FROM _games_dedup;

                COMMIT TRANSACTION;
            """).result()
            logger.info("  ✓ duplicates resolved")
        except Exception as exc:
            # Rows still in BigQuery's streaming buffer reject UPDATE/DELETE for
            # ~90 minutes after insert. Report it and let the next run retry
            # rather than failing the whole validation step.
            logger.warning("  could not dedupe games (will retry next run): %s", exc)
            self.warnings.append(f"game dedupe deferred: {str(exc)[:120]}")

    def check_statcast_duplicates(self):
        """Count rows in statcast_pitches that are byte-identical to another row.

        This used to group by (game_pk, pitcher, batter, game_date, description),
        which is not a uniqueness constraint: a pitcher legitimately throws
        several pitches with the same `description` to the same batter in a game,
        so two called strikes in one at-bat counted as a duplicate. That reported
        185,700 "duplicate groups" against 781,427 rows — permanent noise that
        buried the 13 real `games` duplicates sitting next to it in the output.

        The table has no pitch identifier (no at_bat_number / pitch_number), so
        there is no key to dedupe on. Fully identical rows are the only defensible
        signal: pitches differing in inning, count, type or velocity are distinct
        events. On the same data this reports 9. ~288 MB scanned.
        """
        logger.info("Checking statcast duplicates...")
        table = f"{PROJECT}.{DATASET}.statcast_pitches"
        surplus = self._query_scalar(f"""
            SELECT (SELECT COUNT(*) FROM `{table}`)
                 - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM `{table}`))
        """)
        if surplus and surplus > 0:
            self.warnings.append(
                f"statcast_pitches has {surplus} exact duplicate rows"
            )
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
        """Flag games whose home club is missing from the teams table.

        Exhibition ('E') and All-Star ('A') games are excluded because their
        opponents are legitimately not MLB clubs — the 2026 slate alone has
        Dominican Republic, Cuba, the National League All-Stars, and the
        Sacramento River Cats and Springfield Cardinals affiliates. `teams`
        holds the 30 MLB clubs, so those five will never join and the check
        warned about them on every run forever. Regular-season ('R') and spring
        ('S') games have zero orphans; postseason codes are left in scope
        deliberately, since an orphan there would be a real defect.
        """
        logger.info("Checking team ID consistency...")
        orphans = self._query_scalar(f"""
            SELECT COUNT(DISTINCT g.home_team_id)
            FROM `{PROJECT}.{DATASET}.games` g
            LEFT JOIN `{PROJECT}.{DATASET}.teams` t ON g.home_team_id = t.team_id
            WHERE t.team_id IS NULL AND g.home_team_id IS NOT NULL
              AND g.game_type NOT IN ('E', 'A')
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
