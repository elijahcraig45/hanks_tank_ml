#!/usr/bin/env python3
"""
Backfill lineup-aware V10 predictions across completed 2026 games.

For each date in range:
  1. Rebuild matchup_features into BigQuery
  2. Rebuild game_v10_features into BigQuery
  3. Rerun predictions with the active V10 model

It then evaluates the latest prediction row per game against final 2026 outcomes,
which feeds the diagnostics surface because those metrics are derived from the
latest entries in `game_predictions`.
"""

import argparse
import logging
from datetime import date, timedelta

from google.cloud import bigquery

from build_matchup_features import MatchupFeatureBuilder
from build_v10_features_live import V10LiveFeatureBuilder
from predict_today_games import DailyPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
DATASET = "mlb_2026_season"


class V10BackfillRunner:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.bq = bigquery.Client(project=PROJECT)
        self.matchup_builder = MatchupFeatureBuilder(dry_run=dry_run)
        self.v10_builder = V10LiveFeatureBuilder(dry_run=dry_run)
        self.predictor = DailyPredictor(dry_run=dry_run, fallback_v4=False)
        self.predictor.load_model()

    def run_day(self, target_date: date) -> dict:
        logger.info("Processing %s", target_date.isoformat())
        matchup_result = self.matchup_builder.run_for_date(target_date)
        v10_result = self.v10_builder.run_for_date(target_date, include_final=True)
        pred_result = self.predictor.run_for_date(target_date, include_final=True)
        return {
            "date": target_date.isoformat(),
            "matchup_rows": matchup_result.get("rows_written", 0),
            "v10_rows": v10_result.get("games_processed", 0),
            "predictions": pred_result.get("games_predicted", 0),
        }

    def run_range(self, start_date: date, end_date: date) -> dict:
        results = []
        current = start_date
        while current <= end_date:
            results.append(self.run_day(current))
            current += timedelta(days=1)

        totals = {
            "days_processed": len(results),
            "matchup_rows": sum(row["matchup_rows"] for row in results),
            "v10_rows": sum(row["v10_rows"] for row in results),
            "predictions": sum(row["predictions"] for row in results),
        }
        logger.info("Backfill totals: %s", totals)
        return {"totals": totals, "days": results}

    def evaluate_range(self, start_date: date, end_date: date) -> dict:
        sql = f"""
            WITH latest_pred AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY game_pk
                        ORDER BY predicted_at DESC
                    ) AS rn
                FROM `{PROJECT}.{DATASET}.game_predictions`
                WHERE game_date BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
            ),
            scored AS (
                SELECT
                    p.game_pk,
                    p.game_date,
                    p.model_version,
                    p.confidence_tier,
                    p.home_win_probability,
                    g.home_score,
                    g.away_score,
                    CAST(g.home_score > g.away_score AS INT64) AS actual_home_win,
                    CASE
                        WHEN (
                            (p.home_win_probability >= 0.5 AND g.home_score > g.away_score)
                            OR (p.home_win_probability < 0.5 AND g.away_score > g.home_score)
                        ) THEN 1
                        ELSE 0
                    END AS correct
                FROM latest_pred p
                JOIN `{PROJECT}.{DATASET}.games` g
                  ON g.game_pk = p.game_pk
                WHERE p.rn = 1
                  AND g.home_score IS NOT NULL
                  AND g.away_score IS NOT NULL
            )
            SELECT
                ANY_VALUE(model_version) AS model_version,
                COUNT(*) AS games,
                ROUND(AVG(correct), 4) AS accuracy,
                ROUND(AVG(POW(home_win_probability - actual_home_win, 2)), 4) AS brier,
                ROUND(AVG(CASE WHEN confidence_tier = 'high' THEN correct END), 4) AS high_accuracy,
                ROUND(AVG(CASE WHEN confidence_tier = 'medium' THEN correct END), 4) AS medium_accuracy,
                ROUND(AVG(CASE WHEN confidence_tier = 'low' THEN correct END), 4) AS low_accuracy,
                ROUND(AVG(CASE WHEN confidence_tier = 'high' THEN 1 ELSE 0 END), 4) AS high_coverage,
                ROUND(AVG(CASE WHEN confidence_tier IN ('high', 'medium') THEN 1 ELSE 0 END), 4) AS medium_plus_coverage
            FROM scored
        """
        rows = list(self.bq.query(sql).result())
        if not rows:
            return {}
        row = rows[0]
        result = {key: row[key] for key in row.keys()}
        logger.info("2026 evaluation summary: %s", result)
        return result


def main():
    parser = argparse.ArgumentParser(description="Backfill V10 features and predictions for completed 2026 games")
    parser.add_argument("--start", default="2026-03-27", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runner = V10BackfillRunner(dry_run=args.dry_run)
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    runner.run_range(start_date, end_date)
    if not args.dry_run:
        runner.evaluate_range(start_date, end_date)


if __name__ == "__main__":
    main()
