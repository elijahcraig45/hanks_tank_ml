#!/usr/bin/env python3
"""
V7 Backfill Predictions Script

Backfills predictions for all completed games between two dates using the V7 model.
This catches any games that weren't predicted during the daily pipeline runs.

Usage:
    python backfill_v7_predictions.py --start 2026-03-27 --end 2026-04-06
"""

import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
DATASET = "mlb_2026_season"


def backfill_predictions(start_date: date, end_date: date) -> int:
    """
    Backfill predictions for all games in date range.
    
    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        
    Returns:
        Number of games predicted
    """
    bq = bigquery.Client(project=PROJECT)
    
    from predict_today_games import DailyPredictor
    
    predictor = DailyPredictor(dry_run=False, fallback_v4=False)
    predictor.load_model()
    
    total_predicted = 0
    current_date = start_date
    
    while current_date <= end_date:
        logger.info("Processing %s...", current_date.isoformat())
        try:
            result = predictor.run_for_date(current_date)
            games_predicted = result.get("games_predicted", 0)
            total_predicted += games_predicted
            logger.info("  ✓ Predicted %d games", games_predicted)
        except Exception as e:
            logger.error("  ✗ Error on %s: %s", current_date.isoformat(), e)
        
        current_date += timedelta(days=1)
    
    logger.info("\n✅ Backfill complete: %d total games predicted", total_predicted)
    return total_predicted


def main():
    parser = argparse.ArgumentParser(description="Backfill V7 predictions")
    parser.add_argument(
        "--start",
        type=str,
        default="2026-03-27",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-04-06",
        help="End date (YYYY-MM-DD)",
    )
    args = parser.parse_args()
    
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    
    backfill_predictions(start, end)


if __name__ == "__main__":
    main()
