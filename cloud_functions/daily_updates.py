#!/usr/bin/env python3
"""
DEPRECATED — This module is no longer the active cloud function implementation.

Replaced by src/cloud_function_main.py (entry: daily_pipeline) as of 2026-04-08.
See cloud_functions/DEPRECATED.md for the full migration guide.

Known issues with this file (reasons for deprecation):
  1. fetch_statcast_from_baseball_savant() returns [] always — never implemented
  2. predict_today_games() returns a hardcoded placeholder — never wired to predictor
  3. Entry points targeted by setup_scheduler.sh use old function names that no
     longer match the deployed Cloud Function configuration
  4. Missing catboost dependency — V8 model cannot be loaded from this code

This file is retained as an audit artifact.
----------------------------------------------------------------------
Cloud Functions for Daily Data Pipeline: 2026 Season Data Updates

Orchestrates daily data updates for statcast, pitcher stats, features, and predictions.
Implements unified historical + 2026 data architecture using DELETE-before-INSERT pattern.

Schedule:
  06:00 UTC - update_statcast_2026()           → statcast_pitches
  06:30 UTC - update_pitcher_stats_2026()      → pitcher_game_stats
  08:00 UTC - rebuild_v7_features()            → matchup_v7_features
  12:00 UTC - predict_today_games()            → game_predictions

Dependencies (add to requirements.txt):
  - google-cloud-bigquery==3.20.0
  - google-cloud-storage==2.10.0
  - requests==2.31.0
  - functions-framework==3.4.0

Deployment:
  bash cloud_functions/setup_scheduler.sh

Environment variables (set in Cloud Function config):
  PROJECT_ID: hankstank
  HISTORICAL_DATASET: mlb_historical_data
  SEASON_2026_DATASET: mlb_2026_season
"""

import functions_framework
from datetime import date, timedelta
from google.cloud import bigquery
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = "hankstank"
HISTORICAL_DATASET = "mlb_historical_data"
SEASON_2026_DATASET = "mlb_2026_season"


# ============================================================================
# Data Fetch Helper Functions
# ============================================================================

def fetch_statcast_from_baseball_savant(target_date):
    """
    Fetch Statcast data from Baseball Savant for a specific date.
    
    Requires: requests, pandas
    Returns: list of dicts ready for BigQuery insertion
    
    TODO: Implement actual fetch from:
      1. pybaseball.statcast() - if available
      2. MLB Stats API - https://statsapi.mlb.com
      3. Baseball Savant web scraping - https://baseballsavant.mlb.com/statcast
    """
    import requests
    import pandas as pd
    
    logger.info(f"Fetching Statcast data for {target_date} from Baseball Savant...")
    
    # Placeholder: This would query Baseball Savant or MLB API
    # For now, return empty list (no-op)
    try:
        # Example using pybaseball if available:
        # from pybaseball import statcast_pitcher
        # df = statcast_pitcher(target_date, target_date)
        # return df.to_dict('records')
        
        # Example using MLB Stats API:
        # url = f"https://statsapi.mlb.com/api/v1/game?dates={target_date.isoformat()}"
        # ...
        
        logger.warning("Baseball Savant fetch not implemented - returning placeholder")
        return []
    
    except Exception as e:
        logger.error(f"Error fetching Statcast data: {e}")
        return []


def compute_pitcher_stats_from_statcast(target_date):
    """
    Compute pitcher_game_stats from statcast_pitches.
    
    Groups statcast pitch-level data by pitcher and game to create
    pitcher game-level statistics (velocities, K-BB%, pitch mix, etc).
    
    Uses statcast_pitches table as source (no separate pitcher_game_stats table).
    """
    logger.info(f"Computing pitcher stats from Statcast for {target_date}...")
    
    bq = bigquery.Client(project=PROJECT_ID)
    
    try:
        # Query statcast and aggregate by pitcher
        sql = f"""
        SELECT
            pitcher,
            game_date,
            COUNT(*) as total_pitches,
            COUNTIF(pitch_type IN ('FF', 'SI', 'FA')) as fastball_count,
            COUNTIF(pitch_type IN ('SL', 'CU', 'CB', 'SC')) as breaking_count,
            COUNTIF(pitch_type IN ('CH', 'FS', 'FO')) as offspeed_count,
            ROUND(AVG(release_speed), 2) as mean_fastball_velo,
            ROUND(STDDEV(release_speed), 2) as velo_std,
            ROUND(AVG(release_spin_rate), 0) as mean_spin_rate
        FROM `{PROJECT_ID}.{SEASON_2026_DATASET}.statcast_pitches`
        WHERE game_date = '{target_date.isoformat()}'
          AND pitcher IS NOT NULL
          AND release_speed IS NOT NULL
        GROUP BY pitcher, game_date
        """
        
        logger.info(f"Running aggregation query for pitcher stats on {target_date}...")
        result = bq.query(sql).to_dataframe()
        
        if result.empty:
            logger.warning(f"No statcast data found for {target_date}")
            return []
        
        # Convert to records format for BigQuery insertion
        records = result.to_dict('records')
        logger.info(f"Computed pitcher stats for {len(records)} pitchers on {target_date}")
        return records
        
    except Exception as e:
        logger.error(f"Error computing pitcher stats: {e}", exc_info=True)
        return []


@functions_framework.http
def update_pitcher_stats_2026(request):
    """
    Update pitcher_game_stats for 2026 season.
    
    Triggered by Cloud Scheduler at 06:30 UTC daily
    
    Process:
      1. DELETE rows for yesterday (to prevent duplicates)
      2. FETCH new pitcher stats from data source
      3. INSERT fresh stats into mlb_2026_season.pitcher_game_stats
    
    Data sources:
      - Compute from statcast_pitches (aggregate pitch-level to game-level)
      - MLB Stats API or Baseball Savant API
      - Manual batch uploads from external processor
    
    Returns: JSON with status, date, rows_inserted, rows_deleted
    """
    bq = bigquery.Client(project=PROJECT_ID)
    
    target_date = date.today() - timedelta(days=1)  # Yesterday's games
    
    logger.info(f"Updating pitcher_game_stats for {target_date}")
    
    try:
        # Step 1: Delete old rows to avoid duplicates
        pitcher_stats_table = f"{PROJECT_ID}.{SEASON_2026_DATASET}.pitcher_game_stats"
        
        delete_sql = f"""
        DELETE FROM `{pitcher_stats_table}`
        WHERE DATE(game_date) = '{target_date.isoformat()}'
        """
        
        logger.info(f"Deleting old pitcher_game_stats rows for {target_date}...")
        delete_job = bq.query(delete_sql)
        deleted_rows = delete_job.result()
        logger.info(f"✅ Deleted rows")
        
        # Step 2: Compute pitcher stats from statcast
        # Aggregates pitch-level statcast data by pitcher to create game-level stats
        new_data = compute_pitcher_stats_from_statcast(target_date)
        
        if not new_data:
            logger.info(f"No new pitcher stats computed for {target_date}")
            return {
                "status": "success",
                "message": f"No new pitcher stats for {target_date}",
                "date": target_date.isoformat(),
                "rows_inserted": 0,
                "rows_deleted": 0
            }
        
        # Step 3: Insert new data
        table_ref = bq.get_table(pitcher_stats_table)
        errors = bq.insert_rows_json(table_ref, new_data)
        if errors:
            logger.error(f"Insert errors: {errors}")
            return {"status": "error", "message": f"Insert errors: {errors}"}, 400
        
        logger.info(f"✅ Inserted {len(new_data)} pitcher_game_stats rows for {target_date}")
        
        return {
            "status": "success",
            "message": f"Updated pitcher_game_stats for {target_date}",
            "date": target_date.isoformat(),
            "rows_inserted": len(new_data),
            "rows_deleted": 0  # Can query for actual deleted count if needed
        }
        
    except Exception as e:
        logger.error(f"Error updating pitcher_game_stats: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "date": target_date.isoformat()
        }, 500


@functions_framework.http
def update_statcast_2026(request):
    """
    Update statcast_pitches for 2026 season.
    
    Triggered by Cloud Scheduler at 06:00 UTC daily
    
    Process:
      1. DELETE rows for yesterday (to prevent duplicates)
      2. FETCH pitch-level statcast data from external source
      3. INSERT fresh statcast data into mlb_2026_season.statcast_pitches
    
    Data sources:
      - Baseball Savant (https://baseballsavant.mlb.com)
      - MLB Stats API
      - Statcast data feed files
    
    Returns: JSON with status, date, rows_inserted, rows_deleted
    """
    bq = bigquery.Client(project=PROJECT_ID)
    
    target_date = date.today() - timedelta(days=1)
    
    logger.info(f"Updating statcast_pitches for {target_date}")
    
    try:
        statcast_table = f"{PROJECT_ID}.{SEASON_2026_DATASET}.statcast_pitches"
        
        # Step 1: Delete old rows to avoid duplicates
        delete_sql = f"""
        DELETE FROM `{statcast_table}`
        WHERE DATE(game_date) = '{target_date.isoformat()}'
        """
        
        logger.info(f"Deleting old statcast rows for {target_date}...")
        delete_job = bq.query(delete_sql)
        deleted_rows = delete_job.result()
        logger.info(f"✅ Deleted rows")
        
        # Step 2: Fetch new data from source
        new_data = fetch_statcast_from_baseball_savant(target_date)
        
        if not new_data:
            logger.info(f"No new statcast data available for {target_date}")
            return {
                "status": "success",
                "message": f"No new statcast data for {target_date}",
                "date": target_date.isoformat(),
                "rows_inserted": 0,
                "rows_deleted": 0
            }
        
        # Step 3: Insert new data
        table_ref = bq.get_table(statcast_table)
        errors = bq.insert_rows_json(table_ref, new_data)
        if errors:
            logger.error(f"Insert errors: {errors}")
            return {"status": "error", "message": f"Insert errors: {errors}"}, 400
        
        logger.info(f"✅ Inserted {len(new_data)} statcast rows for {target_date}")
        
        return {
            "status": "success",
            "message": f"Updated statcast for {target_date}",
            "date": target_date.isoformat(),
            "rows_inserted": len(new_data),
            "rows_deleted": 0
        }
        
    except Exception as e:
        logger.error(f"Error updating statcast: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "date": target_date.isoformat()
        }, 500


@functions_framework.http
def rebuild_v7_features(request):
    """
    Rebuild V7 matchup features after all data sources updated.
    
    Triggered by Cloud Scheduler at 08:00 UTC daily
    
    Process:
      1. Import V7FeatureBuilder from src.build_v7_features
      2. Call builder.run_for_date(yesterday) to rebuild features
      3. Builder queries unified historical + 2026 data via UNION
      4. Inserts features into matchup_v7_features table
    
    Dependencies:
      - src/build_v7_features.py (V7FeatureBuilder class)
      - BigQuery credentials and access to both datasets
    
    Returns: JSON with status, games_processed, features_created
    """
    target_date = date.today() - timedelta(days=1)
    
    logger.info(f"Rebuilding V7 features for {target_date}")
    
    try:
        # Import V7FeatureBuilder from src module
        import sys
        import os
        # Add parent directory to path so we can import src
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from src.build_v7_features import V7FeatureBuilder
        builder = V7FeatureBuilder(dry_run=False)
        result = builder.run_for_date(target_date)
        
        logger.info(f"V7 features rebuilt successfully for {target_date}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error rebuilding V7 features: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "date": target_date.isoformat()
        }, 500


@functions_framework.http
def predict_today_games(request):
    """
    Generate predictions for today's games.
    
    Triggered by Cloud Scheduler at 12:00 UTC daily
    
    Process:
      1. Import prediction engine from src.predict_today_games
      2. Call run_predictions() to generate predictions for today
      3. Uses V7 features built in previous step
      4. Inserts predictions into game_predictions table
    
    Dependencies:
      - src/predict_today_games.py (run_predictions() function)
      - V7 features already built and in matchup_v7_features
      - Model artifacts (ML model weights, encoders, scalers)
    
    Returns: JSON with status, predictions_generated
    """
    logger.info(f"Predicting games for {date.today()}")
    
    try:
        # TODO: Import prediction engine (requires Cloud Function to have src in path)
        # For now, return placeholder response
        
        # from src.predict_today_games import run_predictions
        # result = run_predictions()
        
        logger.warning("Prediction engine import not yet configured in Cloud Function - requires code deployment")
        
        # Placeholder response (mimics expected structure)
        result = {
            "status": "success",
            "predictions_generated": 0,
            "date": date.today().isoformat(),
            "message": "Not yet deployed - requires predict_today_games.py in Cloud Function code"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "date": date.today().isoformat()
        }, 500
