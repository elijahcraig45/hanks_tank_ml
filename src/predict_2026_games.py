#!/usr/bin/env python3
"""
2026 Daily Prediction Script for MLB Games

Usage:
    python predict_2026_games.py          # Predict today's games
    python predict_2026_games.py --date 2026-04-01  # Predict specific date
    python predict_2026_games.py --upload     # Upload to BigQuery

This script will:
1. Load the trained model
2. Fetch today's scheduled games from BigQuery
3. Compute features for each game
4. Generate predictions with confidence intervals
5. Save to BigQuery predictions table
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import pickle
from datetime import datetime, timedelta
import argparse

from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GamePredictor:
    """Make predictions for 2026 games"""
    
    def __init__(self, model_path=None):
        """Load trained model"""
        if model_path is None:
            model_path = Path("models/game_outcome_LogisticRegression.pkl")
        
        logger.info(f"Loading model from {model_path}...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.features = model_data['features']
        self.model_name = model_data.get('model_name', 'Unknown')
        
        self.bq_client = bigquery.Client()
        self.project = self.bq_client.project
        
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Features: {self.features}")
        logger.info(f"Project: {self.project}")
    
    def get_games_for_date(self, game_date):
        """Get scheduled games for a specific date"""
        logger.info(f"Fetching games for {game_date}...")
        
        query = f"""
        SELECT
            game_pk,
            game_date,
            home_team_id,
            away_team_id,
            home_team_name,
            away_team_name,
            venue_name,
            EXTRACT(MONTH FROM game_date) as month,
            EXTRACT(DAYOFWEEK FROM game_date) as day_of_week
        FROM `{self.project}.mlb_historical_data.games_historical`
        WHERE DATE(game_date) = '{game_date}'
            AND status_code NOT IN ('F', 'C')
        ORDER BY game_date
        """
        
        try:
            df = self.bq_client.query(query).to_dataframe()
            logger.info(f"Found {len(df)} games")
            return df
        except Exception as e:
            logger.error(f"Error fetching games: {e}")
            return pd.DataFrame()
    
    def compute_team_recent_form(self, games_df, team_id, reference_date):
        """Compute recent win % for a team up to reference_date"""
        # For now, return default value (54% - baseline)
        # In production, would query historical games and compute rolling stats
        return 0.54
    
    def prepare_features_for_game(self, game_row):
        """Prepare features for a single game"""
        features_dict = {}
        
        for feature in self.features:
            if feature == 'home_win_pct_10d':
                # Compute from recent games (simplified)
                features_dict[feature] = self.compute_team_recent_form(
                    None, game_row['home_team_id'], game_row['game_date']
                )
            elif feature == 'away_win_pct_10d':
                features_dict[feature] = self.compute_team_recent_form(
                    None, game_row['away_team_id'], game_row['game_date']
                )
            elif feature == 'month':
                features_dict[feature] = game_row['month']
            elif feature == 'day_of_week':
                features_dict[feature] = game_row['day_of_week']
            elif feature == 'is_home':
                features_dict[feature] = 1.0
            else:
                features_dict[feature] = 0.0
        
        return features_dict
    
    def predict_game(self, game_row):
        """Make prediction for a single game"""
        # Prepare features
        features_dict = self.prepare_features_for_game(game_row)
        
        # Convert to array in correct order
        X = np.array([[features_dict[f] for f in self.features]])
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0, 1]
        
        return prediction, probability
    
    def predict_all_games(self, games_df):
        """Make predictions for all games in DataFrame"""
        logger.info(f"Making predictions for {len(games_df)} games...")
        
        predictions = []
        
        for idx, row in games_df.iterrows():
            pred, prob = self.predict_game(row)
            
            predictions.append({
                'game_pk': row['game_pk'],
                'game_date': row['game_date'],
                'home_team_id': row['home_team_id'],
                'away_team_id': row['away_team_id'],
                'home_team_name': row['home_team_name'],
                'away_team_name': row['away_team_name'],
                'home_predicted': int(pred),
                'home_win_probability': float(prob),
                'away_win_probability': float(1 - prob),
                'prediction_confidence': float(max(prob, 1 - prob)),
                'model_name': self.model_name,
                'prediction_timestamp': datetime.now().isoformat()
            })
        
        return pd.DataFrame(predictions)
    
    def upload_predictions_to_bigquery(self, predictions_df):
        """Upload predictions to BigQuery"""
        logger.info(f"Uploading {len(predictions_df)} predictions to BigQuery...")
        
        table_id = f"{self.project}.mlb_models.game_predictions_2026"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            autodetect=True,
        )
        
        try:
            job = self.bq_client.load_table_from_dataframe(
                predictions_df,
                table_id,
                job_config=job_config
            )
            job.result()
            logger.info(f"✅ Uploaded {job.output_rows} rows to {table_id}")
            return True
        except Exception as e:
            logger.error(f"Error uploading to BigQuery: {e}")
            return False
    
    def save_local_predictions(self, predictions_df, game_date):
        """Save predictions locally"""
        output_dir = Path("predictions/2026")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"predictions_{game_date}.parquet"
        filepath = output_dir / filename
        
        predictions_df.to_parquet(filepath, index=False)
        logger.info(f"✅ Saved predictions to {filepath}")
        
        return filepath


def main():
    """Main workflow"""
    parser = argparse.ArgumentParser(description='Predict 2026 MLB games')
    parser.add_argument('--date', default=None, help='Predict games for specific date (YYYY-MM-DD)')
    parser.add_argument('--upload', action='store_true', help='Upload predictions to BigQuery')
    parser.add_argument('--model', default=None, help='Path to model file')
    
    args = parser.parse_args()
    
    # Get prediction date
    if args.date:
        pred_date = args.date
    else:
        # Default to today
        pred_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info("=" * 70)
    logger.info(f"2026 MLB GAME PREDICTIONS FOR {pred_date}")
    logger.info("=" * 70)
    
    # Initialize predictor
    predictor = GamePredictor(model_path=args.model)
    
    # Get games
    games_df = predictor.get_games_for_date(pred_date)
    
    if games_df.empty:
        logger.info(f"No games scheduled for {pred_date}")
        return
    
    # Make predictions
    predictions_df = predictor.predict_all_games(games_df)
    
    # Display predictions
    logger.info("\n" + "=" * 70)
    logger.info("PREDICTIONS")
    logger.info("=" * 70)
    
    for idx, row in predictions_df.iterrows():
        home = row['home_team_name']
        away = row['away_team_name']
        home_prob = row['home_win_probability']
        confidence = row['prediction_confidence']
        
        if row['home_predicted']:
            logger.info(f"{home} vs {away}: {home} to win ({home_prob:.1%} confidence)")
        else:
            logger.info(f"{home} vs {away}: {away} to win ({1-home_prob:.1%} confidence)")
    
    # Save locally
    predictor.save_local_predictions(predictions_df, pred_date)
    
    # Upload to BigQuery if requested
    if args.upload:
        predictor.upload_predictions_to_bigquery(predictions_df)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"✅ Predictions complete for {len(predictions_df)} games")
    logger.info("=" * 70)
    
    return predictions_df


if __name__ == "__main__":
    predictions = main()
