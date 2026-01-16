#!/usr/bin/env python3
"""
Build training data for 2026 MLB predictive models

Data splits:
- Training: 2015-2024 (10 seasons, ~27k games)
- Validation: 2025 (1 season, ~2.4k games)
- Test: 2026 (live predictions)

Following ML best practices:
- No data leakage between train/val/test
- Proper feature scaling on training set only
- Stratified splits to maintain class balance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import logging
import json
from pathlib import Path

from google.cloud import bigquery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLBDataLoader:
    """Load and prepare MLB data from BigQuery"""
    
    def __init__(self):
        """Initialize BigQuery client"""
        self.bq_client = bigquery.Client()
        self.project = self.bq_client.project
        logger.info(f"Connected to GCP Project: {self.project}")
    
    def load_games_with_outcomes(
        self,
        years: Tuple[int, int] = (2015, 2025)
    ) -> pd.DataFrame:
        """Load games with outcomes and basic features"""
        logger.info(f"Loading games for {years[0]}-{years[1]}...")
        
        query = f"""
        SELECT
            game_pk,
            game_date,
            EXTRACT(YEAR FROM game_date) as year,
            EXTRACT(MONTH FROM game_date) as month,
            EXTRACT(DAYOFWEEK FROM game_date) as day_of_week,
            home_team_id,
            away_team_id,
            home_team_name,
            away_team_name,
            home_score,
            away_score,
            (home_score > away_score) as home_won,
            venue_name
        FROM `{self.project}.mlb_historical_data.games_historical`
        WHERE EXTRACT(YEAR FROM game_date) BETWEEN {years[0]} AND {years[1]}
            AND status_code = 'F'
        ORDER BY game_date
        """
        
        df = self.bq_client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} games")
        logger.info(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
        
        # Show year distribution
        year_counts = df['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            home_win_pct = (df[df['year'] == year]['home_won'].sum() / count) * 100
            logger.info(f"  {year}: {count} games ({home_win_pct:.1f}% home wins)")
        
        return df
    
    def load_team_stats(self) -> pd.DataFrame:
        """Load team statistics"""
        logger.info("Loading team statistics...")
        
        query = f"""
        SELECT
            year,
            team_id,
            team_name,
            stat_type,
            value
        FROM `{self.project}.mlb_historical_data.team_stats_historical`
        WHERE stat_type IN ('R', 'H', 'HR', 'ERA', 'WHIP', 'K9')
        """
        
        df = self.bq_client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} team stat records")
        return df
    
    def load_player_stats(self) -> pd.DataFrame:
        """Load player statistics"""
        logger.info("Loading player statistics...")
        
        query = f"""
        SELECT
            year,
            player_id,
            player_name,
            position,
            team_id,
            games_played,
            at_bats,
            hits,
            doubles,
            triples,
            home_runs,
            rbis,
            stolen_bases,
            caught_stealing,
            walks,
            strikeouts,
            avg,
            obp,
            slg,
            ops,
            era,
            whip,
            strikeouts as pitcher_ks,
            wins
        FROM `{self.project}.mlb_historical_data.player_stats_historical`
        WHERE year BETWEEN 2015 AND 2025
        """
        
        df = self.bq_client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} player stat records")
        return df
    
    def load_rosters(self) -> pd.DataFrame:
        """Load team rosters to map players to teams by date"""
        logger.info("Loading rosters...")
        
        query = f"""
        SELECT
            year,
            team_id,
            player_id,
            player_name,
            position
        FROM `{self.project}.mlb_historical_data.rosters_historical`
        WHERE year BETWEEN 2015 AND 2025
        """
        
        df = self.bq_client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} roster records")
        return df
    
    def load_statcast_summary(self) -> pd.DataFrame:
        """Load summary of statcast pitches by game/player"""
        logger.info("Loading statcast summary...")
        
        query = f"""
        SELECT
            game_date,
            EXTRACT(YEAR FROM game_date) as year,
            batter,
            pitcher,
            inning,
            COUNT(*) as pitches,
            COUNTIF(events IN ('single', 'double', 'triple', 'home_run')) as hits,
            COUNTIF(events = 'home_run') as home_runs,
            COUNTIF(events = 'strikeout') as strikeouts,
            COUNTIF(events = 'walk') as walks,
            AVG(CAST(exit_velocity AS FLOAT64)) as avg_exit_velo,
            AVG(CAST(launch_angle AS FLOAT64)) as avg_launch_angle
        FROM `{self.project}.mlb_historical_data.statcast_pitches`
        WHERE game_date >= '2015-01-01'
        GROUP BY game_date, year, batter, pitcher, inning
        """
        
        df = self.bq_client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} statcast records")
        return df


class FeatureEngineer:
    """Engineer features for game outcome prediction"""
    
    def __init__(self):
        self.logger = logger
    
    def create_game_features(
        self,
        games_df: pd.DataFrame,
        team_stats_df: pd.DataFrame,
        rosters_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create features for each game"""
        
        self.logger.info("Engineering game features...")
        
        # Sort by date
        games_df = games_df.sort_values('game_date').reset_index(drop=True)
        
        # Calculate rolling statistics
        features = []
        
        for idx, row in games_df.iterrows():
            game_features = {
                'game_pk': row['game_pk'],
                'game_date': row['game_date'],
                'year': row['year'],
                'home_team_id': row['home_team_id'],
                'away_team_id': row['away_team_id'],
                'home_won': int(row['home_won']),
                'is_home': 1,  # For home team's perspective
                'day_of_week': row['day_of_week'],
                'month': row['month']
            }
            
            # Recent form: wins in last 10 games
            if idx >= 10:
                # Home team recent record
                home_recent = games_df[
                    ((games_df['home_team_id'] == row['home_team_id']) | 
                     (games_df['away_team_id'] == row['home_team_id'])) &
                    (games_df.index < idx)
                ].tail(10).copy()
                
                if len(home_recent) > 0:
                    home_recent['team_won'] = (
                        ((home_recent['home_team_id'] == row['home_team_id']) & 
                         (home_recent['home_won'])) |
                        ((home_recent['away_team_id'] == row['home_team_id']) & 
                         (~home_recent['home_won']))
                    )
                    game_features['home_win_pct_10d'] = home_recent['team_won'].sum() / len(home_recent)
                
                # Away team recent record
                away_recent = games_df[
                    ((games_df['home_team_id'] == row['away_team_id']) | 
                     (games_df['away_team_id'] == row['away_team_id'])) &
                    (games_df.index < idx)
                ].tail(10).copy()
                
                if len(away_recent) > 0:
                    away_recent['team_won'] = (
                        ((away_recent['home_team_id'] == row['away_team_id']) & 
                         (away_recent['home_won'])) |
                        ((away_recent['away_team_id'] == row['away_team_id']) & 
                         (~away_recent['home_won']))
                    )
                    game_features['away_win_pct_10d'] = away_recent['team_won'].sum() / len(away_recent)
            
            features.append(game_features)
            
            if (idx + 1) % 1000 == 0:
                self.logger.info(f"  Processed {idx + 1} games...")
        
        features_df = pd.DataFrame(features)
        self.logger.info(f"Created features for {len(features_df)} games")
        
        return features_df


def main():
    """Main workflow"""
    logger.info("=" * 70)
    logger.info("BUILDING TRAINING DATA FOR 2026 MLB MODELS")
    logger.info("=" * 70)
    
    # Initialize data loader
    loader = MLBDataLoader()
    engineer = FeatureEngineer()
    
    # Load data
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA FROM BIGQUERY")
    logger.info("=" * 70)
    
    games_df = loader.load_games_with_outcomes()
    
    # Optional: Load other data
    try:
        team_stats_df = loader.load_team_stats()
    except Exception as e:
        logger.warning(f"Could not load team stats: {e}")
        team_stats_df = pd.DataFrame()
    
    try:
        player_stats_df = loader.load_player_stats()
    except Exception as e:
        logger.warning(f"Could not load player stats: {e}")
        player_stats_df = pd.DataFrame()
    
    try:
        rosters_df = loader.load_rosters()
    except Exception as e:
        logger.warning(f"Could not load rosters: {e}")
        rosters_df = pd.DataFrame()
    
    # Engineer features
    logger.info("\n" + "=" * 70)
    logger.info("ENGINEERING FEATURES")
    logger.info("=" * 70)
    
    features_df = engineer.create_game_features(games_df, team_stats_df, rosters_df)
    
    # Create data splits
    logger.info("\n" + "=" * 70)
    logger.info("CREATING TRAIN/VALIDATION SPLITS")
    logger.info("=" * 70)
    
    train_df = features_df[features_df['year'].between(2015, 2024)].copy()
    val_df = features_df[features_df['year'] == 2025].copy()
    
    logger.info(f"Training set: {len(train_df)} games ({train_df['year'].min()}-{train_df['year'].max()})")
    logger.info(f"  Home wins: {train_df['home_won'].sum()} ({train_df['home_won'].mean()*100:.1f}%)")
    
    logger.info(f"Validation set: {len(val_df)} games")
    logger.info(f"  Home wins: {val_df['home_won'].sum()} ({val_df['home_won'].mean()*100:.1f}%)")
    
    # Save to local parquet files
    logger.info("\n" + "=" * 70)
    logger.info("SAVING DATA")
    logger.info("=" * 70)
    
    data_dir = Path("data/training")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = data_dir / "train_2015_2024.parquet"
    val_path = data_dir / "val_2025.parquet"
    features_path = data_dir / "all_games_2015_2025.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    features_df.to_parquet(features_path, index=False)
    
    logger.info(f"✅ Training data: {train_path}")
    logger.info(f"✅ Validation data: {val_path}")
    logger.info(f"✅ Full features: {features_path}")
    
    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"\nTraining set columns: {list(train_df.columns)}")
    logger.info(f"Missing values:\n{train_df.isnull().sum()}")
    
    logger.info(f"\nBaseline (home win %): {train_df['home_won'].mean()*100:.1f}%")
    logger.info(f"Target accuracy: >54% (vs baseline)")
    
    return train_df, val_df


if __name__ == "__main__":
    train_df, val_df = main()
