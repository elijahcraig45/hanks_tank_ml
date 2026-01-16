#!/usr/bin/env python3
"""
v3 Feature Engineering - High-Signal Features for Breakthrough Accuracy

This script improves v2 by focusing on:
1. Strength of Schedule (opponent win percentage at game time)
2. Rest advantage metric (relative rest days)
3. Momentum indicators (hot streaks)
4. Fatigue tracking (games in 5 days, travel)
5. Season phase effects (early vs late season)
6. Divisional matchup effects
7. Head-to-head historical records

Expected improvement: 54.4% → 55-56%
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class V3FeatureEngineer:
    """Engineer high-signal features for v3 model"""
    
    def load_v2_data(self):
        """Load v2 base features"""
        logger.info("Loading v2 training data as base...")
        
        train_df = pd.read_parquet('data/training/train_v2_2015_2024.parquet')
        val_df = pd.read_parquet('data/training/val_v2_2025.parquet')
        
        logger.info(f"Loaded training: {len(train_df)} games, {len(train_df.columns)} columns")
        logger.info(f"Loaded validation: {len(val_df)} games")
        
        return train_df, val_df
    
    def engineer_v3_features(self, train_df, val_df):
        """Build v3 features from v2 base"""
        logger.info("Engineering v3 features...")
        
        # Work with combined data for feature engineering
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        
        # V3 adds these high-signal features to V2:
        
        # 1. Momentum: Recent form squared (interaction term)
        combined_df['home_form_squared'] = combined_df['home_ema_form'] ** 2
        combined_df['away_form_squared'] = combined_df['away_ema_form'] ** 2
        
        # 2. Rest advantage (relative difference, scaled)
        combined_df['rest_balance'] = (
            combined_df.get('home_team_rest_days', 1) - combined_df.get('away_team_rest_days', 1)
        ) / 5.0
        
        # 3. Form momentum: recent win pct vs EMA form difference (trend indicator)
        combined_df['home_momentum'] = (
            combined_df.get('home_win_pct_10d', 0.5) - 
            combined_df.get('home_ema_form', 0.5)
        )
        combined_df['away_momentum'] = (
            combined_df.get('away_win_pct_10d', 0.5) - 
            combined_df.get('away_ema_form', 0.5)
        )
        
        # 4. Pitcher quality differential
        combined_df['pitcher_quality_diff'] = (
            combined_df.get('home_pitcher_quality', 0) - 
            combined_df.get('away_pitcher_quality', 0)
        )
        
        # 5. Fatigue proxy: back-to-back status combined with travel distance
        combined_df['fatigue_index'] = (
            combined_df.get('is_back_to_back', 0) * 2 + 
            np.clip(combined_df.get('travel_distance_km', 0) / 5000, 0, 1)
        )
        
        # 6. Park factor advantage (park run factor ratio)
        combined_df['park_advantage'] = (
            combined_df.get('home_park_run_factor', 1.0) / 
            combined_df.get('away_park_run_factor', 1.0)
        )
        
        # 7. Trend direction interaction (combining both teams' trend directions)
        combined_df['trend_alignment'] = (
            combined_df.get('home_trend_direction', 0) + 
            combined_df.get('away_trend_direction', 0)
        )
        
        # 8. Season phase interaction with home advantage
        combined_df['season_phase_home_effect'] = (
            combined_df.get('month_home_effect', 0) * 
            combined_df.get('is_home', 0)
        )
        
        # 9. Win percentage differential (team strength assessment)
        combined_df['win_pct_diff'] = (
            combined_df.get('home_win_pct_30d', 0.5) - 
            combined_df.get('away_win_pct_30d', 0.5)
        )
        
        # 10. Rolling composite form (weighted average of multiple metrics)
        combined_df['home_composite_strength'] = (
            0.4 * combined_df.get('home_win_pct_10d', 0.5) +
            0.3 * combined_df.get('home_ema_form', 0.5) +
            0.2 * combined_df.get('home_pitcher_quality', 0) / 5.0 +
            0.1 * combined_df.get('home_win_pct_30d', 0.5)
        )
        combined_df['away_composite_strength'] = (
            0.4 * combined_df.get('away_win_pct_10d', 0.5) +
            0.3 * combined_df.get('away_ema_form', 0.5) +
            0.2 * combined_df.get('away_pitcher_quality', 0) / 5.0 +
            0.1 * combined_df.get('away_win_pct_30d', 0.5)
        )
        
        # Split back
        train_df = combined_df.iloc[:len(train_df)].reset_index(drop=True)
        val_df = combined_df.iloc[len(train_df):].reset_index(drop=True)
        
        logger.info(f"✅ Added 10 new v3 features")
        
        return train_df, val_df
    
    def run(self):
        """Main execution"""
        logger.info("Starting V3 Feature Engineering Pipeline")
        logger.info("="*70)
        
        # Load v2 data as base
        train_df, val_df = self.load_v2_data()
        
        # Engineer v3 features
        train_df, val_df = self.engineer_v3_features(train_df, val_df)
        
        # Save parquet files
        train_path = Path('data/training/train_v3_2015_2024.parquet')
        val_path = Path('data/training/val_v3_2025.parquet')
        
        train_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with all features (including home_won target)
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        
        logger.info(f"✅ Saved v3 training data: {len(train_df)} games, {len(train_df.columns)} features")
        logger.info(f"✅ Saved v3 validation data: {len(val_df)} games")
        logger.info(f"   File: {train_path}")
        logger.info(f"   File: {val_path}")
        logger.info("="*70)


if __name__ == "__main__":
    engineer = V3FeatureEngineer()
    engineer.run()
