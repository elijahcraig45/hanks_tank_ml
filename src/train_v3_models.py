#!/usr/bin/env python3
"""
v3 Model Training - Compare with v1 and v2

Trains Logistic Regression, Random Forest, and XGBoost on v3 features.
Expected: 55%+ accuracy (vs v1: 54.0%, v2: 54.4%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class V3ModelTrainer:
    """Train v3 models and compare with v1/v2"""
    
    def __init__(self):
        self.v1_model = None
        self.v1_scaler = None
        self.v1_features = None
    
    def load_v1_model(self):
        """Load the production v1 model"""
        logger.info("Loading v1 model...")
        try:
            with open('models/game_outcome_LogisticRegression.pkl', 'rb') as f:
                v1_data = pickle.load(f)
                self.v1_model = v1_data['model']
                self.v1_scaler = v1_data['scaler']
                self.v1_features = v1_data['features']
                self.v1_accuracy = v1_data.get('val_accuracy', 0.54)
            logger.info(f"✅ v1 Model loaded ({self.v1_accuracy:.1%} accuracy)")
            logger.info(f"   Features: {self.v1_features}")
        except Exception as e:
            logger.error(f"Failed to load v1 model: {e}")
            return False
        return True
    
    def select_v3_features(self, train_df, val_df):
        """Select and prepare v3 features"""
        logger.info("\nSelecting v3 features...")
        
        # All available features
        v3_feature_candidates = [
            'home_win_pct_10d', 'away_win_pct_10d',
            'home_recent_win_pct', 'away_recent_win_pct',
            'home_pitcher_era_10d', 'away_pitcher_era_10d',
            'home_days_rest', 'away_days_rest',
            'home_rest_advantage',
            'home_hot_hand', 'away_hot_hand',
            'home_games_played', 'away_games_played',
            'month', 'is_early_season', 'is_late_season',
            'is_divisional_matchup',
            'h2h_home_win_pct',
            'games_in_5_days',
            'travel_distance_km',
            'is_home',
        ]
        
        # Select features that exist
        available_features = [f for f in v3_feature_candidates if f in train_df.columns]
        logger.info(f"Selected {len(available_features)} features")
        
        # Handle NaN
        for col in available_features:
            if train_df[col].isna().any():
                fill_val = train_df[col].mean() if pd.api.types.is_numeric_dtype(train_df[col]) else 0
                train_df[col] = train_df[col].fillna(fill_val)
                val_df[col] = val_df[col].fillna(fill_val)
        
        X_train = train_df[available_features].values.astype(float)
        X_val = val_df[available_features].values.astype(float)
        
        y_train = train_df['home_won'].values.astype(int)
        y_val = val_df['home_won'].values.astype(int)
        
        return X_train, X_val, y_train, y_val, available_features
    
    def train_v3_models(self, X_train, X_val, y_train, y_val, features):
        """Train v3 models"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING V3 MODELS")
        logger.info("="*70)
        
        results = {}
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 1. Logistic Regression
        logger.info("\nTraining v3 Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_acc = lr.score(X_val_scaled, y_val)
        lr_auc = np.mean(lr.predict(X_val_scaled) == y_val)
        logger.info(f"✅ v3 LR Accuracy: {lr_acc:.1%}")
        logger.info(f"   v3 LR AUC: {lr_auc:.3f}")
        
        results['LogisticRegression'] = {
            'model': lr,
            'scaler': scaler,
            'accuracy': lr_acc,
            'auc': lr_auc,
            'features': features
        }
        
        # 2. Random Forest
        logger.info("\nTraining v3 Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_acc = rf.score(X_val, y_val)
        rf_auc = np.mean(rf.predict(X_val) == y_val)
        logger.info(f"✅ v3 RF Accuracy: {rf_acc:.1%}")
        logger.info(f"   v3 RF AUC: {rf_auc:.3f}")
        
        results['RandomForest'] = {
            'model': rf,
            'accuracy': rf_acc,
            'auc': rf_auc,
            'features': features
        }
        
        # 3. XGBoost
        logger.info("\nTraining v3 XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_acc = xgb_model.score(X_val, y_val)
        xgb_auc = np.mean(xgb_model.predict(X_val) == y_val)
        logger.info(f"✅ v3 XGB Accuracy: {xgb_acc:.1%}")
        logger.info(f"   v3 XGB AUC: {xgb_auc:.3f}")
        
        results['XGBoost'] = {
            'model': xgb_model,
            'accuracy': xgb_acc,
            'auc': xgb_auc,
            'features': features
        }
        
        return results
    
    def get_v1_predictions(self, X_val):
        """Get v1 model predictions on validation set"""
        X_val_scaled = self.v1_scaler.transform(X_val)
        return self.v1_model.predict(X_val_scaled)
    
    def compare_models(self, results, y_val, X_val):
        """Compare v1 vs v2 vs v3 models"""
        logger.info("\n" + "="*70)
        logger.info("V1 vs V2 vs V3 COMPARISON")
        logger.info("="*70)
        
        # Get v1 accuracy
        v1_pred = self.v1_model.predict(self.v1_scaler.transform(X_val[:, :5]))  # v1 has 5 features
        v1_acc = np.mean(v1_pred == y_val)
        
        # Print comparison table
        print("\n")
        print("Model                     Accuracy        AUC        Improvement")
        print("-" * 70)
        print(f"v1 (5 features)           54.0%           0.543      BASELINE")
        print(f"v2 LogisticRegression     54.4%           0.534      +0.4% ✅")
        
        for model_name, model_data in results.items():
            acc = model_data['accuracy']
            auc = model_data['auc']
            improvement = (acc - 0.540) * 100
            status = "✅ IMPROVED" if acc > 0.544 else ("⚠️  MARGINAL" if acc > 0.540 else "❌ WORSE")
            print(f"v3 {model_name:<22} {acc:.1%}           {auc:.3f}      {improvement:+.1f}% {status}")
        
        # Determine best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_name, best_data = best_model
        best_acc = best_data['accuracy']
        
        print("\n" + "="*70)
        logger.info(f"BEST V3 MODEL: {best_name}")
        logger.info(f"Accuracy: {best_acc:.1%} (improvement: {(best_acc - 0.540)*100:+.1f}%)")
        logger.info("="*70)
        
        return best_model
    
    def save_best_model(self, model_name, model_data):
        """Save the best v3 model"""
        output_path = Path(f"models/game_outcome_v3_{model_name}.pkl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data['timestamp'] = datetime.now()
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Saved v3 model to {output_path}")
    
    def run(self):
        """Main execution"""
        logger.info("="*70)
        logger.info("V3 MODEL TRAINING & COMPARISON")
        logger.info("="*70)
        
        # Load v1 model
        if not self.load_v1_model():
            return
        
        # Load v3 training data
        logger.info("\nLoading v3 training data...")
        train_df = pd.read_parquet('data/training/train_v3_2015_2024.parquet')
        val_df = pd.read_parquet('data/training/val_v3_2025.parquet')
        
        logger.info(f"✅ v3 Training set: {len(train_df)} games, {len(train_df.columns)-1} features")
        logger.info(f"✅ v3 Validation set: {len(val_df)} games")
        
        # Select features
        X_train, X_val, y_train, y_val, features = self.select_v3_features(train_df, val_df)
        
        # Train models
        results = self.train_v3_models(X_train, X_val, y_train, y_val, features)
        
        # Compare with v1
        best_model, best_data = self.compare_models(results, y_val, X_val)
        
        # Save best model
        self.save_best_model(best_model, best_data)
        
        logger.info("\n" + "="*70)
        logger.info("NEXT STEPS")
        logger.info("="*70)
        logger.info("1. If v3 accuracy > 55%: Deploy v3 as production model")
        logger.info("2. If v3 accuracy = 54.4-55%: Continue feature engineering")
        logger.info("3. Consider ensemble of v1 + v3 for stability")
        

if __name__ == '__main__':
    trainer = V3ModelTrainer()
    trainer.run()
