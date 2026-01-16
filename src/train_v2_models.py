#!/usr/bin/env python3
"""
v2 Model Training - Compare v1 vs v2 Performance

This script will:
1. Load v2 training data with enhanced features
2. Train models with v2 features
3. Compare v1 vs v2 accuracy on same validation set
4. Show which features are most important
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import pickle
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class V2ModelTrainer:
    """Train v2 models with enhanced features"""
    
    def __init__(self):
        self.v1_model = None
        self.v2_model = None
        self.scaler_v1 = None
        self.scaler_v2 = None
        self.feature_names_v1 = []
        self.feature_names_v2 = []
    
    def load_v1_model(self):
        """Load v1 model for comparison"""
        logger.info("Loading v1 model...")
        
        with open('models/game_outcome_LogisticRegression.pkl', 'rb') as f:
            v1_data = pickle.load(f)
        
        self.v1_model = v1_data['model']
        self.scaler_v1 = v1_data['scaler']
        self.feature_names_v1 = v1_data['features']
        
        logger.info(f"✅ v1 Model loaded (54.0% accuracy)")
        logger.info(f"   Features: {self.feature_names_v1}")
        
        return self.v1_model
    
    def load_v2_data(self):
        """Load v2 training and validation data"""
        logger.info("Loading v2 training data...")
        
        train_path = Path('data/training/train_v2_2015_2024.parquet')
        val_path = Path('data/training/val_v2_2025.parquet')
        
        if not train_path.exists():
            logger.error(f"v2 training data not found: {train_path}")
            logger.info("Run: python src/build_v2_features.py")
            return None, None
        
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        
        logger.info(f"✅ v2 Training set: {len(train_df)} games, {len(train_df.columns)} features")
        logger.info(f"✅ v2 Validation set: {len(val_df)} games")
        
        return train_df, val_df
    
    def select_v2_features(self, train_df, val_df):
        """Select best v2 features (exclude target, IDs, and original temporal)"""
        logger.info("\nSelecting v2 features...")
        
        # Exclude non-features
        exclude = ['game_pk', 'game_date', 'year', 'home_team_id', 'away_team_id', 'home_won']
        
        feature_names = [col for col in train_df.columns if col not in exclude]
        
        logger.info(f"Selected {len(feature_names)} features for v2:")
        for feat in sorted(feature_names):
            logger.info(f"  - {feat}")
        
        # Fill NaN values with mean
        for col in feature_names:
            mean_val = train_df[col].mean()
            train_df[col] = train_df[col].fillna(mean_val)
            val_df[col] = val_df[col].fillna(mean_val)
        
        self.feature_names_v2 = feature_names
        
        X_train = train_df[feature_names].values
        X_val = val_df[feature_names].values
        
        y_train = train_df['home_won'].values.astype(int)
        y_val = val_df['home_won'].values.astype(int)
        
        return X_train, X_val, y_train, y_val
    
    def train_v2_models(self, X_train, X_val, y_train, y_val):
        """Train v2 models"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING V2 MODELS")
        logger.info("="*70)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        self.scaler_v2 = scaler
        
        results = {}
        
        # Logistic Regression
        logger.info("\nTraining v2 Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        
        y_pred_v2_lr = lr.predict(X_val_scaled)
        y_prob_v2_lr = lr.predict_proba(X_val_scaled)[:, 1]
        
        acc_v2_lr = accuracy_score(y_val, y_pred_v2_lr)
        auc_v2_lr = roc_auc_score(y_val, y_prob_v2_lr)
        
        logger.info(f"✅ v2 LR Accuracy: {acc_v2_lr:.1%}")
        logger.info(f"   v2 LR AUC: {auc_v2_lr:.3f}")
        
        results['LogisticRegression'] = {
            'model': lr,
            'accuracy': acc_v2_lr,
            'auc': auc_v2_lr,
            'y_pred': y_pred_v2_lr,
            'y_prob': y_prob_v2_lr
        }
        
        # Random Forest
        logger.info("\nTraining v2 Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        y_pred_v2_rf = rf.predict(X_val)
        y_prob_v2_rf = rf.predict_proba(X_val)[:, 1]
        
        acc_v2_rf = accuracy_score(y_val, y_pred_v2_rf)
        auc_v2_rf = roc_auc_score(y_val, y_prob_v2_rf)
        
        logger.info(f"✅ v2 RF Accuracy: {acc_v2_rf:.1%}")
        logger.info(f"   v2 RF AUC: {auc_v2_rf:.3f}")
        
        results['RandomForest'] = {
            'model': rf,
            'accuracy': acc_v2_rf,
            'auc': auc_v2_rf,
            'y_pred': y_pred_v2_rf,
            'y_prob': y_prob_v2_rf
        }
        
        # XGBoost
        logger.info("\nTraining v2 XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred_v2_xgb = xgb_model.predict(X_val)
        y_prob_v2_xgb = xgb_model.predict_proba(X_val)[:, 1]
        
        acc_v2_xgb = accuracy_score(y_val, y_pred_v2_xgb)
        auc_v2_xgb = roc_auc_score(y_val, y_prob_v2_xgb)
        
        logger.info(f"✅ v2 XGB Accuracy: {acc_v2_xgb:.1%}")
        logger.info(f"   v2 XGB AUC: {auc_v2_xgb:.3f}")
        
        results['XGBoost'] = {
            'model': xgb_model,
            'accuracy': acc_v2_xgb,
            'auc': auc_v2_xgb,
            'y_pred': y_pred_v2_xgb,
            'y_prob': y_prob_v2_xgb
        }
        
        return results
    
    def compare_v1_vs_v2(self, X_val, y_val, v2_results):
        """Compare v1 and v2 performance"""
        logger.info("\n" + "="*70)
        logger.info("V1 vs V2 COMPARISON")
        logger.info("="*70)
        
        # Get v1 predictions on same data
        logger.info("\nGetting v1 predictions on validation set...")
        
        # Extract v1 features
        v1_feature_cols = ['home_win_pct_10d', 'away_win_pct_10d', 'month', 'day_of_week', 'is_home']
        
        # Assuming X_val is full v2 features, extract v1 subset
        # For now, we'll recompute from val data
        
        val_path = Path('data/training/val_v2_2025.parquet')
        val_full = pd.read_parquet(val_path)
        
        X_val_v1 = val_full[v1_feature_cols].values
        X_val_v1_scaled = self.scaler_v1.transform(X_val_v1)
        
        y_pred_v1 = self.v1_model.predict(X_val_v1_scaled)
        y_prob_v1 = self.v1_model.predict_proba(X_val_v1_scaled)[:, 1]
        
        acc_v1 = accuracy_score(y_val, y_pred_v1)
        auc_v1 = roc_auc_score(y_val, y_prob_v1)
        
        logger.info(f"\n{'Model':<20} {'Accuracy':<15} {'AUC':<10} {'Improvement':<15}")
        logger.info("-" * 60)
        
        logger.info(f"{'v1 (5 features)':<20} {acc_v1:.1%}           {auc_v1:.3f}     BASELINE")
        
        for model_name, result in v2_results.items():
            acc_v2 = result['accuracy']
            auc_v2 = result['auc']
            improvement = (acc_v2 - acc_v1) * 100
            
            status = "✅ IMPROVED" if improvement > 0 else "❌ WORSE"
            
            logger.info(f"{'v2 ' + model_name:<20} {acc_v2:.1%}           {auc_v2:.3f}     {improvement:+.1f}% {status}")
        
        # Find best
        best_model = max(v2_results, key=lambda x: v2_results[x]['accuracy'])
        best_acc = v2_results[best_model]['accuracy']
        improvement = (best_acc - acc_v1) * 100
        
        logger.info("\n" + "="*70)
        logger.info(f"BEST V2 MODEL: {best_model}")
        logger.info(f"Accuracy: {best_acc:.1%} (improvement: {improvement:+.1f}%)")
        logger.info("="*70)
        
        return acc_v1, v2_results


def main():
    """Main workflow"""
    logger.info("="*70)
    logger.info("V2 MODEL TRAINING & COMPARISON")
    logger.info("="*70)
    
    trainer = V2ModelTrainer()
    
    # Load models and data
    v1_model = trainer.load_v1_model()
    train_v2, val_v2 = trainer.load_v2_data()
    
    if train_v2 is None:
        logger.error("Could not load v2 data. Aborting.")
        return
    
    # Select features
    X_train, X_val, y_train, y_val = trainer.select_v2_features(train_v2, val_v2)
    
    # Train v2 models
    v2_results = trainer.train_v2_models(X_train, X_val, y_train, y_val)
    
    # Compare
    acc_v1, v2_comparison = trainer.compare_v1_vs_v2(X_val, y_val, v2_results)
    
    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    logger.info("1. If v2 accuracy > 55%: Deploy v2 as production model")
    logger.info("2. If v2 accuracy = 54-55%: Iterate with more features")
    logger.info("3. Add pitcher data, rest days, strength trends for v3")


if __name__ == "__main__":
    main()
