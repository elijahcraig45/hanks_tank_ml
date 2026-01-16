#!/usr/bin/env python3
"""
Train game outcome prediction models for 2026 MLB season

Goal: Predict winner of each game with >54% accuracy

Models to train:
1. Logistic Regression (baseline, interpretable)
2. Random Forest (ensemble, non-linear)
3. XGBoost (gradient boosting, state-of-the-art)
4. Voting Ensemble (combine best models)

Evaluation:
- Accuracy on 2025 validation set
- Cross-validation on training set
- Feature importance
- Calibration (probability alignment)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import pickle
from datetime import datetime

# ML Libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, log_loss, roc_curve, auc
)
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GameOutcomeModel:
    """Train and evaluate game outcome prediction models"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def load_data(self):
        """Load training and validation data"""
        logger.info("Loading training data...")
        
        train_path = Path("data/training/train_2015_2024.parquet")
        val_path = Path("data/training/val_2025.parquet")
        
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        
        logger.info(f"Training set: {len(train_df)} games")
        logger.info(f"Validation set: {len(val_df)} games")
        
        return train_df, val_df
    
    def prepare_features(self, train_df, val_df):
        """Prepare features and targets"""
        logger.info("\nPreparing features...")
        
        # Select features for modeling
        feature_cols = [
            'home_win_pct_10d',
            'away_win_pct_10d',
            'month',
            'day_of_week',
            'is_home'
        ]
        
        # Fill NaN with mean
        for col in feature_cols:
            if col in train_df.columns:
                mean_val = train_df[col].mean()
                train_df[col] = train_df[col].fillna(mean_val)
                val_df[col] = val_df[col].fillna(mean_val)
        
        # Get X and y
        X_train = train_df[feature_cols].values
        y_train = train_df['home_won'].values.astype(int)
        
        X_val = val_df[feature_cols].values
        y_val = val_df['home_won'].values.astype(int)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        self.feature_names = feature_cols
        
        logger.info(f"Features: {feature_cols}")
        logger.info(f"X_train shape: {X_train_scaled.shape}")
        logger.info(f"X_val shape: {X_val_scaled.shape}")
        logger.info(f"Target distribution (train): {np.bincount(y_train)}")
        logger.info(f"Home win % (train): {y_train.mean()*100:.1f}%")
        logger.info(f"Home win % (val): {y_val.mean()*100:.1f}%")
        
        return X_train_scaled, X_val_scaled, y_train, y_val
    
    def train_logistic_regression(self, X_train, X_val, y_train, y_val):
        """Train logistic regression baseline"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING: LOGISTIC REGRESSION (BASELINE)")
        logger.info("="*70)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        
        y_pred_val = model.predict(X_val)
        y_prob_val = model.predict_proba(X_val)[:, 1]
        
        # Metrics
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_val = accuracy_score(y_val, y_pred_val)
        auc_val = roc_auc_score(y_val, y_prob_val)
        
        logger.info(f"Training Accuracy: {acc_train:.1%}")
        logger.info(f"Validation Accuracy: {acc_val:.1%}")
        logger.info(f"Validation AUC: {auc_val:.3f}")
        
        # Feature importance (coefficients)
        logger.info("\nFeature Importance:")
        for fname, coef in sorted(zip(self.feature_names, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
            logger.info(f"  {fname}: {coef:.4f}")
        
        results = {
            'model': model,
            'train_acc': acc_train,
            'val_acc': acc_val,
            'val_auc': auc_val,
            'y_pred': y_pred_val,
            'y_prob': y_prob_val
        }
        
        return results
    
    def train_random_forest(self, X_train, X_val, y_train, y_val):
        """Train random forest"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING: RANDOM FOREST")
        logger.info("="*70)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        
        y_pred_val = model.predict(X_val)
        y_prob_val = model.predict_proba(X_val)[:, 1]
        
        # Metrics
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_val = accuracy_score(y_val, y_pred_val)
        auc_val = roc_auc_score(y_val, y_prob_val)
        
        logger.info(f"Training Accuracy: {acc_train:.1%}")
        logger.info(f"Validation Accuracy: {acc_val:.1%}")
        logger.info(f"Validation AUC: {auc_val:.3f}")
        
        # Feature importance
        logger.info("\nFeature Importance:")
        for fname, imp in sorted(zip(self.feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True):
            logger.info(f"  {fname}: {imp:.4f}")
        
        results = {
            'model': model,
            'train_acc': acc_train,
            'val_acc': acc_val,
            'val_auc': auc_val,
            'y_pred': y_pred_val,
            'y_prob': y_prob_val
        }
        
        return results
    
    def train_xgboost(self, X_train, X_val, y_train, y_val):
        """Train XGBoost"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING: XGBOOST")
        logger.info("="*70)
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        
        y_pred_val = model.predict(X_val)
        y_prob_val = model.predict_proba(X_val)[:, 1]
        
        # Metrics
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_val = accuracy_score(y_val, y_pred_val)
        auc_val = roc_auc_score(y_val, y_prob_val)
        
        logger.info(f"Training Accuracy: {acc_train:.1%}")
        logger.info(f"Validation Accuracy: {acc_val:.1%}")
        logger.info(f"Validation AUC: {auc_val:.3f}")
        
        # Feature importance
        logger.info("\nFeature Importance:")
        for fname, imp in sorted(zip(self.feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True):
            logger.info(f"  {fname}: {imp:.4f}")
        
        results = {
            'model': model,
            'train_acc': acc_train,
            'val_acc': acc_val,
            'val_auc': auc_val,
            'y_pred': y_pred_val,
            'y_prob': y_prob_val
        }
        
        return results
    
    def evaluate_model(self, model_name, y_true, y_pred, y_prob):
        """Detailed evaluation"""
        logger.info(f"\n{model_name} - Detailed Evaluation:")
        logger.info(f"  Accuracy:  {accuracy_score(y_true, y_pred):.1%}")
        logger.info(f"  Precision: {precision_score(y_true, y_pred):.1%}")
        logger.info(f"  Recall:    {recall_score(y_true, y_pred):.1%}")
        logger.info(f"  AUC-ROC:   {roc_auc_score(y_true, y_prob):.3f}")
        logger.info(f"  Log Loss:  {log_loss(y_true, y_prob):.4f}")
        
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"  Confusion Matrix:\n{cm}")


def main():
    """Main workflow"""
    logger.info("=" * 70)
    logger.info("2026 MLB GAME OUTCOME MODEL TRAINING")
    logger.info("=" * 70)
    
    # Initialize
    model_trainer = GameOutcomeModel()
    
    # Load data
    train_df, val_df = model_trainer.load_data()
    X_train, X_val, y_train, y_val = model_trainer.prepare_features(train_df, val_df)
    
    # Train models
    results = {}
    
    results['LogisticRegression'] = model_trainer.train_logistic_regression(X_train, X_val, y_train, y_val)
    results['RandomForest'] = model_trainer.train_random_forest(X_train, X_val, y_train, y_val)
    results['XGBoost'] = model_trainer.train_xgboost(X_train, X_val, y_train, y_val)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)
    
    comparison = []
    for name, result in results.items():
        comparison.append({
            'Model': name,
            'Train Accuracy': f"{result['train_acc']:.1%}",
            'Val Accuracy': f"{result['val_acc']:.1%}",
            'Val AUC': f"{result['val_auc']:.3f}"
        })
        logger.info(f"{name}:")
        logger.info(f"  Train Acc: {result['train_acc']:.1%}")
        logger.info(f"  Val Acc:   {result['val_acc']:.1%}")
        logger.info(f"  AUC:       {result['val_auc']:.3f}")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['val_acc'])
    best_result = results[best_model_name]
    
    logger.info("\n" + "=" * 70)
    logger.info(f"BEST MODEL: {best_model_name}")
    logger.info("=" * 70)
    logger.info(f"Validation Accuracy: {best_result['val_acc']:.1%}")
    
    baseline = y_val.mean()
    improvement = best_result['val_acc'] - baseline
    
    logger.info(f"Baseline (home win %): {baseline:.1%}")
    logger.info(f"Improvement over baseline: {improvement:+.1%}")
    
    if best_result['val_acc'] > 0.54:
        logger.info(f"✅ SUCCESS! Beat 54% target by {(best_result['val_acc'] - 0.54)*100:.1f}%")
    else:
        logger.info(f"⚠️  Below 54% target, need {(0.54 - best_result['val_acc'])*100:.1f}% more accuracy")
    
    # Save best model
    logger.info("\n" + "=" * 70)
    logger.info("SAVING MODEL")
    logger.info("=" * 70)
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / f"game_outcome_{best_model_name}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_result['model'],
            'scaler': model_trainer.scaler,
            'features': model_trainer.feature_names,
            'model_name': best_model_name,
            'val_accuracy': best_result['val_acc'],
            'val_auc': best_result['val_auc'],
            'timestamp': datetime.now().isoformat()
        }, f)
    
    logger.info(f"✅ Model saved: {model_path}")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'actual': y_val,
        'predicted': best_result['y_pred'],
        'probability': best_result['y_prob']
    })
    
    pred_path = model_dir / f"predictions_{best_model_name}.parquet"
    pred_df.to_parquet(pred_path, index=False)
    logger.info(f"✅ Predictions saved: {pred_path}")
    
    # Save results summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_model': best_model_name,
        'validation_accuracy': float(best_result['val_acc']),
        'validation_auc': float(best_result['val_auc']),
        'baseline': float(baseline),
        'improvement': float(improvement),
        'beat_target': best_result['val_acc'] > 0.54,
        'all_results': {
            name: {
                'train_acc': float(result['train_acc']),
                'val_acc': float(result['val_acc']),
                'val_auc': float(result['val_auc'])
            }
            for name, result in results.items()
        }
    }
    
    summary_path = model_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Summary saved: {summary_path}")
    
    return results, best_model_name


if __name__ == "__main__":
    results, best_model = main()
