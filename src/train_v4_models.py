#!/usr/bin/env python3
"""
V4 Model Training - Time-based stacking ensemble

Goal: Improve on V3 (54.6%) by stacking calibrated base models using
expanding-year cross-validation on 2015-2024, with 2025 as the holdout.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class V4StackedTrainer:
    """Train stacked ensemble with time-based CV."""

    def __init__(self):
        self.feature_cols = []
        self.fill_values = {}

    def load_data(self):
        logger.info("Loading v3 training data...")
        train_df = pd.read_parquet("data/training/train_v3_2015_2024.parquet")
        val_df = pd.read_parquet("data/training/val_v3_2025.parquet")
        logger.info(f"✅ Train: {len(train_df)} rows | Val: {len(val_df)} rows")
        return train_df, val_df

    def select_features(self, train_df, val_df):
        logger.info("Selecting numeric features...")
        exclude_cols = {
            "home_won",
            "game_pk",
            "game_date",
            "year",
            "home_team_id",
            "away_team_id",
            "home_team",
            "away_team",
            "home_team_name",
            "away_team_name",
            "season",
        }
        numeric_cols = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])]
        self.feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        logger.info(f"Selected {len(self.feature_cols)} numeric features")

        # Fill values based on train set
        for col in self.feature_cols:
            fill_val = train_df[col].median()
            self.fill_values[col] = fill_val
            train_df[col] = train_df[col].fillna(fill_val)
            val_df[col] = val_df[col].fillna(fill_val)

        X_train = train_df[self.feature_cols].values.astype(float)
        y_train = train_df["home_won"].values.astype(int)
        X_val = val_df[self.feature_cols].values.astype(float)
        y_val = val_df["home_won"].values.astype(int)

        return X_train, y_train, X_val, y_val

    def build_base_models(self):
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=42)),
        ])

        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_lambda=1.0,
            reg_alpha=0.0,
            eval_metric="logloss",
            random_state=42,
        )

        lgbm = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=40,
            random_state=42,
        )

        return {
            "lr": lr,
            "xgb": xgb_model,
            "lgbm": lgbm,
        }

    def time_folds(self, df):
        if "year" not in df.columns:
            logger.warning("No 'year' column found; falling back to StratifiedKFold")
            return None

        years = sorted(df["year"].dropna().unique())
        if len(years) < 5:
            logger.warning("Not enough years for time-based CV; using StratifiedKFold")
            return None

        start_index = 3  # require at least 3 years of training
        fold_years = years[start_index:]
        folds = []
        for year in fold_years:
            train_idx = df.index[df["year"] < year].to_numpy()
            val_idx = df.index[df["year"] == year].to_numpy()
            if len(train_idx) > 0 and len(val_idx) > 0:
                folds.append((train_idx, val_idx, year))
        return folds

    def get_oof_predictions(self, X, y, train_df, base_models):
        logger.info("Generating out-of-fold predictions for stacking...")
        oof_preds = {name: np.full(len(y), np.nan, dtype=float) for name in base_models}

        folds = self.time_folds(train_df)
        if folds is None:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            folds = [(train_idx, val_idx, None) for train_idx, val_idx in skf.split(X, y)]

        for fold_id, (train_idx, val_idx, fold_year) in enumerate(folds, start=1):
            if fold_year is not None:
                logger.info(f"Fold {fold_id}: train < {fold_year}, validate = {fold_year}")
            else:
                logger.info(f"Fold {fold_id}: stratified CV")

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            for name, model in base_models.items():
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_va)[:, 1]
                oof_preds[name][val_idx] = probs

                acc = accuracy_score(y_va, (probs >= 0.5).astype(int))
                auc = roc_auc_score(y_va, probs)
                logger.info(f"  {name}: acc={acc:.3f} auc={auc:.3f}")

        return oof_preds

    def train_meta_model(self, oof_preds, y):
        oof_matrix = np.column_stack([oof_preds[name] for name in sorted(oof_preds.keys())])
        valid_rows = ~np.isnan(oof_matrix).any(axis=1)

        X_meta = oof_matrix[valid_rows]
        y_meta = y[valid_rows]

        meta = LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)
        meta.fit(X_meta, y_meta)
        logger.info(f"Meta model trained on {len(X_meta)} rows")

        return meta, sorted(oof_preds.keys())

    def evaluate_single_model(self, model, X_val, y_val, name):
        probs = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, (probs >= 0.5).astype(int))
        auc = roc_auc_score(y_val, probs)
        logger.info(f"{name} | Val Accuracy: {acc:.4f} | AUC: {auc:.4f}")
        return acc, auc

    def run(self):
        logger.info("=" * 70)
        logger.info("V4 STACKED MODEL TRAINING")
        logger.info("=" * 70)

        train_df, val_df = self.load_data()
        X_train, y_train, X_val, y_val = self.select_features(train_df, val_df)

        base_models = self.build_base_models()

        # OOF predictions for stacking
        oof_preds = self.get_oof_predictions(X_train, y_train, train_df, base_models)
        meta_model, meta_order = self.train_meta_model(oof_preds, y_train)

        # Fit base models on full training data
        for model in base_models.values():
            model.fit(X_train, y_train)

        logger.info("=" * 70)
        logger.info("VALIDATION RESULTS (2025)")
        logger.info("=" * 70)

        base_metrics = {}
        for name, model in base_models.items():
            acc, auc = self.evaluate_single_model(model, X_val, y_val, name)
            base_metrics[name] = {"accuracy": float(acc), "auc": float(auc)}

        # Stacked evaluation
        val_matrix = np.column_stack([base_models[name].predict_proba(X_val)[:, 1] for name in meta_order])
        stacked_probs = meta_model.predict_proba(val_matrix)[:, 1]
        stacked_acc = accuracy_score(y_val, (stacked_probs >= 0.5).astype(int))
        stacked_auc = roc_auc_score(y_val, stacked_probs)

        logger.info(f"STACKED | Val Accuracy: {stacked_acc:.4f} | AUC: {stacked_auc:.4f}")

        model_artifact = {
            "base_models": base_models,
            "meta_model": meta_model,
            "meta_order": meta_order,
            "feature_cols": self.feature_cols,
            "fill_values": self.fill_values,
            "metrics": {
                "stacked": {"accuracy": float(stacked_acc), "auc": float(stacked_auc)},
                "base": base_metrics,
            },
            "timestamp": datetime.utcnow().isoformat(),
            "train_years": sorted(train_df.get("year", pd.Series(dtype=int)).unique().tolist()),
            "val_year": int(val_df["year"].unique()[0]) if "year" in val_df.columns else None,
        }

        output_path = Path("models/game_outcome_v4_stacked.pkl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(model_artifact, f)

        logger.info(f"✅ Saved V4 stacked model to {output_path}")


if __name__ == "__main__":
    trainer = V4StackedTrainer()
    trainer.run()
