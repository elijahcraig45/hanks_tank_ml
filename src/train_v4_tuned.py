#!/usr/bin/env python3
"""
V4 Tuning - Time-based CV hyperparameter search for XGBoost and LightGBM.
Trains on 2015-2024, validates on 2025 holdout.
"""

import logging
from pathlib import Path
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def time_folds(df):
    if "year" not in df.columns:
        return None
    years = sorted(df["year"].dropna().unique())
    if len(years) < 5:
        return None
    start_index = 3
    fold_years = years[start_index:]
    folds = []
    for year in fold_years:
        train_idx = df.index[df["year"] < year].to_numpy()
        val_idx = df.index[df["year"] == year].to_numpy()
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
    return folds


def prepare_data():
    train_df = pd.read_parquet("data/training/train_v3_2015_2024.parquet")
    val_df = pd.read_parquet("data/training/val_v3_2025.parquet")

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
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    fill_values = {}
    for col in feature_cols:
        fill_val = train_df[col].median()
        fill_values[col] = fill_val
        train_df[col] = train_df[col].fillna(fill_val)
        val_df[col] = val_df[col].fillna(fill_val)

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df["home_won"].values.astype(int)
    X_val = val_df[feature_cols].values.astype(float)
    y_val = val_df["home_won"].values.astype(int)

    return train_df, X_train, y_train, X_val, y_val, feature_cols, fill_values


def cv_score_auc(model_builder, X, y, train_df):
    folds = time_folds(train_df)
    if folds is None:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = [(tr, va) for tr, va in skf.split(X, y)]

    aucs = []
    for train_idx, val_idx in folds:
        model = model_builder()
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], probs))
    return float(np.mean(aucs))


def cv_score_accuracy(model_builder, X, y, train_df):
    folds = time_folds(train_df)
    if folds is None:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = [(tr, va) for tr, va in skf.split(X, y)]

    accs = []
    for train_idx, val_idx in folds:
        model = model_builder()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], preds))
    return float(np.mean(accs))


def tune_xgb(train_df, X, y, n_trials=25, metric="auc"):
    logger.info("Tuning XGBoost...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "eval_metric": "auc",
            "random_state": 42,
        }

        def builder():
            return xgb.XGBClassifier(**params)

        if metric == "accuracy":
            return cv_score_accuracy(builder, X, y, train_df)
        return cv_score_auc(builder, X, y, train_df)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best XGB AUC: {study.best_value:.4f}")
    logger.info(f"Best XGB params: {study.best_params}")
    return study.best_params


def tune_lgbm(train_df, X, y, n_trials=25, metric="auc"):
    logger.info("Tuning LightGBM...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "random_state": 42,
        }

        def builder():
            return lgb.LGBMClassifier(**params)

        if metric == "accuracy":
            return cv_score_accuracy(builder, X, y, train_df)
        return cv_score_auc(builder, X, y, train_df)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best LGBM AUC: {study.best_value:.4f}")
    logger.info(f"Best LGBM params: {study.best_params}")
    return study.best_params


def evaluate_on_holdout(model, X_val, y_val, name):
    probs = model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, (probs >= 0.5).astype(int))
    auc = roc_auc_score(y_val, probs)
    logger.info(f"{name} | Holdout Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    return float(acc), float(auc)


def main():
    logger.info("=" * 70)
    logger.info("V4 TUNED MODELS")
    logger.info("=" * 70)

    train_df, X_train, y_train, X_val, y_val, feature_cols, fill_values = prepare_data()

    best_xgb_params = tune_xgb(train_df, X_train, y_train, n_trials=25, metric="auc")
    best_lgbm_params = tune_lgbm(train_df, X_train, y_train, n_trials=25, metric="auc")

    xgb_model = xgb.XGBClassifier(**{**best_xgb_params, "eval_metric": "auc", "random_state": 42})
    lgbm_model = lgb.LGBMClassifier(**{**best_lgbm_params, "random_state": 42})

    xgb_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)

    xgb_acc, xgb_auc = evaluate_on_holdout(xgb_model, X_val, y_val, "XGB (tuned)")
    lgbm_acc, lgbm_auc = evaluate_on_holdout(lgbm_model, X_val, y_val, "LGBM (tuned)")

    best_model = ("xgb", xgb_model, xgb_acc, xgb_auc) if xgb_auc >= lgbm_auc else ("lgbm", lgbm_model, lgbm_acc, lgbm_auc)

    artifact = {
        "model_type": best_model[0],
        "model": best_model[1],
        "metrics": {"accuracy": best_model[2], "auc": best_model[3]},
        "feature_cols": feature_cols,
        "fill_values": fill_values,
        "xgb_params": best_xgb_params,
        "lgbm_params": best_lgbm_params,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Accuracy-optimized search (secondary)
    logger.info("=" * 70)
    logger.info("ACCURACY-OPTIMIZED SEARCH")
    logger.info("=" * 70)

    best_xgb_params_acc = tune_xgb(train_df, X_train, y_train, n_trials=20, metric="accuracy")
    best_lgbm_params_acc = tune_lgbm(train_df, X_train, y_train, n_trials=20, metric="accuracy")

    xgb_model_acc = xgb.XGBClassifier(**{**best_xgb_params_acc, "eval_metric": "auc", "random_state": 42})
    lgbm_model_acc = lgb.LGBMClassifier(**{**best_lgbm_params_acc, "random_state": 42})

    xgb_model_acc.fit(X_train, y_train)
    lgbm_model_acc.fit(X_train, y_train)

    xgb_acc_acc, xgb_auc_acc = evaluate_on_holdout(xgb_model_acc, X_val, y_val, "XGB (acc-tuned)")
    lgbm_acc_acc, lgbm_auc_acc = evaluate_on_holdout(lgbm_model_acc, X_val, y_val, "LGBM (acc-tuned)")

    artifact["accuracy_tuned"] = {
        "xgb": {
            "params": best_xgb_params_acc,
            "metrics": {"accuracy": xgb_acc_acc, "auc": xgb_auc_acc},
        },
        "lgbm": {
            "params": best_lgbm_params_acc,
            "metrics": {"accuracy": lgbm_acc_acc, "auc": lgbm_auc_acc},
        },
    }

    output_path = Path("models/game_outcome_v4_tuned.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)

    logger.info(f"✅ Saved tuned model to {output_path}")


if __name__ == "__main__":
    main()
