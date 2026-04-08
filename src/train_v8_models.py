#!/usr/bin/env python3
"""
V8 Model Training — Accuracy-First Iterative Experiments

Goal: Break the 54.6% V3 accuracy ceiling and reach 60%+ accuracy.

Architecture Evolution:
  V8.1 — Baseline check: V3 features re-run for ground truth
  V8.2 — Elo + Pythagorean features added (most impactful new signals)
  V8.3 — Full V8 feature set (run diff, streaks, H2H, context)
  V8.4 — Feature selection (SHAP-based pruning of low-signal features)
  V8.5 — CatBoost + better hyperparameters via Optuna
  V8.6 — Stacked ensemble (LR + XGB + LightGBM + CatBoost → meta LR)
  V8.7 — Walk-forward cross-validation to verify no overfitting
  V8.8 — Final best model

Usage:
    cd src/ && python train_v8_models.py
    python train_v8_models.py --iterations 1,2,3   # specific iterations
    python train_v8_models.py --quick              # skip Optuna tuning (faster)
"""

import argparse
import json
import logging
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/v8_experiments.log", mode="w"),
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "training"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

TARGET = "home_won"
EXCLUDE_COLS = {
    TARGET, "game_pk", "game_date", "year", "season",
    "home_team_id", "away_team_id", "home_team", "away_team",
    "home_team_name", "away_team_name",
}

# ---------------------------------------------------------------------------
# Feature sets for each iteration
# ---------------------------------------------------------------------------

V3_FEATURES = [
    "home_ema_form", "away_ema_form", "pitcher_quality_diff",
    "home_form_squared", "away_form_squared", "rest_balance",
    "home_momentum", "away_momentum", "fatigue_index", "park_advantage",
    "trend_alignment", "season_phase_home_effect", "win_pct_diff",
    "home_composite_strength", "away_composite_strength",
    "is_home", "day_of_week", "month",
    "home_win_pct_10d", "away_win_pct_10d",
    "home_pitcher_quality", "away_pitcher_quality",
    "home_team_rest_days", "away_team_rest_days",
    "travel_distance_km", "is_back_to_back",
    "home_win_pct_30d", "away_win_pct_30d",
    "home_trend_direction", "away_trend_direction",
    "home_park_run_factor", "away_park_run_factor",
    "month_3", "month_4", "month_5", "month_6", "month_7",
    "month_8", "month_9", "month_10", "month_11",
    "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6", "dow_7",
    "form_interaction", "form_difference", "month_home_effect",
]

V8_ELO_PYTHAG = [
    # Elo ratings (most powerful new signal)
    "home_elo", "away_elo",
    "elo_differential",
    "elo_home_win_prob",
    "elo_win_prob_differential",
    # Pythagorean win percentage
    "home_pythag_season", "away_pythag_season",
    "home_pythag_last30", "away_pythag_last30",
    "pythag_differential",
    # Luck factors (regression predictor)
    "home_luck_factor", "away_luck_factor",
    "luck_differential",
]

V8_RUN_DIFF = [
    # Rolling run differentials
    "home_run_diff_10g", "away_run_diff_10g",
    "home_run_diff_30g", "away_run_diff_30g",
    "run_diff_differential",
    # Scoring rates
    "home_runs_scored_10g", "away_runs_scored_10g",
    "home_runs_allowed_10g", "away_runs_allowed_10g",
    "home_runs_scored_30g", "away_runs_scored_30g",
    "home_runs_allowed_30g", "away_runs_allowed_30g",
    # Real ERA proxies (replaces fake pitcher_quality=0.5)
    "home_era_proxy_10g", "away_era_proxy_10g",
    "home_era_proxy_30g", "away_era_proxy_30g",
    "era_proxy_differential",
    # Season stats
    "home_win_pct_season", "away_win_pct_season",
    "home_scoring_momentum", "away_scoring_momentum",
    "home_consistency_score", "away_consistency_score",
]

V8_STREAK = [
    "home_current_streak", "away_current_streak",
    "home_streak_magnitude", "away_streak_magnitude",
    "home_streak_direction", "away_streak_direction",
    "home_win_pct_7g", "away_win_pct_7g",
    "home_win_pct_14g", "away_win_pct_14g",
    "home_on_winning_streak", "away_on_winning_streak",
    "home_on_losing_streak", "away_on_losing_streak",
    "streak_differential",
]

V8_H2H_CONTEXT = [
    "h2h_win_pct_season", "h2h_games_season",
    "h2h_win_pct_3yr", "h2h_games_3yr",
    "h2h_advantage_season", "h2h_advantage_3yr",
    "season_pct_complete", "season_stage",
    "is_divisional", "is_interleague",
    "season_stage_late", "season_stage_early",
]

ALL_V8_FEATURES = V3_FEATURES + V8_ELO_PYTHAG + V8_RUN_DIFF + V8_STREAK + V8_H2H_CONTEXT


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load V8 train and val datasets."""
    train_path = DATA_DIR / "train_v8_2015_2024.parquet"
    val_path = DATA_DIR / "val_v8_2025.parquet"

    if not train_path.exists():
        logger.info("V8 features not found — running feature builder...")
        sys.path.insert(0, str(Path(__file__).parent))
        from build_v8_features import V8FeatureBuilder
        builder = V8FeatureBuilder()
        train_df, val_df = builder.build()
    else:
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)

    logger.info(f"Loaded TRAIN: {train_df.shape}, VAL: {val_df.shape}")
    return train_df, val_df


def prepare_xy(
    df: pd.DataFrame,
    feature_list: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract X (features) and y (target) from dataframe."""
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        logger.warning(f"Missing features ({len(missing)}): {missing[:5]}...")

    X = df[available].copy()
    # Convert bool columns to int
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)
    # Fill any remaining NaN
    X = X.fillna(X.median(numeric_only=True))

    y = df[TARGET].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_lr(X_train, y_train) -> Pipeline:
    """Logistic Regression with standard scaling."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    return model


def build_xgb(X_train, y_train, params: Optional[dict] = None) -> xgb.XGBClassifier:
    """XGBoost classifier."""
    default_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "eval_metric": "logloss",
        "verbosity": 0,
        "use_label_encoder": False,
    }
    if params:
        default_params.update(params)
    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def build_lgb(X_train, y_train, params: Optional[dict] = None) -> lgb.LGBMClassifier:
    """LightGBM classifier."""
    default_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)
    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def build_catboost(X_train, y_train, params: Optional[dict] = None) -> CatBoostClassifier:
    """CatBoost classifier."""
    default_params = {
        "iterations": 300,
        "depth": 4,
        "learning_rate": 0.05,
        "random_seed": 42,
        "verbose": False,
        "eval_metric": "Logloss",
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "l2_leaf_reg": 3.0,
    }
    if params:
        default_params.update(params)
    model = CatBoostClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def build_stacked_ensemble(
    base_models: List,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[object, pd.DataFrame]:
    """
    Build stacked ensemble using base model predictions as meta-features.
    Uses held-out base model predictions (from train set using OOF).
    """
    # Generate OOF predictions for meta-training
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.zeros((len(X_train), len(base_models)))

    for fold, (fold_train_idx, fold_val_idx) in enumerate(tscv.split(X_train)):
        X_fold_train = X_train.iloc[fold_train_idx]
        y_fold_train = y_train.iloc[fold_train_idx]
        X_fold_val = X_train.iloc[fold_val_idx]

        for i, model_type in enumerate(base_models):
            if model_type == "lr":
                m = build_lr(X_fold_train, y_fold_train)
            elif model_type == "xgb":
                m = build_xgb(X_fold_train, y_fold_train)
            elif model_type == "lgb":
                m = build_lgb(X_fold_train, y_fold_train)
            elif model_type == "catboost":
                m = build_catboost(X_fold_train, y_fold_train)
            else:
                continue

            oof_preds[fold_val_idx, i] = m.predict_proba(X_fold_val)[:, 1]

    # Train final base models on full training set
    trained_base = []
    val_preds = np.zeros((len(X_val), len(base_models)))

    for i, model_type in enumerate(base_models):
        if model_type == "lr":
            m = build_lr(X_train, y_train)
        elif model_type == "xgb":
            m = build_xgb(X_train, y_train)
        elif model_type == "lgb":
            m = build_lgb(X_train, y_train)
        elif model_type == "catboost":
            m = build_catboost(X_train, y_train)
        else:
            continue
        trained_base.append(m)
        val_preds[:, i] = m.predict_proba(X_val)[:, 1]

    # Train meta LR on OOF predictions
    meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_lr.fit(oof_preds, y_train)

    # Create ensemble structure
    ensemble = {
        "base_models": trained_base,
        "base_model_types": base_models,
        "meta_model": meta_lr,
        "oof_accuracy": accuracy_score(y_train, (oof_preds.mean(axis=1) > 0.5).astype(int)),
    }

    oof_meta_probs = meta_lr.predict_proba(oof_preds)[:, 1]
    oof_acc = accuracy_score(y_train, (oof_meta_probs > 0.5).astype(int))
    ensemble["oof_meta_accuracy"] = oof_acc

    val_meta_probs = meta_lr.predict_proba(val_preds)[:, 1]
    val_df_out = pd.DataFrame(val_preds, columns=[f"pred_{t}" for t in base_models])
    val_df_out["meta_prob"] = val_meta_probs

    return ensemble, val_df_out


def predict_ensemble(ensemble: dict, X: pd.DataFrame) -> np.ndarray:
    """Get probability predictions from stacked ensemble."""
    base_preds = np.zeros((len(X), len(ensemble["base_models"])))
    for i, model in enumerate(ensemble["base_models"]):
        base_preds[:, i] = model.predict_proba(X)[:, 1]
    return ensemble["meta_model"].predict_proba(base_preds)[:, 1]


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------

def tune_xgb(X_train, y_train, n_trials: int = 50) -> dict:
    """Tune XGBoost hyperparameters with Optuna."""
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "random_state": 42, "verbosity": 0, "eval_metric": "logloss",
        }
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            m = xgb.XGBClassifier(**params)
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = m.predict_proba(X_train.iloc[val_idx])[:, 1]
            scores.append(accuracy_score(y_train.iloc[val_idx], (pred > 0.5).astype(int)))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"XGB best CV accuracy: {study.best_value:.4f}")
    return study.best_params


def tune_lgb(X_train, y_train, n_trials: int = 50) -> dict:
    """Tune LightGBM hyperparameters with Optuna."""
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "random_state": 42, "verbose": -1,
        }
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = m.predict_proba(X_train.iloc[val_idx])[:, 1]
            scores.append(accuracy_score(y_train.iloc[val_idx], (pred > 0.5).astype(int)))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"LGB best CV accuracy: {study.best_value:.4f}")
    return study.best_params


def tune_catboost(X_train, y_train, n_trials: int = 40) -> dict:
    """Tune CatBoost hyperparameters with Optuna."""
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 600),
            "depth": trial.suggest_int("depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "bootstrap_type": "Bernoulli",
            "random_seed": 42, "verbose": False,
        }
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            m = CatBoostClassifier(**params)
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = m.predict_proba(X_train.iloc[val_idx])[:, 1]
            scores.append(accuracy_score(y_train.iloc[val_idx], (pred > 0.5).astype(int)))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"CatBoost best CV accuracy: {study.best_value:.4f}")
    return study.best_params


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model, X_val, y_val,
    model_name: str,
    proba_func=None,
) -> dict:
    """Full evaluation metrics for a model."""
    if proba_func:
        y_prob = proba_func(X_val)
    elif hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        y_prob = model.predict(X_val).astype(float)

    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    brier = brier_score_loss(y_val, y_prob)
    ll = log_loss(y_val, y_prob)

    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # High-confidence accuracy: games where model is >60% confident
    confident_mask = np.abs(y_prob - 0.5) > 0.10
    if confident_mask.sum() > 0:
        conf_acc = accuracy_score(
            y_val[confident_mask],
            y_pred[confident_mask]
        )
        conf_pct = confident_mask.mean()
    else:
        conf_acc = acc
        conf_pct = 0.0

    result = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
        "recall": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
        "confident_pct": round(conf_pct, 4),
        "confident_accuracy": round(conf_acc, 4),
        "n_val": len(y_val),
        "home_win_rate_val": round(y_val.mean(), 4),
    }

    logger.info(
        f"[{model_name}] Acc={acc:.4f} | AUC={auc:.4f} | Brier={brier:.4f} | "
        f"Conf={conf_pct:.1%}@{conf_acc:.4f}"
    )
    return result


def get_feature_importance(model, feature_names: List[str], top_n: int = 20) -> List[Tuple]:
    """Extract top feature importances from tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        # Pipeline
        clf = list(model.named_steps.values())[-1]
        if hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            return []
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return []

    feature_imp = list(zip(feature_names, importances))
    feature_imp.sort(key=lambda x: x[1], reverse=True)
    return feature_imp[:top_n]


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def walk_forward_cv(
    train_df: pd.DataFrame,
    feature_list: List[str],
    n_folds: int = 5,
) -> dict:
    """
    Walk-forward temporal cross-validation.

    Uses years as splits: train on earlier years, validate on next year.
    This mirrors production deployment (always predicting future games).
    """
    years = sorted(train_df["year"].unique())
    results = []

    # Use last n_folds years for validation
    val_years = years[-(n_folds):]

    for val_year in val_years:
        train_years = [y for y in years if y < val_year]
        if len(train_years) < 3:  # Need at least 3 years to train
            continue

        fold_train = train_df[train_df["year"].isin(train_years)]
        fold_val = train_df[train_df["year"] == val_year]

        X_train, y_train = prepare_xy(fold_train, feature_list)
        X_val, y_val = prepare_xy(fold_val, feature_list)

        # Quick XGBoost model for each fold
        model = build_xgb(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, (y_prob > 0.5).astype(int))

        results.append({"val_year": val_year, "accuracy": acc, "n_games": len(fold_val)})
        logger.info(f"  Walk-forward fold {val_year}: {acc:.4f} ({len(fold_val)} games)")

    avg_acc = np.mean([r["accuracy"] for r in results])
    std_acc = np.std([r["accuracy"] for r in results])
    logger.info(f"  Walk-forward CV: {avg_acc:.4f} ± {std_acc:.4f}")

    return {"folds": results, "mean_accuracy": avg_acc, "std_accuracy": std_acc}


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

class V8ExperimentRunner:
    """Runs all V8 iterations and tracks results."""

    def __init__(self, quick: bool = False):
        self.quick = quick
        self.results: List[dict] = []
        self.all_results_detail: Dict[str, dict] = {}
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.X_train_full: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_val_full: Optional[pd.DataFrame] = None
        self.y_val: Optional[pd.Series] = None

    def _load(self):
        self.train_df, self.val_df = load_data()
        self.X_train_full, self.y_train = prepare_xy(self.train_df, ALL_V8_FEATURES)
        self.X_val_full, self.y_val = prepare_xy(self.val_df, ALL_V8_FEATURES)
        logger.info(f"Full feature matrix: train={self.X_train_full.shape}, val={self.X_val_full.shape}")

    def _record(self, metrics: dict):
        self.results.append(metrics)
        logger.info(f"\n{'='*60}")
        logger.info(f"RESULT: {metrics['model']} → ACCURACY={metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # ITER 1: V3 Baseline (ground truth)
    # ------------------------------------------------------------------
    def iter_1_v3_baseline(self):
        """Re-run V3 models for ground truth comparison."""
        logger.info("\n" + "="*70)
        logger.info("ITERATION 1: V3 Baseline Ground Truth")
        logger.info("="*70)
        logger.info("Features: 50 V3 core features (same as original V3 training)")
        logger.info("Purpose: Establish exact baseline on same data split")

        X_train, y_train = prepare_xy(self.train_df, V3_FEATURES)
        X_val, y_val = prepare_xy(self.val_df, V3_FEATURES)

        models = {
            "V8_iter1_LR": build_lr(X_train, y_train),
            "V8_iter1_XGBoost": build_xgb(X_train, y_train),
            "V8_iter1_LightGBM": build_lgb(X_train, y_train),
        }

        for name, model in models.items():
            metrics = evaluate(model, X_val, y_val, name)
            metrics["iteration"] = 1
            metrics["feature_group"] = "V3_baseline"
            metrics["n_features"] = X_train.shape[1]
            self._record(metrics)

        return models

    # ------------------------------------------------------------------
    # ITER 2: V3 + Elo + Pythagorean (core insight)
    # ------------------------------------------------------------------
    def iter_2_elo_pythag(self):
        """Add Elo ratings and Pythagorean win% — the biggest expected gains."""
        logger.info("\n" + "="*70)
        logger.info("ITERATION 2: V3 + Elo + Pythagorean Win%")
        logger.info("="*70)
        logger.info("NEW: Elo ratings (probabilistic team strength) + Pythagorean win pct")
        logger.info("Hypothesis: Elo is the single most predictive signal in sports forecasting")

        feat_list = V3_FEATURES + V8_ELO_PYTHAG
        X_train, y_train = prepare_xy(self.train_df, feat_list)
        X_val, y_val = prepare_xy(self.val_df, feat_list)

        models = {
            "V8_iter2_LR": build_lr(X_train, y_train),
            "V8_iter2_XGBoost": build_xgb(X_train, y_train),
            "V8_iter2_LightGBM": build_lgb(X_train, y_train),
            "V8_iter2_CatBoost": build_catboost(X_train, y_train),
        }

        for name, model in models.items():
            metrics = evaluate(model, X_val, y_val, name)
            metrics["iteration"] = 2
            metrics["feature_group"] = "V3+Elo+Pythag"
            metrics["n_features"] = X_train.shape[1]
            self._record(metrics)

        # Feature importance from XGB
        fi = get_feature_importance(models["V8_iter2_XGBoost"], X_train.columns.tolist(), 15)
        logger.info("Top 15 features by XGB importance:")
        for rank, (feat, imp) in enumerate(fi, 1):
            logger.info(f"  {rank:2d}. {feat:<45s} {imp:.4f}")

        return models

    # ------------------------------------------------------------------
    # ITER 3: V3 + Elo + Pythag + Run Differential
    # ------------------------------------------------------------------
    def iter_3_run_diff(self):
        """Add run differential metrics — real pitching quality and scoring."""
        logger.info("\n" + "="*70)
        logger.info("ITERATION 3: V3 + Elo + Pythag + Run Differential")
        logger.info("="*70)
        logger.info("NEW: Run differential, real ERA proxy (replaces fake 0.5 pitcher_quality)")
        logger.info("Hypothesis: Real pitching data is far more predictive than placeholder values")

        feat_list = V3_FEATURES + V8_ELO_PYTHAG + V8_RUN_DIFF
        X_train, y_train = prepare_xy(self.train_df, feat_list)
        X_val, y_val = prepare_xy(self.val_df, feat_list)

        models = {
            "V8_iter3_XGBoost": build_xgb(X_train, y_train),
            "V8_iter3_LightGBM": build_lgb(X_train, y_train),
            "V8_iter3_CatBoost": build_catboost(X_train, y_train),
        }

        for name, model in models.items():
            metrics = evaluate(model, X_val, y_val, name)
            metrics["iteration"] = 3
            metrics["feature_group"] = "V3+Elo+Pythag+RunDiff"
            metrics["n_features"] = X_train.shape[1]
            self._record(metrics)

        return models

    # ------------------------------------------------------------------
    # ITER 4: Full V8 features
    # ------------------------------------------------------------------
    def iter_4_full_v8(self):
        """All V8 features including streaks, H2H, context."""
        logger.info("\n" + "="*70)
        logger.info("ITERATION 4: Full V8 Feature Set")
        logger.info("="*70)
        logger.info("NEW: Streak features, H2H records, game context (divisional, season phase)")

        X_train, y_train = prepare_xy(self.train_df, ALL_V8_FEATURES)
        X_val, y_val = prepare_xy(self.val_df, ALL_V8_FEATURES)

        models = {
            "V8_iter4_XGBoost": build_xgb(X_train, y_train),
            "V8_iter4_LightGBM": build_lgb(X_train, y_train),
            "V8_iter4_CatBoost": build_catboost(X_train, y_train),
        }

        for name, model in models.items():
            metrics = evaluate(model, X_val, y_val, name)
            metrics["iteration"] = 4
            metrics["feature_group"] = "All_V8"
            metrics["n_features"] = X_train.shape[1]
            self._record(metrics)

        fi = get_feature_importance(models["V8_iter4_XGBoost"], X_train.columns.tolist(), 20)
        logger.info("Top 20 features (full V8 XGBoost):")
        for rank, (feat, imp) in enumerate(fi, 1):
            logger.info(f"  {rank:2d}. {feat:<45s} {imp:.4f}")

        return models, X_train.columns.tolist()

    # ------------------------------------------------------------------
    # ITER 5: Feature selection (remove noise, keep signal)
    # ------------------------------------------------------------------
    def iter_5_feature_selection(
        self,
        best_model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_names: List[str],
    ):
        """SHAP/importance-based feature selection."""
        logger.info("\n" + "="*70)
        logger.info("ITERATION 5: Feature Selection (Remove Low-Signal Features)")
        logger.info("="*70)
        logger.info("Method: Remove features below 80th percentile of XGBoost importance")

        # Get full importance ranking
        fi_all = get_feature_importance(best_model, feature_names, len(feature_names))
        fi_sorted = sorted(fi_all, key=lambda x: x[1], reverse=True)

        # Keep top 50% of features (above median importance)
        n_keep = max(30, int(len(fi_sorted) * 0.55))
        selected_features = [f for f, _ in fi_sorted[:n_keep]]
        dropped_features = [f for f, _ in fi_sorted[n_keep:]]

        logger.info(f"Keeping {len(selected_features)} features (dropped {len(dropped_features)})")
        logger.info(f"Top dropped (low importance): {dropped_features[:10]}")

        X_train_sel, y_train_sel = prepare_xy(self.train_df, selected_features)
        X_val_sel, y_val_sel = prepare_xy(self.val_df, selected_features)

        models = {
            "V8_iter5_XGBoost_selected": build_xgb(X_train_sel, y_train_sel),
            "V8_iter5_LightGBM_selected": build_lgb(X_train_sel, y_train_sel),
            "V8_iter5_CatBoost_selected": build_catboost(X_train_sel, y_train_sel),
        }

        for name, model in models.items():
            metrics = evaluate(model, X_val_sel, y_val_sel, name)
            metrics["iteration"] = 5
            metrics["feature_group"] = "V8_selected"
            metrics["n_features"] = X_train_sel.shape[1]
            metrics["selected_features"] = selected_features
            self._record(metrics)

        return models, selected_features

    # ------------------------------------------------------------------
    # ITER 6: Optuna hyperparameter tuning
    # ------------------------------------------------------------------
    def iter_6_optuna_tuning(self, selected_features: List[str]):
        """Tune hyperparameters with Optuna for best features."""
        logger.info("\n" + "="*70)
        logger.info("ITERATION 6: Optuna Hyperparameter Tuning")
        logger.info("="*70)

        n_trials = 30 if self.quick else 60
        logger.info(f"Trials: {n_trials} per model")

        X_train, y_train = prepare_xy(self.train_df, selected_features)
        X_val, y_val = prepare_xy(self.val_df, selected_features)

        best_models = {}

        logger.info("Tuning XGBoost...")
        xgb_params = tune_xgb(X_train, y_train, n_trials)
        xgb_params.update({"random_state": 42, "verbosity": 0, "eval_metric": "logloss"})
        xgb_tuned = xgb.XGBClassifier(**xgb_params)
        xgb_tuned.fit(X_train, y_train)
        metrics_xgb = evaluate(xgb_tuned, X_val, y_val, "V8_iter6_XGB_tuned")
        metrics_xgb.update({"iteration": 6, "feature_group": "V8_selected_tuned",
                             "n_features": X_train.shape[1], "params": xgb_params})
        self._record(metrics_xgb)
        best_models["xgb"] = xgb_tuned

        logger.info("Tuning LightGBM...")
        lgb_params = tune_lgb(X_train, y_train, n_trials)
        lgb_params.update({"random_state": 42, "verbose": -1})
        lgb_tuned = lgb.LGBMClassifier(**lgb_params)
        lgb_tuned.fit(X_train, y_train)
        metrics_lgb = evaluate(lgb_tuned, X_val, y_val, "V8_iter6_LGB_tuned")
        metrics_lgb.update({"iteration": 6, "feature_group": "V8_selected_tuned",
                             "n_features": X_train.shape[1], "params": lgb_params})
        self._record(metrics_lgb)
        best_models["lgb"] = lgb_tuned

        logger.info("Tuning CatBoost...")
        cat_params = tune_catboost(X_train, y_train, n_trials=30 if self.quick else 50)
        cat_params.update({"random_seed": 42, "verbose": False, "bootstrap_type": "Bernoulli"})
        cat_tuned = CatBoostClassifier(**cat_params)
        cat_tuned.fit(X_train, y_train)
        metrics_cat = evaluate(cat_tuned, X_val, y_val, "V8_iter6_CatBoost_tuned")
        metrics_cat.update({"iteration": 6, "feature_group": "V8_selected_tuned",
                             "n_features": X_train.shape[1], "params": cat_params})
        self._record(metrics_cat)
        best_models["catboost"] = cat_tuned

        return best_models, xgb_params, lgb_params, cat_params

    # ------------------------------------------------------------------
    # ITER 7: Stacked ensemble
    # ------------------------------------------------------------------
    def iter_7_stacked_ensemble(
        self,
        selected_features: List[str],
        tuned_params: Optional[dict] = None,
    ):
        """Build stacked ensemble with tuned base models."""
        logger.info("\n" + "="*70)
        logger.info("ITERATION 7: Stacked Ensemble (LR + XGB + LGB + CatBoost → meta LR)")
        logger.info("="*70)

        X_train, y_train = prepare_xy(self.train_df, selected_features)
        X_val, y_val = prepare_xy(self.val_df, selected_features)

        base_model_types = ["lr", "xgb", "lgb", "catboost"]
        ensemble, val_preds_df = build_stacked_ensemble(
            base_model_types, X_train, y_train, X_val, y_val
        )

        y_prob = val_preds_df["meta_prob"].values
        y_pred = (y_prob > 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)

        # Also try simple average ensemble
        avg_prob = val_preds_df[[f"pred_{t}" for t in base_model_types]].mean(axis=1).values
        avg_acc = accuracy_score(y_val, (avg_prob > 0.5).astype(int))

        metrics_stacked = {
            "model": "V8_iter7_StackedEnsemble",
            "accuracy": round(acc, 4),
            "auc": round(auc, 4),
            "brier_score": round(brier_score_loss(y_val, y_prob), 4),
            "log_loss": round(log_loss(y_val, y_prob), 4),
            "iteration": 7,
            "feature_group": "V8_stacked",
            "n_features": X_train.shape[1],
            "oof_accuracy": ensemble["oof_meta_accuracy"],
            "avg_ensemble_accuracy": round(avg_acc, 4),
        }
        self._record(metrics_stacked)

        metrics_avg = {
            "model": "V8_iter7_AverageEnsemble",
            "accuracy": round(avg_acc, 4),
            "auc": round(roc_auc_score(y_val, avg_prob), 4),
            "brier_score": round(brier_score_loss(y_val, avg_prob), 4),
            "log_loss": round(log_loss(y_val, avg_prob), 4),
            "iteration": 7,
            "feature_group": "V8_avg_ensemble",
            "n_features": X_train.shape[1],
        }
        self._record(metrics_avg)

        return ensemble

    # ------------------------------------------------------------------
    # ITER 8: Walk-forward cross-validation
    # ------------------------------------------------------------------
    def iter_8_walk_forward(self, selected_features: List[str]):
        """Walk-forward CV to validate no time-leakage."""
        logger.info("\n" + "="*70)
        logger.info("ITERATION 8: Walk-Forward Cross-Validation")
        logger.info("="*70)
        logger.info("Validates temporal stability: no future data leakage")

        cv_results = walk_forward_cv(self.train_df, selected_features)

        metrics = {
            "model": "V8_iter8_WalkForwardCV",
            "accuracy": round(cv_results["mean_accuracy"], 4),
            "accuracy_std": round(cv_results["std_accuracy"], 4),
            "iteration": 8,
            "feature_group": "walk_forward",
            "n_features": len(selected_features),
            "cv_folds": cv_results["folds"],
        }
        self._record(metrics)
        return cv_results

    # ------------------------------------------------------------------
    # Final best model
    # ------------------------------------------------------------------
    def build_final_model(self, selected_features: List[str], best_params: dict):
        """Build final V8 model using best configuration."""
        logger.info("\n" + "="*70)
        logger.info("FINAL V8 MODEL: Training on full 2015-2024 data")
        logger.info("="*70)

        X_train, y_train = prepare_xy(self.train_df, selected_features)
        X_val, y_val = prepare_xy(self.val_df, selected_features)

        # XGBoost with best params
        xgb_best = build_xgb(X_train, y_train, best_params.get("xgb", {}) or {})
        lgb_best = build_lgb(X_train, y_train, best_params.get("lgb", {}) or {})
        cat_best = build_catboost(X_train, y_train, best_params.get("catboost", {}) or {})

        # Simple average ensemble (often more robust than stacked for deployment)
        probs_xgb = xgb_best.predict_proba(X_val)[:, 1]
        probs_lgb = lgb_best.predict_proba(X_val)[:, 1]
        probs_cat = cat_best.predict_proba(X_val)[:, 1]

        # Weighted average favoring best individual model
        probs_ensemble = (probs_xgb * 0.35 + probs_lgb * 0.35 + probs_cat * 0.30)

        acc_final = accuracy_score(y_val, (probs_ensemble > 0.5).astype(int))
        auc_final = roc_auc_score(y_val, probs_ensemble)

        metrics_final = {
            "model": "V8_FINAL",
            "accuracy": round(acc_final, 4),
            "auc": round(auc_final, 4),
            "brier_score": round(brier_score_loss(y_val, probs_ensemble), 4),
            "log_loss": round(log_loss(y_val, probs_ensemble), 4),
            "iteration": 99,
            "feature_group": "V8_final_ensemble",
            "n_features": X_train.shape[1],
            "n_train_games": len(X_train),
            "n_val_games": len(X_val),
            "selected_features": selected_features,
        }
        self._record(metrics_final)

        # Save final model bundle
        model_bundle = {
            "xgb": xgb_best,
            "lgb": lgb_best,
            "catboost": cat_best,
            "weights": [0.35, 0.35, 0.30],
            "feature_names": selected_features,
            "version": "v8",
            "trained_at": datetime.utcnow().isoformat(),
            "val_accuracy": acc_final,
            "val_auc": auc_final,
            "fill_values": {col: float(X_train[col].median()) for col in selected_features},
        }

        model_path = MODEL_DIR / "game_outcome_2026_v8.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_bundle, f)
        logger.info(f"Saved final model: {model_path}")

        return model_bundle

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    def print_summary(self):
        """Print full results summary table."""
        logger.info("\n" + "="*70)
        logger.info("V8 EXPERIMENT RESULTS SUMMARY")
        logger.info("="*70)

        # Sort by accuracy
        sorted_results = sorted(self.results, key=lambda x: x["accuracy"], reverse=True)

        print("\n" + "="*90)
        print(f"{'Model':<45s} {'Acc':>8s} {'AUC':>8s} {'Features':>10s} {'Iter':>6s}")
        print("-"*90)

        # Historical baselines
        print(f"{'V1 (LogReg, 5 features)':<45s} {'54.0%':>8s} {'0.543':>8s} {'5':>10s} {'hist':>6s}")
        print(f"{'V3 (XGBoost, 57 features) — PREV BEST':<45s} {'54.6%':>8s} {'0.546':>8s} {'57':>10s} {'hist':>6s}")
        print("-"*90)

        for r in sorted_results:
            if r.get("iteration") in [1, 2, 3, 4, 5, 6, 7, 8, 99]:
                acc_pct = f"{r['accuracy']*100:.2f}%"
                auc = f"{r.get('auc', 0):.4f}"
                n_feat = str(r.get("n_features", "?"))
                it = str(r.get("iteration", "?"))
                print(f"{r['model']:<45s} {acc_pct:>8s} {auc:>8s} {n_feat:>10s} {it:>6s}")

        print("="*90)

        # Find best
        best = max(self.results, key=lambda x: x["accuracy"])
        print(f"\nBEST MODEL: {best['model']} — {best['accuracy']*100:.2f}% accuracy")
        print(f"Previous record: V3 XGBoost — 54.6%")
        improvement = (best['accuracy'] - 0.546) * 100
        print(f"Improvement: {improvement:+.2f}%")

        if best["accuracy"] >= 0.60:
            print("TARGET ACHIEVED: 60%+ accuracy ✓")
        elif best["accuracy"] >= 0.57:
            print("TARGET APPROACHING: 57%+ accuracy — strong progress")
        elif best["accuracy"] >= 0.55:
            print("IMPROVEMENT: 55%+ accuracy — meaningful gain over V3")
        else:
            print("NOTE: Limited improvement — examine feature distributions")

    def save_results_json(self):
        """Save all experiment results to JSON."""
        results_path = LOG_DIR / "v8_experiment_results.json"
        output = {
            "experiment": "V8 MLB Game Prediction",
            "date": datetime.utcnow().isoformat(),
            "baseline_v3": {"accuracy": 0.546, "auc": 0.546, "n_features": 57},
            "results": self.results,
        }
        # Convert numpy types to Python native
        def convert(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(results_path, "w") as f:
            json.dump(output, f, default=convert, indent=2)
        logger.info(f"Results saved: {results_path}")

    # ------------------------------------------------------------------
    # Run all iterations
    # ------------------------------------------------------------------
    def run(self, iterations: Optional[List[int]] = None):
        """Run all experiment iterations."""
        start = time.time()
        logger.info(f"\n{'#'*70}")
        logger.info("V8 MLB GAME PREDICTION EXPERIMENT")
        logger.info(f"Date: {datetime.utcnow().isoformat()}")
        logger.info(f"Goal: Break 54.6% accuracy, reach 60%+")
        logger.info(f"Mode: {'QUICK' if self.quick else 'FULL'}")
        logger.info(f"{'#'*70}\n")

        self._load()

        run_all = iterations is None

        # Iteration 1: Baseline
        if run_all or 1 in iterations:
            self.iter_1_v3_baseline()

        # Iteration 2: Elo + Pythagorean
        if run_all or 2 in iterations:
            self.iter_2_elo_pythag()

        # Iteration 3: + Run differential
        if run_all or 3 in iterations:
            self.iter_3_run_diff()

        # Iteration 4: Full V8
        if run_all or 4 in iterations:
            models_iter4, feature_names_iter4 = self.iter_4_full_v8()
            best_model_iter4 = models_iter4["V8_iter4_XGBoost"]
        else:
            # Train a default model if needed
            X_full, y_full = prepare_xy(self.train_df, ALL_V8_FEATURES)
            best_model_iter4 = build_xgb(X_full, y_full)
            feature_names_iter4 = X_full.columns.tolist()

        # Iteration 5: Feature selection
        if run_all or 5 in iterations:
            X_full, y_full = prepare_xy(self.train_df, ALL_V8_FEATURES)
            _, selected_features = self.iter_5_feature_selection(
                best_model_iter4,
                X_full, y_full,
                feature_names_iter4,
            )
        else:
            selected_features = ALL_V8_FEATURES

        # Iteration 6: Hyperparameter tuning
        if run_all or 6 in iterations:
            _, xgb_params, lgb_params, cat_params = self.iter_6_optuna_tuning(selected_features)
            best_params = {"xgb": xgb_params, "lgb": lgb_params, "catboost": cat_params}
        else:
            best_params = {}

        # Iteration 7: Stacked ensemble
        if run_all or 7 in iterations:
            self.iter_7_stacked_ensemble(selected_features, best_params)

        # Iteration 8: Walk-forward CV
        if run_all or 8 in iterations:
            self.iter_8_walk_forward(selected_features)

        # Final model
        final_model = self.build_final_model(selected_features, best_params)

        elapsed = time.time() - start
        logger.info(f"\nTotal experiment time: {elapsed/60:.1f} minutes")

        self.print_summary()
        self.save_results_json()

        return self.results, final_model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V8 Model Training Experiments")
    parser.add_argument(
        "--iterations", type=str, default=None,
        help="Comma-separated iteration numbers to run (default: all)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer Optuna trials (faster, less optimal)"
    )
    args = parser.parse_args()

    iterations = None
    if args.iterations:
        iterations = [int(x) for x in args.iterations.split(",")]

    runner = V8ExperimentRunner(quick=args.quick)
    results, final_model = runner.run(iterations=iterations)
    return results


if __name__ == "__main__":
    main()
