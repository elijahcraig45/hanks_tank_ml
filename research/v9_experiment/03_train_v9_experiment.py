#!/usr/bin/env python3
"""
V9 Model Experiment — Systematic Search for Best MLB Game Predictor

Runs a rigorous multi-phase experiment to find the model/feature combination
that best predicts 2026 MLB games.

Phases:
  1. Feature ablation   — which feature groups add the most value?
  2. Model comparison   — CatBoost vs XGBoost vs LightGBM vs MLP vs FLAML
  3. Ensemble study     — how to combine models optimally
  4. Calibration        — isotonic vs Platt scaling
  5. Confidence analysis — accuracy vs coverage tradeoff curve
  6. 2026 live test     — evaluate on games played so far in 2026

Evaluation metric hierarchy:
  PRIMARY:   Walk-forward CV accuracy (true generalization estimate)
  SECONDARY: 2025 holdout accuracy (year-level split)
  TERTIARY:  High-confidence accuracy at ≥60% predicted probability

Usage:
    python 03_train_v9_experiment.py
    python 03_train_v9_experiment.py --quick         # skip FLAML + Optuna
    python 03_train_v9_experiment.py --phase 1,2,3   # run specific phases
    python 03_train_v9_experiment.py --no-mlp        # skip MLP (slow)
"""

import argparse
import json
import logging
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path(__file__).parent.parent.parent / "logs" / "v9_experiment.log",
            mode="w",
        ),
    ],
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "v9" / "features"
MODEL_DIR = REPO_ROOT / "models" / "v9"
LOG_DIR = REPO_ROOT / "logs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "home_won"
EXCLUDE_COLS = {
    TARGET, "game_pk", "game_date", "season", "home_team_id", "away_team_id",
    "home_score", "away_score", "home_team_name", "away_team_name",
    "home_team_fg", "away_team_fg", "_type", "fetch_team",
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature group definitions
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "elo": [
        "home_elo", "away_elo", "elo_differential", "elo_home_win_prob",
        "elo_win_prob_differential",
    ],
    "pythag": [
        "home_pythag_season", "away_pythag_season",
        "home_pythag_last30", "away_pythag_last30",
        "pythag_differential",
        "home_luck_factor", "away_luck_factor", "luck_differential",
    ],
    "rolling_form": [
        "home_win_pct_3g", "away_win_pct_3g",
        "home_win_pct_7g", "away_win_pct_7g",
        "home_win_pct_10g", "away_win_pct_10g",
        "home_win_pct_30g", "away_win_pct_30g",
        "home_win_pct_season", "away_win_pct_season", "win_pct_diff",
        "home_run_diff_3g", "away_run_diff_3g",
        "home_run_diff_7g", "away_run_diff_7g",
        "home_run_diff_10g", "away_run_diff_10g",
        "home_run_diff_30g", "away_run_diff_30g",
        "run_diff_differential",
        "home_runs_scored_10g", "away_runs_scored_10g",
        "home_runs_scored_30g", "away_runs_scored_30g",
        "home_runs_allowed_10g", "away_runs_allowed_10g",
        "home_runs_allowed_30g", "away_runs_allowed_30g",
        "home_scoring_momentum", "away_scoring_momentum",
        "home_era_proxy_10g", "away_era_proxy_10g",
        "home_era_proxy_30g", "away_era_proxy_30g",
        "era_proxy_differential",
    ],
    "streak": [
        "home_current_streak", "away_current_streak",
        "home_streak_direction", "away_streak_direction",
        "streak_differential",
    ],
    "h2h": [
        "home_h2h_win_pct_season", "home_h2h_games_season",
        "home_h2h_win_pct_3yr",
    ],
    "fg_pitching": [
        "home_fg_era", "away_fg_era",
        "home_fg_whip", "away_fg_whip",
        "home_fg_k9", "away_fg_k9",
        "home_fg_bb9", "away_fg_bb9",
        # Statcast percentile ranks (xERA, K%, BB%, whiff%, fb velocity)
        "home_fg_xfip", "away_fg_xfip",     # mapped from sc_sp_xera_pct
        "home_fg_k_pct", "away_fg_k_pct",
        "home_fg_bb_pct", "away_fg_bb_pct",
        "home_fg_whiff_pct", "away_fg_whiff_pct",
        "home_fg_fbv_pct", "away_fg_fbv_pct",
        "fg_era_differential", "fg_xfip_differential", "fg_whip_differential",
    ],
    "fg_batting": [
        "home_fg_ops", "away_fg_ops",
        "home_fg_obp", "away_fg_obp",
        "home_fg_slg", "away_fg_slg",
        # Statcast percentile ranks (xwOBA, exit velo, hard hit, barrel)
        "home_fg_woba", "away_fg_woba",     # mapped from sc_bat_xwoba_pct
        "home_fg_ev_pct", "away_fg_ev_pct",
        "home_fg_hh_pct", "away_fg_hh_pct",
        "home_fg_brl_pct", "away_fg_brl_pct",
        "fg_ops_differential", "fg_woba_differential", "fg_obp_differential",
    ],
    "park": [
        "home_park_factor", "home_park_factor_ratio",
    ],
    "calendar": [
        "day_of_week", "month", "is_weekend",
        "month_3", "month_4", "month_5", "month_6",
        "month_7", "month_8", "month_9", "month_10",
    ],
    "context": [
        "season_game_number", "season_pct_complete",
        "is_late_season", "is_early_season",
    ],
}

# Baseline V8-equivalent feature set (for comparison)
V8_EQUIV = (
    FEATURE_GROUPS["elo"]
    + FEATURE_GROUPS["pythag"]
    + FEATURE_GROUPS["rolling_form"]
    + FEATURE_GROUPS["streak"]
    + FEATURE_GROUPS["h2h"]
    + FEATURE_GROUPS["calendar"]
    + FEATURE_GROUPS["context"]
)

# Full V9 feature set (adds FG quality and park factors)
V9_FULL = V8_EQUIV + FEATURE_GROUPS["fg_pitching"] + FEATURE_GROUPS["fg_batting"] + FEATURE_GROUPS["park"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_splits() -> dict[str, pd.DataFrame]:
    splits = {}
    for name in ["train", "dev", "val", "test_2026"]:
        path = FEATURES_DIR / f"{name}_v9.parquet"
        if path.exists():
            splits[name] = pd.read_parquet(path)
            logger.info(f"  Loaded {name}: {splits[name].shape}")
        else:
            logger.warning(f"  {name}_v9.parquet not found — run 02_build_v9_dataset.py first")
    return splits


def prepare_xy(
    df: pd.DataFrame,
    features: list[str],
    fill_value: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.warning(f"    Missing {len(missing)} features: {missing[:5]}...")

    X = df[available].copy()
    # bool → int
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)
    # fill NaN
    X = X.fillna(fill_value)
    y = df[TARGET].astype(int)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X: pd.DataFrame, y: pd.Series, label: str = "") -> dict:
    """Compute all metrics for a model on given data."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, proba)
    brier = brier_score_loss(y, proba)
    ll = log_loss(y, proba)

    # Confidence-filtered accuracy
    results = {}
    for thresh in [0.55, 0.57, 0.60, 0.62, 0.65]:
        mask = (proba >= thresh) | (proba <= (1.0 - thresh))
        if mask.sum() >= 10:
            conf_acc = accuracy_score(y[mask], pred[mask])
            coverage = mask.mean()
        else:
            conf_acc = np.nan
            coverage = 0.0
        results[f"conf_acc_{int(thresh*100)}"] = conf_acc
        results[f"conf_cov_{int(thresh*100)}"] = coverage

    metrics = {
        "label": label,
        "accuracy": acc,
        "auc": auc,
        "brier": brier,
        "log_loss": ll,
        **results,
    }

    logger.info(
        f"  {label:<35} acc={acc:.4f}  auc={auc:.4f}  brier={brier:.4f}"
        f"  conf60={results.get('conf_acc_60', float('nan')):.4f}"
        f" ({results.get('conf_cov_60', 0):.1%} of games)"
    )
    return metrics


def walk_forward_cv(
    train_val_df: pd.DataFrame,
    features: list[str],
    model_factory,
    n_splits: int = 5,
) -> tuple[float, float]:
    """
    Walk-forward cross-validation using yearly splits.
    Each fold trains on all prior years, validates on the next year.
    Returns (mean_accuracy, std_accuracy).
    """
    seasons = sorted(train_val_df["season"].unique())
    if len(seasons) < 3:
        return np.nan, np.nan

    # Use last n_splits + 1 seasons for WF-CV
    wf_seasons = seasons[-(n_splits + 1):]
    accs = []

    for i in range(1, len(wf_seasons)):
        train_yrs = wf_seasons[:i]
        val_yr = wf_seasons[i]
        train = train_val_df[train_val_df["season"].isin(train_yrs)]
        val = train_val_df[train_val_df["season"] == val_yr]

        X_tr, y_tr = prepare_xy(train, features)
        X_val, y_val = prepare_xy(val, features)

        model = model_factory()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        accs.append(accuracy_score(y_val, pred))

    return float(np.mean(accs)), float(np.std(accs))


# ─────────────────────────────────────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────────────────────────────────────

def make_catboost(params: dict | None = None):
    defaults = {
        "iterations": 500, "depth": 6, "learning_rate": 0.05,
        "l2_leaf_reg": 3, "eval_metric": "Logloss",
        "random_seed": 42, "verbose": 0,
    }
    if params:
        defaults.update(params)
    return CatBoostClassifier(**defaults)


def make_xgboost(params: dict | None = None):
    defaults = {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "eval_metric": "logloss", "random_state": 42,
        "n_jobs": -1, "verbosity": 0,
    }
    if params:
        defaults.update(params)
    return xgb.XGBClassifier(**defaults)


def make_lightgbm(params: dict | None = None):
    defaults = {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
        "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    if params:
        defaults.update(params)
    return lgb.LGBMClassifier(**defaults)


def make_mlp(params: dict | None = None):
    defaults = {
        "hidden_layer_sizes": (128, 64, 32),
        "max_iter": 500, "random_state": 42,
        "early_stopping": True, "validation_fraction": 0.1,
        "alpha": 0.01,
    }
    if params:
        defaults.update(params)

    class MLPWithScaler:
        """MLP wrapped with StandardScaler (MLP is not scale-invariant)."""
        def __init__(self, **kw):
            self.scaler = StandardScaler()
            self.mlp = MLPClassifier(**kw)

        def fit(self, X, y):
            Xs = self.scaler.fit_transform(X)
            self.mlp.fit(Xs, y)
            return self

        def predict(self, X):
            return self.mlp.predict(self.scaler.transform(X))

        def predict_proba(self, X):
            return self.mlp.predict_proba(self.scaler.transform(X))

    class _F:
        def __call__(self):
            return MLPWithScaler(**defaults)

    return _F()()


# ─────────────────────────────────────────────────────────────────────────────
# Optuna tuning
# ─────────────────────────────────────────────────────────────────────────────

def tune_catboost(X_tr, y_tr, X_dev, y_dev, n_trials: int = 50) -> dict:
    """Tune CatBoost with Optuna."""
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "verbose": 0, "random_seed": 42,
        }
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_dev)[:, 1]
        return log_loss(y_dev, proba)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"    Best Optuna val log_loss: {study.best_value:.4f}")
    return study.best_params


def tune_xgboost(X_tr, y_tr, X_dev, y_dev, n_trials: int = 50) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "random_state": 42, "verbosity": 0, "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_dev)[:, 1]
        return log_loss(y_dev, proba)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

class WeightedEnsemble:
    """Soft-vote ensemble with learnable weights (optimized on dev set)."""

    def __init__(self, models: list, weights: list[float] | None = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self._classes = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        probas = np.array([m.predict_proba(X)[:, 1] for m in self.models])
        w = np.array(self.weights)
        w = w / w.sum()
        avg = (probas.T @ w).reshape(-1, 1)
        return np.hstack([1 - avg, avg])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @classmethod
    def optimize_weights(
        cls, models: list, X_dev: pd.DataFrame, y_dev: pd.Series
    ) -> "WeightedEnsemble":
        """Find optimal weights by minimizing log_loss on dev set."""
        probas = np.array([m.predict_proba(X_dev)[:, 1] for m in models])
        n = len(models)

        def objective(trial):
            raw_w = [trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(n)]
            w = np.array(raw_w)
            if w.sum() < 1e-6:
                return 1.0
            w = w / w.sum()
            avg_proba = (probas.T @ w)
            avg_proba = np.clip(avg_proba, 1e-7, 1 - 1e-7)
            return log_loss(y_dev, avg_proba)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, show_progress_bar=False)
        best_w = [study.best_params[f"w{i}"] for i in range(n)]
        best_w_arr = np.array(best_w) / np.sum(best_w)
        logger.info(f"    Optimal ensemble weights: {[f'{w:.3f}' for w in best_w_arr]}")
        return cls(models, list(best_w_arr))


class IsotonicEnsemble:
    """Ensemble with isotonic calibration applied to each model."""

    def __init__(self, models: list, X_cal: pd.DataFrame, y_cal: pd.Series):
        from sklearn.isotonic import IsotonicRegression
        self.models = models
        self.calibrators = []
        for m in models:
            prob = m.predict_proba(X_cal)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(prob.reshape(-1, 1), y_cal)
            self.calibrators.append(iso)

    def predict_proba(self, X) -> np.ndarray:
        cal_probas = []
        for m, cal in zip(self.models, self.calibrators):
            raw = m.predict_proba(X)[:, 1]
            cal_probas.append(cal.predict(raw.reshape(-1, 1)))
        avg = np.mean(cal_probas, axis=0).reshape(-1, 1)
        return np.hstack([1 - avg, avg])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# FLAML AutoML
# ─────────────────────────────────────────────────────────────────────────────

def run_flaml(X_tr, y_tr, X_val, y_val, time_budget: int = 300) -> dict:
    """Run FLAML AutoML and return results dict."""
    try:
        from flaml import AutoML
        logger.info(f"  Running FLAML AutoML (budget={time_budget}s)...")
        automl = AutoML()
        automl.fit(
            X_tr, y_tr,
            task="classification",
            metric="log_loss",
            time_budget=time_budget,
            seed=42,
            verbose=0,
        )
        proba = automl.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_val, pred)
        auc = roc_auc_score(y_val, proba)
        brier = brier_score_loss(y_val, proba)
        logger.info(f"  FLAML best model: {automl.best_estimator}")
        logger.info(f"  FLAML  acc={acc:.4f}  auc={auc:.4f}  brier={brier:.4f}")
        return {
            "label": f"FLAML_{automl.best_estimator}",
            "accuracy": acc,
            "auc": auc,
            "brier": brier,
            "best_estimator": automl.best_estimator,
            "best_config": automl.best_config,
            "model": automl,
        }
    except ImportError:
        logger.warning("  FLAML not installed — skipping")
        return {}
    except Exception as e:
        logger.error(f"  FLAML failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature importance from tree models."""
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "get_feature_importance"):
            imp = model.get_feature_importance()
        else:
            return pd.DataFrame()

        return (
            pd.DataFrame({"feature": feature_names[:len(imp)], "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Phase implementations
# ─────────────────────────────────────────────────────────────────────────────

def phase1_feature_ablation(splits: dict, all_results: list) -> None:
    """
    Phase 1: Feature group ablation study.
    Tests which feature groups add the most predictive value.
    """
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: FEATURE ABLATION STUDY")
    logger.info("="*60)

    # Combine train+dev for WF-CV; use val as holdout
    train = splits["train"]
    dev = splits["dev"]
    val = splits["val"]
    train_dev = pd.concat([train, dev], ignore_index=True)

    X_val_v9, y_val = prepare_xy(val, V9_FULL)
    X_val_v8, _ = prepare_xy(val, V8_EQUIV)

    # Baseline: V8-equivalent features
    logger.info("\nBaseline (V8-equivalent features):")
    model = make_catboost()
    X_tr, y_tr = prepare_xy(pd.concat([train, dev]), V8_EQUIV)
    model.fit(X_tr, y_tr)
    res = evaluate(model, X_val_v8, y_val, "V8_equiv_CatBoost")
    wf_mean, wf_std = walk_forward_cv(
        train_dev, V8_EQUIV,
        model_factory=lambda: make_catboost({"iterations": 300, "verbose": 0}),
        n_splits=4,
    )
    res["wf_cv_mean"] = wf_mean
    res["wf_cv_std"] = wf_std
    logger.info(f"    WF-CV: {wf_mean:.4f} ± {wf_std:.4f}")
    all_results.append(res)

    # Full V9 features
    logger.info("\nFull V9 features (V8 + FanGraphs pitching + batting + park):")
    model = make_catboost()
    X_tr, y_tr = prepare_xy(pd.concat([train, dev]), V9_FULL)
    model.fit(X_tr, y_tr)
    res = evaluate(model, X_val_v9, y_val, "V9_full_CatBoost")
    wf_mean, wf_std = walk_forward_cv(
        train_dev, V9_FULL,
        model_factory=lambda: make_catboost({"iterations": 300, "verbose": 0}),
        n_splits=4,
    )
    res["wf_cv_mean"] = wf_mean
    res["wf_cv_std"] = wf_std
    logger.info(f"    WF-CV: {wf_mean:.4f} ± {wf_std:.4f}")
    all_results.append(res)

    # Ablation: V8 + FG pitching only
    feats_fp = V8_EQUIV + FEATURE_GROUPS["fg_pitching"]
    model = make_catboost()
    X_tr, y_tr = prepare_xy(pd.concat([train, dev]), feats_fp)
    model.fit(X_tr, y_tr)
    X_v, _ = prepare_xy(val, feats_fp)
    res = evaluate(model, X_v, y_val, "V8+FG_pitching")
    all_results.append(res)

    # Ablation: V8 + FG batting only
    feats_fb = V8_EQUIV + FEATURE_GROUPS["fg_batting"]
    model = make_catboost()
    X_tr, y_tr = prepare_xy(pd.concat([train, dev]), feats_fb)
    model.fit(X_tr, y_tr)
    X_v, _ = prepare_xy(val, feats_fb)
    res = evaluate(model, X_v, y_val, "V8+FG_batting")
    all_results.append(res)

    # Ablation: Elo alone
    model = make_catboost({"iterations": 100})
    X_tr, y_tr = prepare_xy(pd.concat([train, dev]), FEATURE_GROUPS["elo"])
    model.fit(X_tr, y_tr)
    X_v, _ = prepare_xy(val, FEATURE_GROUPS["elo"])
    res = evaluate(model, X_v, y_val, "Elo_only")
    all_results.append(res)

    # Log feature importance for full V9 model
    imp = get_feature_importance(model, V9_FULL)
    if not imp.empty:
        logger.info("\n  Top 20 features by importance:")
        for _, row in imp.head(20).iterrows():
            logger.info(f"    {row['feature']:<40} {row['importance']:.4f}")


def phase2_model_comparison(
    splits: dict, all_results: list, quick: bool = False, no_mlp: bool = False
) -> dict[str, Any]:
    """
    Phase 2: Compare all model types on the full V9 feature set.
    """
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: MODEL COMPARISON")
    logger.info("="*60)

    train = splits["train"]
    dev = splits["dev"]
    val = splits["val"]

    X_tr, y_tr = prepare_xy(train, V9_FULL)
    X_dev, y_dev = prepare_xy(dev, V9_FULL)
    X_val, y_val = prepare_xy(val, V9_FULL)

    trained_models = {}

    # 2a. Logistic Regression (scale-normalized baseline)
    logger.info("\n[2a] Logistic Regression...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_tr_s, y_tr)

    class LRWrapped:
        def __init__(self, lr, scaler):
            self.lr = lr
            self.scaler = scaler
        def predict(self, X):
            return self.lr.predict(self.scaler.transform(X))
        def predict_proba(self, X):
            return self.lr.predict_proba(self.scaler.transform(X))

    lr_model = LRWrapped(lr, scaler)
    res = evaluate(lr_model, X_val, y_val, "LogisticRegression")
    all_results.append(res)
    trained_models["lr"] = lr_model

    # 2b. XGBoost
    logger.info("\n[2b] XGBoost...")
    xgb_model = make_xgboost()
    xgb_model.fit(X_tr, y_tr)
    res = evaluate(xgb_model, X_val, y_val, "XGBoost_default")
    all_results.append(res)
    trained_models["xgb"] = xgb_model

    # 2c. LightGBM
    logger.info("\n[2c] LightGBM...")
    lgb_model = make_lightgbm()
    lgb_model.fit(X_tr, y_tr)
    res = evaluate(lgb_model, X_val, y_val, "LightGBM_default")
    all_results.append(res)
    trained_models["lgb"] = lgb_model

    # 2d. CatBoost
    logger.info("\n[2d] CatBoost...")
    cb_model = make_catboost()
    cb_model.fit(X_tr, y_tr)
    res = evaluate(cb_model, X_val, y_val, "CatBoost_default")
    all_results.append(res)
    trained_models["cb"] = cb_model

    # 2e. MLP
    if not no_mlp:
        logger.info("\n[2e] MLP (128-64-32)...")
        mlp_model = make_mlp()
        mlp_model.fit(X_tr, y_tr)
        res = evaluate(mlp_model, X_val, y_val, "MLP_128-64-32")
        all_results.append(res)
        trained_models["mlp"] = mlp_model
    else:
        logger.info("\n[2e] MLP: skipped (--no-mlp)")

    # 2f. Optuna-tuned CatBoost (best historical performer)
    if not quick:
        logger.info("\n[2f] CatBoost + Optuna tuning (50 trials)...")
        best_params = tune_catboost(X_tr, y_tr, X_dev, y_dev, n_trials=50)
        best_params["verbose"] = 0
        cb_tuned = CatBoostClassifier(**best_params)
        cb_tuned.fit(X_tr, y_tr)
        res = evaluate(cb_tuned, X_val, y_val, "CatBoost_Optuna")
        all_results.append(res)
        trained_models["cb_tuned"] = cb_tuned

        logger.info("\n[2g] XGBoost + Optuna tuning (50 trials)...")
        best_xgb_params = tune_xgboost(X_tr, y_tr, X_dev, y_dev, n_trials=50)
        best_xgb_params.update({"random_state": 42, "verbosity": 0, "n_jobs": -1})
        xgb_tuned = xgb.XGBClassifier(**best_xgb_params)
        xgb_tuned.fit(X_tr, y_tr)
        res = evaluate(xgb_tuned, X_val, y_val, "XGBoost_Optuna")
        all_results.append(res)
        trained_models["xgb_tuned"] = xgb_tuned
    else:
        logger.info("\n[2f-g] Skipping Optuna tuning (--quick mode)")

    return trained_models


def phase3_ensemble(
    splits: dict, trained_models: dict, all_results: list
) -> None:
    """
    Phase 3: Ensemble experiments.
    """
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: ENSEMBLE EXPERIMENTS")
    logger.info("="*60)

    dev = splits["dev"]
    val = splits["val"]
    X_dev, y_dev = prepare_xy(dev, V9_FULL)
    X_val, y_val = prepare_xy(val, V9_FULL)

    core_models = [
        v for k, v in trained_models.items()
        if k in ("xgb", "lgb", "cb", "xgb_tuned", "cb_tuned")
        and v is not None
    ]
    if len(core_models) < 2:
        logger.warning("  Not enough models for ensemble (need ≥2)")
        return

    # Simple average ensemble
    logger.info("\n[3a] Simple average ensemble...")
    avg_ens = WeightedEnsemble(core_models)
    res = evaluate(avg_ens, X_val, y_val, "SimpleAvg_Ensemble")
    all_results.append(res)

    # Optimized-weight ensemble
    logger.info("\n[3b] Optimized-weight ensemble (Optuna weights on dev)...")
    opt_ens = WeightedEnsemble.optimize_weights(core_models, X_dev, y_dev)
    res = evaluate(opt_ens, X_val, y_val, "OptWeight_Ensemble")
    all_results.append(res)

    # Isotonic-calibrated ensemble
    logger.info("\n[3c] Isotonic-calibrated ensemble...")
    cal_ens = IsotonicEnsemble(core_models, X_dev, y_dev)
    res = evaluate(cal_ens, X_val, y_val, "Isotonic_Calibrated_Ensemble")
    all_results.append(res)


def phase4_confidence_curve(
    best_model, splits: dict, all_results: list
) -> None:
    """
    Phase 4: Generate full confidence vs accuracy curve.
    This is the key chart for professional-grade use.
    """
    logger.info("\n" + "="*60)
    logger.info("PHASE 4: CONFIDENCE VS ACCURACY CURVE")
    logger.info("="*60)

    val = splits["val"]
    X_val, y_val = prepare_xy(val, V9_FULL)
    proba = best_model.predict_proba(X_val)[:, 1]

    logger.info(f"\n{'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'N games':>10}")
    logger.info("-" * 45)

    conf_results = []
    for thresh in np.arange(0.50, 0.70, 0.01):
        mask = (proba >= thresh) | (proba <= (1.0 - thresh))
        n = mask.sum()
        if n < 20:
            break
        acc = accuracy_score(y_val[mask], (proba[mask] >= 0.5).astype(int))
        cov = mask.mean()
        conf_results.append({"threshold": float(thresh), "accuracy": acc, "coverage": cov, "n": int(n)})
        logger.info(f"{thresh:>10.2f} {cov:>10.1%} {acc:>10.4f} {n:>10d}")

    all_results.append({
        "label": "confidence_curve",
        "confidence_results": conf_results,
    })


def phase5_2026_test(best_model, splits: dict, all_results: list) -> None:
    """
    Phase 5: Test on 2026 games (only if we have them).
    """
    if "test_2026" not in splits or splits["test_2026"].empty:
        logger.info("\nPhase 5: No 2026 data available — skipping")
        return

    logger.info("\n" + "="*60)
    logger.info("PHASE 5: 2026 LIVE TEST")
    logger.info("="*60)

    test = splits["test_2026"]
    logger.info(f"  2026 games available: {len(test)}")
    logger.info(f"  Date range: {test['game_date'].min()} to {test['game_date'].max()}")

    X_test, y_test = prepare_xy(test, V9_FULL)
    if len(y_test) < 10:
        logger.warning("  Too few 2026 games for meaningful evaluation")
        return

    res = evaluate(best_model, X_test, y_test, "BestModel_2026_YTD")
    all_results.append(res)

    # Show per-team accuracy
    proba = best_model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test = test.copy()
    test["pred_home_win"] = pred
    test["correct"] = (pred == y_test.values).astype(int)
    test["prob"] = proba


def phase6_flaml(splits: dict, all_results: list, time_budget: int = 300) -> None:
    """Phase 6: FLAML AutoML (finds best algorithm automatically)."""
    logger.info("\n" + "="*60)
    logger.info("PHASE 6: FLAML AUTOML")
    logger.info("="*60)

    train = splits["train"]
    dev = splits["dev"]
    val = splits["val"]

    X_tr, y_tr = prepare_xy(pd.concat([train, dev]), V9_FULL)
    X_val, y_val = prepare_xy(val, V9_FULL)

    res = run_flaml(X_tr, y_tr, X_val, y_val, time_budget=time_budget)
    if res:
        all_results.append(res)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(all_results: list) -> None:
    """Print a formatted results comparison table."""
    logger.info("\n" + "="*90)
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("="*90)
    logger.info(
        f"{'Model':<40} {'Acc':>7} {'AUC':>7} {'Brier':>7} "
        f"{'Conf60Acc':>10} {'Conf60Cov':>10} {'WF-CV':>10}"
    )
    logger.info("-" * 90)

    # Filter out non-model results
    model_results = [r for r in all_results if r.get("accuracy") is not None]
    model_results.sort(key=lambda r: r.get("accuracy", 0), reverse=True)

    for r in model_results:
        wfcv = r.get("wf_cv_mean")
        wfcv_str = f"{wfcv:.4f}" if wfcv and not np.isnan(wfcv) else "  —  "
        c60 = r.get("conf_acc_60")
        c60_str = f"{c60:.4f}" if c60 and not np.isnan(c60) else "  —  "
        c60cov = r.get("conf_cov_60")
        c60cov_str = f"{c60cov:.1%}" if c60cov else "  —  "
        logger.info(
            f"{r['label']:<40} {r['accuracy']:>7.4f} {r['auc']:>7.4f} {r['brier']:>7.4f} "
            f"{c60_str:>10} {c60cov_str:>10} {wfcv_str:>10}"
        )


def main():
    parser = argparse.ArgumentParser(description="V9 model experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Skip Optuna tuning and FLAML (faster, ~5 min)")
    parser.add_argument("--no-mlp", action="store_true",
                        help="Skip MLP (can be slow on large datasets)")
    parser.add_argument("--phase", type=str, default="1,2,3,4,5,6",
                        help="Comma-separated phases to run (default: all)")
    parser.add_argument("--flaml-budget", type=int, default=300,
                        help="FLAML AutoML time budget in seconds (default: 300)")
    parser.add_argument("--rebuild-features", action="store_true",
                        help="Force rebuild feature dataset")
    args = parser.parse_args()

    phases = set(int(p.strip()) for p in args.phase.split(","))

    logger.info("=" * 70)
    logger.info("V9 EXPERIMENT — PROFESSIONAL ANALYST GRADE MLB PREDICTION")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # Load feature data (build if needed)
    logger.info("\nLoading V9 feature splits...")
    splits = load_splits()

    if not splits:
        logger.error("No V9 feature data found! Run 02_build_v9_dataset.py first.")
        return

    for name, df in splits.items():
        if not df.empty:
            fg_cols = [c for c in df.columns if "fg_" in c]
            logger.info(f"  {name}: {df.shape}, FanGraphs cols: {len(fg_cols)}")

    all_results: list[dict] = []
    trained_models: dict = {}

    t0 = time.time()

    if 1 in phases:
        phase1_feature_ablation(splits, all_results)

    if 2 in phases:
        trained_models = phase2_model_comparison(
            splits, all_results,
            quick=args.quick,
            no_mlp=args.no_mlp,
        )

    if 3 in phases and trained_models:
        phase3_ensemble(splits, trained_models, all_results)

    # Pick best model for phases 4+5
    best_result = max(
        (r for r in all_results if r.get("accuracy")),
        key=lambda r: r.get("accuracy", 0),
        default=None,
    )
    # Prefer CatBoost/XGBoost/LGB for pickling (MLP has local class issues)
    PICKLABLE = ["catboost", "xgboost", "lightgbm", "logistic"]
    best_model_key = None
    # Try to find best picklable model by accuracy
    best_acc = -1
    for k, m in trained_models.items():
        if any(p in k.lower() for p in PICKLABLE):
            # find its result
            for r in all_results:
                if r.get("label", "").lower() == k.lower() and r.get("accuracy", 0) > best_acc:
                    best_acc = r["accuracy"]
                    best_model_key = k
    if not best_model_key:
        best_model_key = list(trained_models.keys())[0]

    best_model = trained_models.get(best_model_key) if trained_models else None

    if 4 in phases and best_model:
        phase4_confidence_curve(best_model, splits, all_results)

    if 5 in phases and best_model:
        phase5_2026_test(best_model, splits, all_results)

    if 6 in phases and not args.quick:
        phase6_flaml(splits, all_results, time_budget=args.flaml_budget)

    elapsed = time.time() - t0

    # Print results table
    print_results_table(all_results)

    # Save best model — try in priority order, skip unpicklable local classes
    model_path = MODEL_DIR / "v9_best_model.pkl"
    SAVE_PRIORITY = ["XGBoost_Optuna", "CatBoost_Optuna", "XGBoost_default",
                     "CatBoost_default", "LightGBM_default"]
    saved = False
    for pref_key in SAVE_PRIORITY:
        m = trained_models.get(pref_key)
        if m is not None:
            try:
                with open(model_path, "wb") as f:
                    pickle.dump({"model": m, "model_key": pref_key,
                                 "features": V9_FULL, "results": all_results}, f)
                logger.info(f"\nBest model ({pref_key}) saved to: {model_path}")
                saved = True
                break
            except Exception as e:
                logger.warning(f"Could not pickle {pref_key}: {e}")
    if not saved:
        logger.warning("No model saved (all failed to pickle)")

    # Save all results to JSON
    results_path = LOG_DIR / "v9_experiment_results.json"
    serializable = []
    for r in all_results:
        safe_r = {k: v for k, v in r.items() if k != "model" and isinstance(v, (int, float, str, list, dict, type(None)))}
        serializable.append(safe_r)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info(f"\nAll results saved to: {results_path}")
    logger.info(f"Total experiment time: {elapsed/60:.1f} minutes")
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*70)

    # Key summary
    model_results = [r for r in all_results if r.get("accuracy")]
    if model_results:
        best = max(model_results, key=lambda r: r["accuracy"])
        logger.info(f"\n*** Best model: {best['label']}")
        logger.info(f"   Accuracy:    {best['accuracy']:.4f}")
        logger.info(f"   AUC:         {best['auc']:.4f}")
        conf60 = best.get("conf_acc_60")
        if conf60 and not np.isnan(conf60):
            logger.info(f"   Conf>=60%:   {conf60:.4f} on {best.get('conf_cov_60', 0):.1%} of games")
        wfcv = best.get("wf_cv_mean")
        if wfcv and not np.isnan(wfcv):
            logger.info(f"   WF-CV:       {wfcv:.4f} ± {best.get('wf_cv_std', 0):.4f}")

        v8_equiv = next((r for r in all_results if "V8_equiv" in r.get("label", "")), None)
        if v8_equiv:
            delta = best["accuracy"] - v8_equiv["accuracy"]
            logger.info(f"\n   vs V8-equiv: {delta:+.4f} ({delta*100:+.2f}%)")


if __name__ == "__main__":
    main()
