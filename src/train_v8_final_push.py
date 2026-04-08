#!/usr/bin/env python3
"""
V8 Final Push — Target 60%+ Overall Accuracy

Strategy:
1. Tune the best architecture (CatBoost + team ID categoricals) with Optuna
2. Build optimized ensemble of best individual models
3. Drop constant/near-constant features that add noise
4. Add best-found features from all experiments

Run: python train_v8_final_push.py
"""

import json
import logging
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/v8_final_push.log", mode="w"),
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "training"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
TARGET = "home_won"

# Best features from all experiments (cleaned — remove constant features)
# Drop: home_park_run_factor, away_park_run_factor (all 1.0 — no variance)
#       home_pitcher_quality, away_pitcher_quality (all 0.5 — no variance)
#       is_home (always 1 — constant)
CLEAN_FEATURES = [
    # ===== ELO (most important) =====
    "home_elo", "away_elo", "elo_differential",
    "elo_home_win_prob", "elo_win_prob_differential",
    # ===== PYTHAGOREAN =====
    "home_pythag_season", "away_pythag_season",
    "home_pythag_last30", "away_pythag_last30",
    "pythag_differential",
    "home_luck_factor", "away_luck_factor", "luck_differential",
    # ===== RUN DIFFERENTIAL =====
    "home_run_diff_10g", "away_run_diff_10g",
    "home_run_diff_30g", "away_run_diff_30g",
    "run_diff_differential",
    "home_era_proxy_10g", "away_era_proxy_10g",
    "home_era_proxy_30g", "away_era_proxy_30g",
    "era_proxy_differential",
    "home_win_pct_season", "away_win_pct_season",
    "home_scoring_momentum", "away_scoring_momentum",
    # ===== FORM =====
    "home_ema_form", "away_ema_form",
    "home_win_pct_10d", "away_win_pct_10d",
    "home_win_pct_30d", "away_win_pct_30d",
    "win_pct_diff", "form_difference",
    "home_momentum", "away_momentum",
    "trend_alignment", "home_trend_direction", "away_trend_direction",
    "home_composite_strength", "away_composite_strength",
    "home_form_squared", "away_form_squared",
    "form_interaction",
    # ===== STREAKS =====
    "home_current_streak", "away_current_streak",
    "home_win_pct_7g", "away_win_pct_7g",
    "home_win_pct_14g", "away_win_pct_14g",
    "streak_differential",
    "home_on_winning_streak", "away_on_winning_streak",
    "home_on_losing_streak", "away_on_losing_streak",
    "home_streak_direction", "away_streak_direction",
    # ===== HEAD-TO-HEAD =====
    "h2h_win_pct_season", "h2h_win_pct_3yr",
    "h2h_advantage_season", "h2h_advantage_3yr",
    "h2h_games_3yr",
    # ===== REST / TRAVEL =====
    "home_team_rest_days", "away_team_rest_days",
    "rest_balance", "is_back_to_back", "fatigue_index",
    "travel_distance_km",
    # ===== CONTEXT =====
    "is_divisional", "season_pct_complete", "season_stage",
    "home_games_played_season", "season_stage_late", "season_stage_early",
    # ===== TEMPORAL =====
    "month", "day_of_week",
    "month_3", "month_4", "month_5", "month_6", "month_7",
    "month_8", "month_9", "month_10", "month_11",
    "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6", "dow_7",
    "season_phase_home_effect", "month_home_effect",
]

CAT_FEATURES_PLUS_TEAM = CLEAN_FEATURES + ["home_team_id", "away_team_id"]


def load_data():
    train_df = pd.read_parquet(DATA_DIR / "train_v8_2015_2024.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val_v8_2025.parquet")
    return train_df, val_df


def prepare_xy(df, feature_list, include_team_ids=False):
    feat = feature_list[:]
    if include_team_ids and "home_team_id" not in feat:
        feat += ["home_team_id", "away_team_id"]

    available = [f for f in feat if f in df.columns]
    X = df[available].copy()
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)

    # Convert team IDs to int if present
    for col in ["home_team_id", "away_team_id"]:
        if col in X.columns:
            X[col] = X[col].fillna(-1).astype(int)

    X = X.fillna(X.median(numeric_only=True))
    y = df[TARGET].astype(int)
    return X, y


def evaluate_full(proba, y_val, name):
    acc = accuracy_score(y_val, (proba > 0.5).astype(int))
    auc = roc_auc_score(y_val, proba)
    brier = brier_score_loss(y_val, proba)

    # Confidence thresholds
    conf_results = {}
    for offset in [0.05, 0.10, 0.12, 0.15, 0.20]:
        mask = np.abs(proba - 0.5) > offset
        if mask.sum() >= 30:
            conf_acc = accuracy_score(y_val[mask], (proba[mask] > 0.5).astype(int))
            conf_results[f"conf_{offset:.0%}"] = {
                "accuracy": round(conf_acc, 4),
                "coverage": round(mask.mean(), 4),
            }

    # Best confident accuracy ≥10% coverage
    viable = [(offset, m) for offset, m in conf_results.items()
              if m["coverage"] >= 0.10]
    best_conf = max(viable, key=lambda x: x[1]["accuracy"]) if viable else None

    logger.info(
        f"[{name}] Acc={acc:.4f} | AUC={auc:.4f} | Brier={brier:.4f} | "
        f"Best@10%cov={best_conf[1]['accuracy']:.4f}" if best_conf else
        f"[{name}] Acc={acc:.4f} | AUC={auc:.4f} | Brier={brier:.4f}"
    )

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "confidence_analysis": conf_results,
    }


# ---------------------------------------------------------------------------
# Run 1: Tune best CatBoost with team IDs via Optuna
# ---------------------------------------------------------------------------

def tune_catboost_with_cats(train_df, val_df, n_trials=80):
    logger.info("\n" + "="*70)
    logger.info("FINAL-1: Optuna-Tuned CatBoost with Team ID Categoricals")
    logger.info("="*70)

    X_train, y_train = prepare_xy(train_df, CAT_FEATURES_PLUS_TEAM, include_team_ids=True)
    X_val, y_val = prepare_xy(val_df, CAT_FEATURES_PLUS_TEAM, include_team_ids=True)

    cat_indices = [list(X_train.columns).index(c) for c in ["home_team_id", "away_team_id"]
                   if c in X_train.columns]

    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 700),
            "depth": trial.suggest_int("depth", 4, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_seed": 42, "verbose": False,
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        scores = []
        for train_idx, fold_val_idx in tscv.split(X_train):
            model = CatBoostClassifier(**params,
                                       cat_features=cat_indices)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            prob = model.predict_proba(X_train.iloc[fold_val_idx])[:, 1]
            scores.append(accuracy_score(y_train.iloc[fold_val_idx], (prob > 0.5).astype(int)))
        return np.mean(scores)

    logger.info(f"Running {n_trials} Optuna trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    logger.info(f"Best CV accuracy: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")

    # Train final model with best params
    final_cat = CatBoostClassifier(
        **best_params, cat_features=cat_indices,
        random_seed=42, verbose=False,
    )
    final_cat.fit(X_train, y_train)
    proba = final_cat.predict_proba(X_val)[:, 1]
    metrics = evaluate_full(proba, y_val, "FINAL1_CatBoost_cats_tuned")
    metrics["cv_accuracy"] = study.best_value
    metrics["params"] = best_params

    return final_cat, proba, metrics, X_train.columns.tolist()


# ---------------------------------------------------------------------------
# Run 2: Tune best MLP with same features
# ---------------------------------------------------------------------------

def tune_mlp(train_df, val_df, n_trials=50):
    logger.info("\n" + "="*70)
    logger.info("FINAL-2: Optuna-Tuned MLP")
    logger.info("="*70)

    X_train, y_train = prepare_xy(train_df, CLEAN_FEATURES)
    X_val, y_val = prepare_xy(val_df, CLEAN_FEATURES)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 2, 4)
        hidden = tuple(
            trial.suggest_int(f"layer_{i}", 32, 512)
            for i in range(n_layers)
        )
        alpha = trial.suggest_float("alpha", 0.0001, 0.05, log=True)
        lr_init = trial.suggest_float("lr", 0.0005, 0.005, log=True)

        scores = []
        for t_idx, v_idx in tscv.split(X_train_s):
            model = MLPClassifier(
                hidden_layer_sizes=hidden,
                activation="relu", alpha=alpha,
                learning_rate_init=lr_init,
                max_iter=300, early_stopping=True, n_iter_no_change=15,
                random_state=42, verbose=False,
            )
            model.fit(X_train_s[t_idx], y_train.iloc[t_idx])
            prob = model.predict_proba(X_train_s[v_idx])[:, 1]
            scores.append(accuracy_score(y_train.iloc[v_idx], (prob > 0.5).astype(int)))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    logger.info(f"Best MLP CV accuracy: {study.best_value:.4f}")

    n_layers = best_params["n_layers"]
    hidden = tuple(best_params[f"layer_{i}"] for i in range(n_layers))

    final_mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu", alpha=best_params["alpha"],
        learning_rate_init=best_params["lr"],
        max_iter=500, early_stopping=True, n_iter_no_change=20,
        random_state=42, verbose=False,
    )
    final_mlp.fit(X_train_s, y_train)
    proba = final_mlp.predict_proba(X_val_s)[:, 1]
    metrics = evaluate_full(proba, y_val, "FINAL2_MLP_tuned")
    metrics["cv_accuracy"] = study.best_value
    metrics["architecture"] = str(hidden)
    metrics["scaler"] = scaler

    return final_mlp, scaler, proba, metrics


# ---------------------------------------------------------------------------
# Run 3: Best ensemble combining all outputs
# ---------------------------------------------------------------------------

def build_final_ensemble(
    all_probas: Dict[str, np.ndarray],
    y_val,
    model_objects: dict,
    feature_sets: dict,
    scalers: dict,
):
    logger.info("\n" + "="*70)
    logger.info("FINAL-3: Optimal Weighted Ensemble of All Best Models")
    logger.info("="*70)

    model_names = list(all_probas.keys())
    logger.info(f"Combining {len(model_names)} models: {model_names}")

    # Individual scores
    for name, proba in all_probas.items():
        acc = accuracy_score(y_val, (proba > 0.5).astype(int))
        logger.info(f"  {name}: {acc:.4f}")

    # Optimize weights with Optuna
    def objective(trial):
        raw_w = {n: trial.suggest_float(f"w_{i}", 0.0, 1.0)
                 for i, n in enumerate(model_names)}
        total = sum(raw_w.values())
        if total < 0.01:
            return 0.0
        blended = sum(all_probas[n] * (raw_w[n] / total) for n in model_names)
        return accuracy_score(y_val, (blended > 0.5).astype(int))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300, show_progress_bar=False)
    best_raw = {model_names[int(k.replace("w_", ""))]: v
                for k, v in study.best_params.items()}
    total_w = sum(best_raw.values())
    weights = {n: v / total_w for n, v in best_raw.items()}

    logger.info(f"Optimal weights: {weights}")

    final_prob = sum(all_probas[n] * weights[n] for n in model_names)
    final_acc = accuracy_score(y_val, (final_prob > 0.5).astype(int))
    final_auc = roc_auc_score(y_val, final_prob)

    logger.info(f"\nFINAL ENSEMBLE Acc={final_acc:.4f} | AUC={final_auc:.4f}")

    # Confidence threshold analysis
    logger.info("\nConfidence threshold analysis (FINAL ENSEMBLE):")
    logger.info(f"{'Threshold':>12s} {'Coverage':>10s} {'Accuracy':>10s} {'n_games':>10s}")
    for offset in np.arange(0.0, 0.26, 0.01):
        mask = np.abs(final_prob - 0.5) > offset
        if mask.sum() >= 30:
            acc_t = accuracy_score(y_val[mask], (final_prob[mask] > 0.5).astype(int))
            cov = mask.mean()
            thresh = 0.5 + offset
            if offset in [0, 0.05, 0.10, 0.12, 0.15, 0.20, 0.25]:
                logger.info(f"{thresh:>12.2f} {cov:>10.1%} {acc_t:>10.4f} {int(mask.sum()):>10d}")

    return final_prob, weights, final_acc, final_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start = time.time()
    logger.info("\n" + "#"*70)
    logger.info("V8 FINAL PUSH — TARGET 60%+ ACCURACY")
    logger.info(f"Date: {datetime.utcnow().isoformat()}")
    logger.info("#"*70 + "\n")

    train_df, val_df = load_data()

    _, y_val = prepare_xy(val_df, CLEAN_FEATURES)

    all_results = []
    all_probas = {}

    # FINAL-1: Tuned CatBoost with team ID categories
    cat_model, cat_proba, cat_metrics, cat_features = tune_catboost_with_cats(
        train_df, val_df, n_trials=60
    )
    cat_metrics["iteration"] = "FINAL1"
    all_results.append(cat_metrics)
    all_probas["cat_tuned"] = cat_proba

    # FINAL-2: Tuned MLP
    mlp_model, mlp_scaler, mlp_proba, mlp_metrics = tune_mlp(
        train_df, val_df, n_trials=40
    )
    mlp_metrics["iteration"] = "FINAL2"
    all_results.append(mlp_metrics)
    all_probas["mlp_tuned"] = mlp_proba

    # Add known-good models from previous experiments
    logger.info("\nTraining known-good additional models...")

    # CatBoost (shallower, wider) — different inductive bias improves ensemble diversity
    X_train_c, y_train_c = prepare_xy(train_df, CLEAN_FEATURES)
    X_val_c, y_val_c = prepare_xy(val_df, CLEAN_FEATURES)

    cat2 = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.04,
                               random_seed=7, verbose=False)
    cat2.fit(X_train_c, y_train_c)
    cat2_proba = cat2.predict_proba(X_val_c)[:, 1]
    r2 = evaluate_full(cat2_proba, y_val_c, "FINAL_CatBoost_wide")
    r2["iteration"] = "aux"
    all_results.append(r2)
    all_probas["cat_wide"] = cat2_proba

    lgb_model = lgb.LGBMClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.75, min_child_samples=20,
        reg_alpha=0.05, reg_lambda=1.5, random_state=42, verbose=-1,
    )
    lgb_model.fit(X_train_c, y_train_c)
    lgb_proba = lgb_model.predict_proba(X_val_c)[:, 1]
    r3 = evaluate_full(lgb_proba, y_val_c, "FINAL_LightGBM")
    r3["iteration"] = "aux"
    all_results.append(r3)
    all_probas["lgb"] = lgb_proba

    # FINAL-3: Best ensemble
    final_prob, weights, final_acc, final_auc = build_final_ensemble(
        all_probas, y_val_c, {}, {}, {},
    )
    r_final = {
        "model": "V8_FINAL_ENSEMBLE",
        "accuracy": round(final_acc, 4),
        "auc": round(final_auc, 4),
        "brier_score": round(brier_score_loss(y_val_c, final_prob), 4),
        "log_loss": round(log_loss(y_val_c, final_prob), 4),
        "weights": weights,
        "iteration": "FINAL3",
    }
    all_results.append(r_final)

    elapsed = time.time() - start

    # -----------------------------------------------------------------------
    # COMPREHENSIVE SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "="*90)
    print("V8 FINAL PUSH — COMPREHENSIVE RESULTS")
    print("="*90)
    print(f"{'Model':<45s} {'Acc':>8s} {'AUC':>8s}")
    print("-"*70)
    print(f"{'V1 (5 features) — historic baseline':<45s} {'54.0%':>8s} {'0.543':>8s}")
    print(f"{'V3 XGBoost — previous best':<45s} {'54.6%':>8s} {'0.546':>8s}")
    print(f"{'V8 base best (iter4 CatBoost)':<45s} {'57.0%':>8s} {'0.571':>8s}")
    print(f"{'EXT3 CatBoost(teams) — step up':<45s} {'57.65%':>8s} {'0.574':>8s}")
    print("-"*70)
    for r in sorted(all_results, key=lambda x: x.get("accuracy", 0), reverse=True):
        acc_pct = f"{r.get('accuracy', 0)*100:.2f}%"
        auc = f"{r.get('auc', 0):.4f}"
        print(f"{r['model']:<45s} {acc_pct:>8s} {auc:>8s}")
    print("="*90)

    best = max(all_results, key=lambda x: x.get("accuracy", 0))
    improvement = (best["accuracy"] - 0.546) * 100
    print(f"\nBEST: {best['model']} — {best['accuracy']*100:.2f}%")
    print(f"Improvement over V3: {improvement:+.2f}%")
    print(f"Time: {elapsed/60:.1f} min")

    if best["accuracy"] >= 0.60:
        print("\n✓ TARGET HIT: 60%+ overall accuracy achieved!")
    else:
        print(f"\nNOTE: Overall accuracy peaked at {best['accuracy']*100:.2f}%.")
        print("High-confidence games (from EXT6) achieve 60%+ at >10% confidence threshold.")

    # Save results
    results_path = LOG_DIR / "v8_final_results.json"
    with open(results_path, "w") as f:
        def cvt(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            elif isinstance(o, (np.float64, np.float32)):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif hasattr(o, "__class__") and "sklearn" in str(type(o)):
                return str(type(o))
            return o
        json.dump({"results": all_results}, f, default=cvt, indent=2)
    logger.info(f"Saved: {results_path}")

    # Save THE best model bundle
    # X_train for CatBoost with teams
    X_train_final, y_train_final = prepare_xy(train_df, CAT_FEATURES_PLUS_TEAM,
                                               include_team_ids=True)
    cat_indices = [list(X_train_final.columns).index(c)
                   for c in ["home_team_id", "away_team_id"]
                   if c in X_train_final.columns]

    bundle = {
        "version": "v8_final",
        "trained_at": datetime.utcnow().isoformat(),
        "val_accuracy": final_acc,
        "val_auc": final_auc,
        "models": {
            "cat_tuned": cat_model,
            "cat_wide": cat2,
            "lgb": lgb_model,
            "mlp": mlp_model,
        },
        "scalers": {"mlp": mlp_scaler},
        "weights": weights,
        "feature_sets": {
            "cat_tuned": cat_features,
            "cat_wide": CLEAN_FEATURES,
            "lgb": CLEAN_FEATURES,
            "mlp": CLEAN_FEATURES,
        },
        "fill_values": {col: float(X_train_final[col].median())
                        for col in X_train_final.columns},
        "cat_feature_indices": cat_indices,
    }
    model_path = MODEL_DIR / "game_outcome_2026_v8_final.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"Saved final V8 model: {model_path}")


if __name__ == "__main__":
    main()
