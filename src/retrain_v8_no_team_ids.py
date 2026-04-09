#!/usr/bin/env python3
"""
V8 Retrain — No Team ID Embeddings (v8_nocat)

The original V8 final ensemble has CatBoost sub-models that use home_team_id
and away_team_id as categorical features, giving them ~54% of the blend weight.
These team embeddings overfit to 2015-2025 team-specific home/away patterns and
drown out the quantitative signals (Elo, Pythagorean, run differential) that
actually contain the useful predictive information.

This script retrains the ensemble WITHOUT team ID categoricals:
  - CatBoost: uses CLEAN_FEATURES only (no team IDs)
  - LightGBM: same CLEAN_FEATURES
  - MLP:      same CLEAN_FEATURES
  - Optuna weights still optimized on 2025 val set
  - 2026 data joined in (training augmentation) if available

After training it evaluates on:
  A) 2025 holdout (same as original V8 — apples-to-apples)
  B) 2026 completed games (new — tests early-season generalization)

Outputs:
  models/game_outcome_2026_v8_nocat.pkl  — bundle ready for predict_today_games.py

Usage:
    python retrain_v8_no_team_ids.py                    # full retrain
    python retrain_v8_no_team_ids.py --dry-run          # skip save
    python retrain_v8_no_team_ids.py --no-2026          # skip 2026 augmentation
    python retrain_v8_no_team_ids.py --n-trials 100     # Optuna trials
"""

import argparse
import json
import logging
import os
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
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
        logging.FileHandler(Path(__file__).parent.parent / "logs" / "v8_nocat.log", mode="w"),
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data" / "training"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR   = BASE_DIR / "logs"

TARGET = "home_won"
PROJECT = "hankstank"

# All quantitative features — NO team ID categoricals
FEATURES = [
    # ===== ELO (highest-signal group) =====
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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_historical_data():
    """Load 2015-2024 training + 2025 validation parquets."""
    train_df = pd.read_parquet(DATA_DIR / "train_v8_2015_2024.parquet")
    val_df   = pd.read_parquet(DATA_DIR / "val_v8_2025.parquet")
    logger.info("Historical data: train=%d  val=%d", len(train_df), len(val_df))
    return train_df, val_df


def load_2026_completed_games() -> pd.DataFrame:
    """
    Load completed 2026 games with V8 features from BigQuery.
    Returns empty DataFrame if BQ unavailable or no games yet.
    """
    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    try:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT)
        sql = """
        WITH v8_latest AS (
            SELECT *
            FROM `hankstank.mlb_2026_season.game_v8_features`
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) = 1
        )
        SELECT
            g.game_pk,
            g.game_date,
            g.home_team_id,
            g.away_team_id,
            CAST(g.home_score > g.away_score AS INT64) AS home_won,
            v8.* EXCEPT (game_pk, game_date, home_team_id, away_team_id, computed_at, data_completeness)
        FROM `hankstank.mlb_2026_season.games` g
        JOIN v8_latest v8 ON g.game_pk = v8.game_pk
        WHERE g.status = 'Final'
          AND g.home_score IS NOT NULL
          AND g.game_date < CURRENT_DATE()
        ORDER BY g.game_date
        """
        df = bq.query(sql).to_dataframe()
        logger.info("Loaded %d completed 2026 games from BQ", len(df))
        return df
    except Exception as e:
        logger.warning("Could not load 2026 BQ data: %s — skipping 2026 augmentation", e)
        return pd.DataFrame()


def prepare_xy(df: pd.DataFrame, features: list):
    """Extract X, y from dataframe; fill missing features with median."""
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.debug("Features not in df (will be 0-filled): %s", missing[:5])

    X = pd.DataFrame(index=df.index)
    for f in features:
        if f in df.columns:
            X[f] = df[f]
        else:
            X[f] = 0.0

    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)
    # Also cast object/category columns that should be numeric
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = df[TARGET].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(proba: np.ndarray, y: np.ndarray, label: str) -> dict:
    acc   = accuracy_score(y, (proba > 0.5).astype(int))
    auc   = roc_auc_score(y, proba)
    brier = brier_score_loss(y, proba)
    ll    = log_loss(y, proba)
    avg_pred = proba.mean()

    conf = {}
    for offset in [0.05, 0.08, 0.10, 0.12, 0.15]:
        mask = np.abs(proba - 0.5) > offset
        n = mask.sum()
        if n >= 15:
            conf[f">={50+offset*100:.0f}%"] = {
                "acc": round(accuracy_score(y[mask], (proba[mask] > 0.5).astype(int)), 4),
                "n": int(n),
                "pct": round(n / len(y) * 100, 1),
            }

    logger.info("  %-38s  acc=%.4f  auc=%.4f  brier=%.4f  avg_pred=%.3f",
                label, acc, auc, brier, avg_pred)
    for k, v in conf.items():
        logger.info("    conf %s: %.4f  (%d games, %.1f%%)", k, v["acc"], v["n"], v["pct"])

    return {"label": label, "accuracy": acc, "auc": auc, "brier": brier,
            "log_loss": ll, "avg_pred": avg_pred, "conf": conf}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def tune_catboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series,
                  n_trials: int = 50) -> tuple:
    """Optuna-tuned CatBoost (no categorical features)."""
    logger.info("Tuning CatBoost (no team IDs) with %d Optuna trials...", n_trials)

    def objective(trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 100, 500),
            depth=trial.suggest_int("depth", 3, 6),
            learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
            l2_leaf_reg=trial.suggest_float("l2", 1.0, 10.0),
            subsample=trial.suggest_float("sub", 0.6, 1.0),
            random_seed=42, verbose=False,
        )
        tss = TimeSeriesSplit(n_splits=3)
        accs = []
        for tr_idx, vl_idx in tss.split(X_train):
            m = CatBoostClassifier(**params)
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            p = m.predict_proba(X_train.iloc[vl_idx])[:, 1]
            accs.append(accuracy_score(y_train.iloc[vl_idx], (p > 0.5).astype(int)))
        return np.mean(accs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    model = CatBoostClassifier(
        iterations=best["iterations"], depth=best["depth"],
        learning_rate=best["lr"], l2_leaf_reg=best["l2"],
        subsample=best["sub"], random_seed=42, verbose=False,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    logger.info("  CatBoost best params: %s", best)
    return model, proba


def build_lgb(X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame) -> tuple:
    """LightGBM with fixed good hyperparameters (from V8 extended experiments)."""
    model = lgb.LGBMClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.75, min_child_samples=20,
        reg_alpha=0.05, reg_lambda=1.5, random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    return model, proba


def build_mlp(X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame) -> tuple:
    """MLP with standard scaler (from V8 FINAL2)."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_vl_s = scaler.transform(X_val)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu", solver="adam",
        alpha=0.001, learning_rate_init=0.001,
        max_iter=300, early_stopping=True,
        validation_fraction=0.1,
        random_state=42, verbose=False,
    )
    model.fit(X_tr_s, y_train)
    proba = model.predict_proba(X_vl_s)[:, 1]
    return model, scaler, proba


def optimize_weights(probas: dict, y_val: np.ndarray, n_trials: int = 200) -> dict:
    """Optuna-optimized blend weights for the ensemble."""
    names = list(probas.keys())

    def objective(trial):
        w = {n: trial.suggest_float(f"w_{i}", 0.0, 1.0) for i, n in enumerate(names)}
        total = sum(w.values())
        if total < 0.01:
            return 0.0
        blended = sum(probas[n] * (w[n] / total) for n in names)
        return accuracy_score(y_val, (blended > 0.5).astype(int))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw_w = {names[int(k.replace("w_", ""))]: v for k, v in study.best_params.items()}
    total = sum(raw_w.values())
    return {n: v / total for n, v in raw_w.items()}


# ---------------------------------------------------------------------------
# Fill-value computation
# ---------------------------------------------------------------------------
def compute_fill_values(df: pd.DataFrame, features: list) -> dict:
    """Compute median fill values from training data."""
    fv = {}
    for f in features:
        if f in df.columns:
            med = df[f].median()
            fv[f] = float(med) if pd.notna(med) else 0.0
        else:
            fv[f] = 0.0
    return fv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true", help="Skip saving model")
    parser.add_argument("--no-2026",  action="store_true", help="Skip 2026 augmentation")
    parser.add_argument("--n-trials", type=int, default=60,
                        help="Optuna trials for CatBoost tuning (default: 60)")
    parser.add_argument("--weight-trials", type=int, default=200,
                        help="Optuna trials for blend weights (default: 200)")
    args = parser.parse_args()

    start = time.time()
    logger.info("="*65)
    logger.info("V8 RETRAIN — NO TEAM ID CATEGORICALS (v8_nocat)")
    logger.info("="*65)

    # --- Load data ---
    train_df, val_df = load_historical_data()

    # 2026 augmentation: add completed 2026 games to training set
    df_2026 = pd.DataFrame()
    if not args.no_2026:
        df_2026 = load_2026_completed_games()

    if not df_2026.empty:
        # Hold back last 20% of 2026 games for out-of-sample 2026 evaluation
        n_hold = max(10, int(len(df_2026) * 0.20))
        df_2026_train = df_2026.iloc[:-n_hold]
        df_2026_eval  = df_2026.iloc[-n_hold:]
        logger.info("2026 augmentation: %d train / %d held-out for 2026 eval",
                    len(df_2026_train), len(df_2026_eval))

        # Concatenate 2026 training games at the end of the historical train set
        # (temporal ordering maintained — no lookahead)
        combined_train = pd.concat([train_df, df_2026_train], ignore_index=True, sort=False)
    else:
        combined_train = train_df
        df_2026_eval   = pd.DataFrame()
        logger.info("No 2026 data available — training on historical only")

    X_train, y_train = prepare_xy(combined_train, FEATURES)
    X_val,   y_val   = prepare_xy(val_df, FEATURES)

    logger.info("\nFeatures: %d | Train: %d | Val (2025): %d",
                len(FEATURES), len(X_train), len(X_val))

    # --- Train sub-models ---
    logger.info("\n--- Training sub-models ---")
    all_probas = {}
    model_objects = {}
    scalers = {}

    cat_model, cat_proba = tune_catboost(
        X_train, y_train, X_val, y_val, n_trials=args.n_trials
    )
    all_probas["cat"] = cat_proba
    model_objects["cat"] = cat_model
    evaluate(cat_proba, y_val.values, "CatBoost (no team IDs) — 2025 val")

    lgb_model, lgb_proba = build_lgb(X_train, y_train, X_val)
    all_probas["lgb"] = lgb_proba
    model_objects["lgb"] = lgb_model
    evaluate(lgb_proba, y_val.values, "LightGBM — 2025 val")

    mlp_model, mlp_scaler, mlp_proba = build_mlp(X_train, y_train, X_val)
    all_probas["mlp"] = mlp_proba
    model_objects["mlp"] = mlp_model
    scalers["mlp"] = mlp_scaler
    evaluate(mlp_proba, y_val.values, "MLP — 2025 val")

    # --- Optimize blend weights on 2025 val ---
    logger.info("\n--- Optimizing ensemble weights on 2025 val (%d games) ---", len(y_val))
    weights = optimize_weights(all_probas, y_val.values, n_trials=args.weight_trials)
    logger.info("Optimal weights: %s", {k: round(v, 4) for k, v in weights.items()})

    total_w = sum(weights.values())
    blended_2025 = sum(all_probas[n] * (weights[n] / total_w) for n in all_probas)
    r_2025 = evaluate(blended_2025, y_val.values, "v8_nocat ensemble — 2025 holdout")

    # --- Compare against original V8 on 2025 val ---
    logger.info("\n--- Comparison: loading original V8 for 2025 val benchmark ---")
    orig_pkl = MODEL_DIR / "game_outcome_2026_v8_final.pkl"
    if orig_pkl.exists():
        import sys; sys.path.insert(0, str(Path(__file__).parent))
        from predict_today_games import V8EnsemblePredictor
        with open(orig_pkl, "rb") as f:
            orig_bundle = pickle.load(f)
        orig_ensemble = V8EnsemblePredictor(orig_bundle)
        orig_feat_rows = []
        for _, row in val_df.iterrows():
            feat = {f: orig_bundle["fill_values"].get(f, 0.0) for f in orig_ensemble.all_features}
            for col in orig_ensemble.all_features:
                if col in row.index and pd.notna(row[col]):
                    feat[col] = row[col]
            orig_feat_rows.append(feat)
        X_orig = pd.DataFrame(orig_feat_rows)
        orig_proba_2025 = orig_ensemble.predict_proba(X_orig)[:, 1]
        r_orig_2025 = evaluate(orig_proba_2025, y_val.values, "v8_final (original) — 2025 holdout")
    else:
        logger.info("  Original pkl not found — skipping comparison on 2025")
        r_orig_2025 = None

    # --- Evaluate on 2026 holdout (if available) ---
    r_nocat_2026 = None
    r_orig_2026  = None
    if not df_2026_eval.empty:
        logger.info("\n--- 2026 held-out games (%d) ---", len(df_2026_eval))
        X_2026_eval, y_2026_eval = prepare_xy(df_2026_eval, FEATURES)
        hold_probas = {}
        hold_probas["cat"] = cat_model.predict_proba(X_2026_eval)[:, 1]
        hold_probas["lgb"] = lgb_model.predict_proba(X_2026_eval)[:, 1]
        X_2026_s = mlp_scaler.transform(X_2026_eval)
        hold_probas["mlp"] = mlp_model.predict_proba(X_2026_s)[:, 1]

        blended_2026 = sum(hold_probas[n] * (weights[n] / total_w) for n in hold_probas)
        r_nocat_2026 = evaluate(blended_2026, y_2026_eval.values,
                                "v8_nocat ensemble — 2026 holdout")

        if orig_pkl.exists():
            orig_2026_rows = []
            for _, row in df_2026_eval.iterrows():
                feat = {f: orig_bundle["fill_values"].get(f, 0.0) for f in orig_ensemble.all_features}
                for col in orig_ensemble.all_features:
                    if col in row.index and pd.notna(row[col]):
                        feat[col] = row[col]
                orig_2026_rows.append(feat)
            X_orig_2026 = pd.DataFrame(orig_2026_rows)
            orig_proba_2026 = orig_ensemble.predict_proba(X_orig_2026)[:, 1]
            r_orig_2026 = evaluate(orig_proba_2026, y_2026_eval.values,
                                   "v8_final (original) — 2026 holdout")

    # --- Summary table ---
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*70)
    logger.info("  %-42s  %s  %s  %s  %s",
                "Model", "Acc-2025", "Acc-2026", "AUC-2025", "AvgPred")
    logger.info("  " + "-"*65)
    for r, r26 in [(r_2025, r_nocat_2026), (r_orig_2025, r_orig_2026)]:
        if r is None:
            continue
        a26_str = f"{r26['accuracy']:.4f}" if r26 else "  N/A  "
        logger.info("  %-42s  %.4f    %s    %.4f    %.3f",
                    r["label"], r["accuracy"], a26_str, r["auc"], r["avg_pred"])
    logger.info("="*70)

    if args.dry_run:
        logger.info("\n[DRY RUN] Model not saved.")
        return

    # --- Save bundle ---
    fill_values = compute_fill_values(combined_train, FEATURES)

    bundle = {
        "version": "v8_nocat",
        "trained_at": datetime.utcnow().isoformat(),
        "val_accuracy": r_2025["accuracy"],
        "val_auc": r_2025["auc"],
        "val_2026_accuracy": r_nocat_2026["accuracy"] if r_nocat_2026 else None,
        "models": {name: model_objects[name] for name in model_objects},
        "scalers": scalers,
        "weights": weights,
        "feature_sets": {name: FEATURES for name in model_objects},
        "fill_values": fill_values,
        "cat_feature_indices": [],  # No categorical features
        "n_train_games": len(combined_train),
        "n_2026_train_games": len(df_2026_train) if not df_2026.empty else 0,
    }

    out_path = MODEL_DIR / "game_outcome_2026_v8_nocat.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    elapsed = time.time() - start
    logger.info("\nSaved → %s  (%.1fs)", out_path, elapsed)

    # Save summary JSON for quick inspection
    summary = {
        "version": "v8_nocat",
        "trained_at": bundle["trained_at"],
        "n_train": len(combined_train),
        "n_2026_train": bundle["n_2026_train_games"],
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "results_2025_val": {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in r_2025.items() if k != "conf"},
        "results_2026_holdout": ({k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in r_nocat_2026.items() if k != "conf"}
                                 if r_nocat_2026 else None),
        "vs_original_2025": round(r_2025["accuracy"] - r_orig_2025["accuracy"], 4)
                             if r_orig_2025 else None,
        "vs_original_2026": round(r_nocat_2026["accuracy"] - r_orig_2026["accuracy"], 4)
                             if (r_nocat_2026 and r_orig_2026) else None,
    }
    with open(LOG_DIR / "v8_nocat_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved → logs/v8_nocat_results.json")


if __name__ == "__main__":
    main()
