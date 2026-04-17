#!/usr/bin/env python3
"""
V10 Model Experiment — Game-Level SP Quality + Park Factors + Rest/Travel

Builds directly on V9 results and tests whether the three new V10 feature
groups actually improve predictions.

Phases:
  1. ABLATION     — which V10 feature group adds the most? (SP / park / rest)
  2. V9 vs V10    — XGBoost_Optuna trained on V9 features vs V10 features
  3. WF-CV        — walk-forward CV to get reliable estimate of improvement
  4. CONFIDENCE   — confidence curve comparison: V9 vs V10
  5. 2026 TEST    — evaluate on live 2026 games

V9 baseline for comparison:
  XGBoost_Optuna: 57.33% val acc, 0.5884 AUC (2025 holdout)
  2026 live: 58.66% acc, 0.6135 AUC

Usage:
    python 03_train_v10_experiment.py
    python 03_train_v10_experiment.py --quick      # fast run, no Optuna
    python 03_train_v10_experiment.py --phase 1    # single phase
"""

import argparse
import json
import logging
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path(__file__).parent.parent.parent / "logs" / "v10_experiment.log",
            mode="w", encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR  = REPO_ROOT / "data" / "v10" / "features"
V9_FEATS_DIR  = REPO_ROOT / "data" / "v9" / "features"
MODEL_DIR     = REPO_ROOT / "models" / "v10"
LOG_DIR       = REPO_ROOT / "logs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "home_won"
EXCLUDE_COLS = {
    TARGET, "game_pk", "game_date", "season", "home_team_id", "away_team_id",
    "home_score", "away_score", "home_team_name", "away_team_name",
    "home_team_fg", "away_team_fg", "_type", "fetch_team", "venue_id",
    "rest_differential",  # included explicitly below — just avoiding accidental double
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature group definitions
# ─────────────────────────────────────────────────────────────────────────────

# V9 features (inherited — same as V9 FULL set)
V9_BASE_GROUPS = {
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
        "home_fg_xfip", "away_fg_xfip",
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
        "home_fg_woba", "away_fg_woba",
        "home_fg_ev_pct", "away_fg_ev_pct",
        "home_fg_hh_pct", "away_fg_hh_pct",
        "home_fg_brl_pct", "away_fg_brl_pct",
        "fg_ops_differential", "fg_woba_differential", "fg_obp_differential",
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

# V10 NEW feature groups
V10_NEW_GROUPS = {
    "sp_quality": [
        "home_sp_xera", "away_sp_xera", "sp_xera_diff",
        "home_sp_k_pct", "away_sp_k_pct", "sp_k_pct_diff",
        "home_sp_bb_pct", "away_sp_bb_pct", "sp_bb_pct_diff",
        "home_sp_whiff", "away_sp_whiff", "sp_whiff_diff",
        "home_sp_fbv", "away_sp_fbv", "sp_fbv_diff",
        "sp_quality_composite_diff",
        "home_sp_known", "away_sp_known",
    ],
    "park_factors": [
        "home_park_factor", "home_park_factor_100",
        "is_hitter_park", "is_pitcher_park",
        "park_factor_known",
    ],
    "rest_travel": [
        "home_days_rest", "away_days_rest", "rest_differential",
        "away_road_trip_length",
        "home_rested", "away_tired", "long_road_trip",
    ],
    "series_context": [
        "series_game_number", "games_in_series", "is_series_opener",
    ],
}

# V9 full feature set (from V9 experiment, re-run with V10 data)
def _v9_features() -> list[str]:
    return (
        V9_BASE_GROUPS["elo"]
        + V9_BASE_GROUPS["pythag"]
        + V9_BASE_GROUPS["rolling_form"]
        + V9_BASE_GROUPS["streak"]
        + V9_BASE_GROUPS["h2h"]
        + V9_BASE_GROUPS["fg_pitching"]
        + V9_BASE_GROUPS["fg_batting"]
        + ["home_park_factor", "home_park_factor_100"]   # park from V10 data
        + V9_BASE_GROUPS["calendar"]
        + V9_BASE_GROUPS["context"]
    )

# V10 full feature set (all V9 + all new V10 groups)
def _v10_features() -> list[str]:
    return (
        _v9_features()
        + V10_NEW_GROUPS["sp_quality"]
        + V10_NEW_GROUPS["rest_travel"]
        + V10_NEW_GROUPS["series_context"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_splits() -> dict[str, pd.DataFrame]:
    splits = {}
    for name in ["train", "dev", "val", "test_2026"]:
        path = FEATURES_DIR / f"{name}_v10.parquet"
        if path.exists():
            splits[name] = pd.read_parquet(path)
            logger.info(f"  Loaded {name}: {splits[name].shape}")
        else:
            raise FileNotFoundError(f"{path} not found — run 02_build_v10_dataset.py first")
    return splits


def prepare_xy(df: pd.DataFrame, features: list[str], fill: float = 0.0):
    # Deduplicate while preserving order (duplicate feature names crash XGBoost)
    features = list(dict.fromkeys(features))
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        logger.warning(f"    Missing {len(missing)} features: {missing[:8]}...")
    X = df[available].copy()
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)
    X = X.fillna(fill)
    y = df[TARGET].astype(int)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X: pd.DataFrame, y, label: str = "") -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    acc   = accuracy_score(y, pred)
    auc   = roc_auc_score(y, proba)
    brier = brier_score_loss(y, proba)

    conf_results = {}
    for thresh in [0.55, 0.57, 0.60, 0.62, 0.64, 0.65, 0.67]:
        mask = (proba >= thresh) | (proba <= (1.0 - thresh))
        if mask.sum() >= 10:
            conf_acc = accuracy_score(y[mask], pred[mask])
            coverage = mask.mean()
        else:
            conf_acc = np.nan
            coverage = 0.0
        conf_results[f"conf_acc_{int(thresh*100)}"] = conf_acc
        conf_results[f"conf_cov_{int(thresh*100)}"] = coverage

    metrics = {"label": label, "accuracy": acc, "auc": auc, "brier": brier, **conf_results}

    logger.info(
        f"  {label:<40} acc={acc:.4f}  auc={auc:.4f}  brier={brier:.4f}"
        f"  conf60={conf_results.get('conf_acc_60', float('nan')):.4f}"
        f" ({conf_results.get('conf_cov_60', 0):.1%})"
    )
    return metrics


def walk_forward_cv(df: pd.DataFrame, features: list[str], model_factory, n_splits: int = 5):
    seasons = sorted(df["season"].unique())
    wf_seasons = seasons[-(n_splits + 1):]
    accs = []
    for i in range(1, len(wf_seasons)):
        train_yrs = wf_seasons[:i]
        val_yr    = wf_seasons[i]
        tr  = df[df["season"].isin(train_yrs)]
        val = df[df["season"] == val_yr]
        X_tr,  y_tr  = prepare_xy(tr,  features)
        X_val, y_val = prepare_xy(val, features)
        m = model_factory()
        m.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_val, m.predict(X_val)))
    return float(np.mean(accs)), float(np.std(accs))


# ─────────────────────────────────────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────────────────────────────────────

def make_xgb_default():
    return xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42, verbosity=0,
    )


def tune_xgboost(X_tr, y_tr, X_val, y_val, n_trials: int = 80) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        m = xgb.XGBClassifier(
            **params, eval_metric="logloss",
            use_label_encoder=False, random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr)
        return accuracy_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def make_lr(C: float = 0.1):
    return LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Ablation — which V10 feature group adds the most?
# ─────────────────────────────────────────────────────────────────────────────

def phase1_ablation(splits: dict) -> list[dict]:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: FEATURE ABLATION (V10 groups)")
    logger.info("  Which new feature group adds the most value over V9?")
    logger.info("=" * 60)

    train = splits["train"]
    val   = splits["val"]
    v9_feats  = _v9_features()
    v10_feats = _v10_features()

    X_tr_v9,  y_tr = prepare_xy(train, v9_feats)
    X_val_v9, y_val = prepare_xy(val,  v9_feats)

    # Baseline: V9 features with XGBoost default
    logger.info("  Training V9 baseline (XGBoost default)...")
    base_model = make_xgb_default()
    base_model.fit(X_tr_v9, y_tr)
    base_metrics = evaluate(base_model, X_val_v9, y_val, "V9_baseline (rerun on V10 data)")

    results = [base_metrics]

    # Test each new group independently (V9 + one group)
    for group_name, group_feats in V10_NEW_GROUPS.items():
        feats = v9_feats + group_feats
        X_tr,  _    = prepare_xy(train, feats)
        X_val_g, _ = prepare_xy(val,   feats)
        m = make_xgb_default()
        m.fit(X_tr, y_tr)
        m_metrics = evaluate(m, X_val_g, y_val, f"V9 + {group_name}")
        m_metrics["delta_acc"] = m_metrics["accuracy"] - base_metrics["accuracy"]
        results.append(m_metrics)

    # All V10 features combined
    X_tr_v10,  _ = prepare_xy(train, v10_feats)
    X_val_v10, _ = prepare_xy(val,   v10_feats)
    full_m = make_xgb_default()
    full_m.fit(X_tr_v10, y_tr)
    full_metrics = evaluate(full_m, X_val_v10, y_val, "V10_full (all groups)")
    full_metrics["delta_acc"] = full_metrics["accuracy"] - base_metrics["accuracy"]
    results.append(full_metrics)

    logger.info("\nPhase 1 Summary — Accuracy delta vs V9 baseline:")
    for r in results[1:]:
        delta = r.get("delta_acc", 0)
        sign = "+" if delta >= 0 else ""
        logger.info(f"  {r['label']:<40} {sign}{delta*100:.2f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: V9 vs V10 with Optuna-tuned XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def phase2_v9_vs_v10(splits: dict, quick: bool = False) -> list[dict]:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: V9 vs V10 — Optuna-tuned XGBoost")
    logger.info("=" * 60)

    train = splits["train"]
    dev   = splits["dev"]
    val   = splits["val"]
    train_dev = pd.concat([train, dev], ignore_index=True)

    results = []

    for label, feats in [("V9_features", _v9_features()), ("V10_features", _v10_features())]:
        logger.info(f"\n  Tuning {label} ({len(feats)} features)...")
        X_tr,   y_tr  = prepare_xy(train,     feats)
        X_dev,  y_dev = prepare_xy(dev,        feats)
        X_val,  y_val = prepare_xy(val,        feats)

        if quick:
            best_params = {}
            logger.info("  Quick mode: using default XGBoost params")
        else:
            n_trials = 60
            logger.info(f"  Optuna: {n_trials} trials on 2024 dev set...")
            t0 = time.time()
            best_params = tune_xgboost(X_tr, y_tr, X_dev, y_dev, n_trials=n_trials)
            logger.info(f"  Tuning done in {time.time()-t0:.0f}s. Best: {best_params}")

        # Retrain on train+dev with best params
        X_td, y_td = prepare_xy(train_dev, feats)
        model = xgb.XGBClassifier(
            **best_params,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, verbosity=0,
        ) if best_params else make_xgb_default()
        model.fit(X_td, y_td)

        metrics = evaluate(model, X_val, y_val, f"{label}_XGB_Optuna")
        metrics["best_params"] = best_params
        metrics["n_features"]  = len([f for f in feats if f in val.columns])
        results.append(metrics)

        # Save model
        model_path = MODEL_DIR / f"xgb_optuna_{label.lower()}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # Delta
    if len(results) == 2:
        delta = results[1]["accuracy"] - results[0]["accuracy"]
        sign = "+" if delta >= 0 else ""
        logger.info(f"\n  V10 vs V9 accuracy delta: {sign}{delta*100:.2f}%")
        results[1]["delta_vs_v9"] = delta

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Walk-forward CV
# ─────────────────────────────────────────────────────────────────────────────

def phase3_wf_cv(splits: dict) -> list[dict]:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: WALK-FORWARD CV (5 yearly folds)")
    logger.info("=" * 60)

    train_dev = pd.concat([splits["train"], splits["dev"]], ignore_index=True)
    results = []

    for label, feats in [
        ("V9_WF-CV", _v9_features()),
        ("V10_WF-CV", _v10_features()),
    ]:
        logger.info(f"  Running {label}...")
        mean, std = walk_forward_cv(train_dev, feats, make_xgb_default, n_splits=5)
        logger.info(f"  {label}: {mean:.4f} +/- {std:.4f}")
        results.append({"label": label, "wf_cv_mean": mean, "wf_cv_std": std})

    if len(results) == 2:
        delta = results[1]["wf_cv_mean"] - results[0]["wf_cv_mean"]
        sign = "+" if delta >= 0 else ""
        logger.info(f"  V10 vs V9 WF-CV delta: {sign}{delta*100:.2f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Confidence curve comparison
# ─────────────────────────────────────────────────────────────────────────────

def phase4_confidence_curve(splits: dict) -> dict:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: CONFIDENCE CURVE — V9 vs V10")
    logger.info("=" * 60)

    train = splits["train"]
    val   = splits["val"]

    curves = {}
    thresholds = [0.50, 0.52, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.62, 0.64, 0.65, 0.67]

    for label, feats in [("V9", _v9_features()), ("V10", _v10_features())]:
        X_tr, y_tr = prepare_xy(train, feats)
        X_val, y_val = prepare_xy(val,   feats)

        model = make_xgb_default()
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        curve = []
        for thresh in thresholds:
            mask = (proba >= thresh) | (proba <= (1.0 - thresh))
            if mask.sum() >= 10:
                acc = accuracy_score(y_val[mask], pred[mask])
                cov = mask.mean()
            else:
                acc, cov = np.nan, 0.0
            curve.append({"threshold": thresh, "accuracy": acc, "coverage": cov})
        curves[label] = curve

    # Print comparison table
    logger.info(f"\n  {'Threshold':>12}  {'V9 Acc':>8}  {'V10 Acc':>8}  {'Delta':>7}  {'V10 Cov':>8}")
    logger.info(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}")
    for i, thresh in enumerate(thresholds):
        v9_acc  = curves["V9"][i]["accuracy"]
        v10_acc = curves["V10"][i]["accuracy"]
        v10_cov = curves["V10"][i]["coverage"]
        if not (np.isnan(v9_acc) or np.isnan(v10_acc)):
            delta = v10_acc - v9_acc
            sign  = "+" if delta >= 0 else ""
            logger.info(
                f"  >= {thresh:.2f}          {v9_acc:.4f}    {v10_acc:.4f}   {sign}{delta:.4f}   {v10_cov:.1%}"
            )

    return curves


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: 2026 live test
# ─────────────────────────────────────────────────────────────────────────────

def phase5_2026_test(splits: dict, quick: bool = False) -> list[dict]:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 5: 2026 LIVE TEST")
    logger.info("=" * 60)

    train_dev_val = pd.concat(
        [splits["train"], splits["dev"], splits["val"]], ignore_index=True
    )
    test = splits["test_2026"]
    results = []

    for label, feats in [("V9_features", _v9_features()), ("V10_features", _v10_features())]:
        X_tr,   y_tr   = prepare_xy(train_dev_val, feats)
        X_test, y_test = prepare_xy(test,          feats)

        model = make_xgb_default()
        model.fit(X_tr, y_tr)

        metrics = evaluate(model, X_test, y_test, f"{label}_2026")
        results.append(metrics)
        logger.info(f"  {label}: {len(test)} 2026 games evaluated")

    if len(results) == 2:
        delta = results[1]["accuracy"] - results[0]["accuracy"]
        sign = "+" if delta >= 0 else ""
        logger.info(f"\n  V10 vs V9 2026 test delta: {sign}{delta*100:.2f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip Optuna tuning")
    parser.add_argument("--phase", default="1,2,3,4,5", help="Phases to run, comma-separated")
    args = parser.parse_args()

    phases_to_run = {int(p.strip()) for p in args.phase.split(",")}

    logger.info("=" * 60)
    logger.info("V10 EXPERIMENT — MLB Game Prediction")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Phases:  {sorted(phases_to_run)}")
    logger.info("=" * 60)

    logger.info("\nLoading V10 feature splits...")
    splits = load_splits()

    all_results: dict = {
        "experiment": "v10",
        "timestamp": datetime.now().isoformat(),
        "v9_baseline": {
            "val_acc": 0.5733, "val_auc": 0.5884, "val_brier": 0.2422,
            "live_2026_acc": 0.5866, "live_2026_auc": 0.6135,
            "description": "XGBoost_Optuna from V9 experiment (2025 val set)",
        },
    }

    if 1 in phases_to_run:
        t0 = time.time()
        res = phase1_ablation(splits)
        all_results["phase1_ablation"] = res
        logger.info(f"  Phase 1 done in {time.time()-t0:.0f}s")

    if 2 in phases_to_run:
        t0 = time.time()
        res = phase2_v9_vs_v10(splits, quick=args.quick)
        all_results["phase2_v9_vs_v10"] = res
        logger.info(f"  Phase 2 done in {time.time()-t0:.0f}s")

    if 3 in phases_to_run:
        t0 = time.time()
        res = phase3_wf_cv(splits)
        all_results["phase3_wf_cv"] = res
        logger.info(f"  Phase 3 done in {time.time()-t0:.0f}s")

    if 4 in phases_to_run:
        t0 = time.time()
        res = phase4_confidence_curve(splits)
        # Convert for JSON serialization
        serializable = {}
        for k, curve in res.items():
            serializable[k] = [
                {
                    "threshold": pt["threshold"],
                    "accuracy": float(pt["accuracy"]) if not np.isnan(pt["accuracy"]) else None,
                    "coverage": float(pt["coverage"]),
                }
                for pt in curve
            ]
        all_results["phase4_confidence_curve"] = serializable
        logger.info(f"  Phase 4 done in {time.time()-t0:.0f}s")

    if 5 in phases_to_run:
        t0 = time.time()
        res = phase5_2026_test(splits, quick=args.quick)
        all_results["phase5_2026_test"] = res
        logger.info(f"  Phase 5 done in {time.time()-t0:.0f}s")

    # Save results
    results_path = LOG_DIR / "v10_experiment_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        # Make JSON-serializable
        def make_serial(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return None if np.isnan(obj) else float(obj)
            if isinstance(obj, dict):
                return {k: make_serial(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serial(v) for v in obj]
            return obj

        json.dump(make_serial(all_results), f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("V10 EXPERIMENT COMPLETE — SUMMARY")
    logger.info("=" * 60)

    if "phase2_v9_vs_v10" in all_results:
        for r in all_results["phase2_v9_vs_v10"]:
            label = r.get("label", "")
            acc   = r.get("accuracy")
            auc   = r.get("auc")
            if acc:
                logger.info(f"  {label}: val_acc={acc:.4f}  auc={auc:.4f}")

    if "phase3_wf_cv" in all_results:
        for r in all_results["phase3_wf_cv"]:
            m, s = r.get("wf_cv_mean"), r.get("wf_cv_std")
            if m:
                logger.info(f"  {r['label']}: WF-CV {m:.4f} +/- {s:.4f}")

    if "phase5_2026_test" in all_results:
        logger.info("  2026 Live Test:")
        for r in all_results["phase5_2026_test"]:
            label = r.get("label", "")
            acc   = r.get("accuracy")
            if acc:
                logger.info(f"    {label}: acc={acc:.4f}")

    v9_baseline_acc = 0.5733
    if "phase2_v9_vs_v10" in all_results:
        v10_res = next((r for r in all_results["phase2_v9_vs_v10"] if "V10" in r.get("label", "")), None)
        if v10_res and v10_res.get("accuracy"):
            delta = v10_res["accuracy"] - v9_baseline_acc
            sign = "+" if delta >= 0 else ""
            logger.info(f"\n  Net improvement V10 vs V9 baseline: {sign}{delta*100:.2f}%")

    logger.info("\nNext: Update research/v10_experiment/README.md with results")


if __name__ == "__main__":
    main()
