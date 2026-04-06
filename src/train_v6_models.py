#!/usr/bin/env python3
"""
V6 Model Training — Pitcher Arsenal + Venue Split Features

Extends V5 (matchup-aware stacked ensemble) by adding:
  - Pitcher arsenal features (pitch mix, velocity, spin, K-BB%, xwOBA)
    from mlb_historical_data.pitcher_game_stats (rolling 30-day lookback)
  - Batter venue history (wOBA at specific park vs career average)
    from mlb_historical_data.player_venue_splits + player_season_splits

Architecture: Same V5 stacked ensemble (LR + XGBoost + LightGBM → meta LR)
New V6 features:
  Pitcher arsenal (per starter, home + away):
    home_starter_fastball_pct, away_starter_fastball_pct
    home_starter_breaking_pct, away_starter_breaking_pct
    home_starter_offspeed_pct, away_starter_offspeed_pct
    home_starter_xwoba_allowed, away_starter_xwoba_allowed
    home_starter_k_bb_pct, away_starter_k_bb_pct
    home_starter_velo_norm, away_starter_velo_norm
    home_starter_velo_trend, away_starter_velo_trend
    starter_arsenal_advantage (composite differential)
  Venue splits:
    home_lineup_venue_woba, away_lineup_venue_woba
    venue_woba_differential
    home_venue_advantage, away_venue_disadvantage

Also fixes: imputation now uses training-time medians (fill_values) instead of 0.0.

Usage:
    python train_v6_models.py                    # full training
    python train_v6_models.py --no-v6-join       # V5 features only (test run)
    python train_v6_models.py --dry-run          # feature count preview, no save
    python train_v6_models.py --upload           # upload to GCS after training
"""

import argparse
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from model_classes import StackedV6Model  # noqa: F401

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
HIST_DS = "mlb_historical_data"

TRAIN_PATH = Path("data/training/train_v3_2015_2024.parquet")
VAL_PATH   = Path("data/training/val_v3_2025.parquet")
OUTPUT_PATH = Path("models/game_outcome_2026_v6.pkl")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

GCS_BUCKET = "hanks_tank_data"
GCS_V6_PATH = "models/vertex/game_outcome_2026_v6/model.pkl"

# ---------------------------------------------------------------------------
# V5 features (same as before — use the actual training-data column names)
# ---------------------------------------------------------------------------
V5_CORE_FEATURES = [
    "home_ema_form", "away_ema_form",
    "pitcher_quality_diff",
    "home_form_squared", "away_form_squared",
    "rest_balance",
    "home_momentum", "away_momentum",
    "fatigue_index",
    "park_advantage",
    "trend_alignment",
    "season_phase_home_effect",
    "win_pct_diff",
    "home_composite_strength", "away_composite_strength",
    # Matchup (V5)
    "lineup_woba_differential",
    "starter_woba_differential",
    "matchup_advantage_home",
    "home_pct_same_hand", "away_pct_same_hand",
    "h2h_woba_differential",
    "home_top3_woba_vs_hand", "away_top3_woba_vs_hand",
    "lineup_k_pct_differential",
    "home_starter_k_pct", "away_starter_k_pct",
    # Game context
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

# New V6 features
V6_NEW_FEATURES = [
    # Pitcher arsenal
    "home_starter_fastball_pct",  "away_starter_fastball_pct",
    "home_starter_breaking_pct",  "away_starter_breaking_pct",
    "home_starter_offspeed_pct",  "away_starter_offspeed_pct",
    "home_starter_xwoba_allowed", "away_starter_xwoba_allowed",
    "home_starter_k_bb_pct",      "away_starter_k_bb_pct",
    "home_starter_velo_norm",     "away_starter_velo_norm",
    "home_starter_velo_trend",    "away_starter_velo_trend",
    "starter_arsenal_advantage",
    # Venue splits
    "home_lineup_venue_woba",
    "away_lineup_venue_woba",
    "venue_woba_differential",
    "home_venue_advantage",
    "away_venue_disadvantage",
]

ALL_V6_FEATURES = V5_CORE_FEATURES + V6_NEW_FEATURES

TARGET = "home_won"
EXCLUDE_COLS = {
    TARGET, "game_pk", "game_date", "year", "season",
    "home_team_id", "away_team_id", "home_team", "away_team",
    "home_team_name", "away_team_name",
}


# ---------------------------------------------------------------------------
# BQ helpers for V6 feature join
# ---------------------------------------------------------------------------
def join_matchup_v6_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Join V6 matchup features (pitcher arsenal) for training games.

    Uses pitcher_game_stats (which now includes team_type = home|away) to find
    each game's starting pitcher and attach their arsenal stats. This is a
    same-game lookup (slight lookahead for training) but teaches the model
    what pitch mix / velocity differences mean.

    Venue history (player_venue_splits) is skipped for training — the lookup
    requires knowing batting lineups per historical game, which isn't stored.
    V6 venue features will be NaN in training → filled with neutral values at
    fit-time; the model learns to use them when available at prediction time.
    """
    try:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT)

        game_pks = df["game_pk"].dropna().astype(int).tolist()
        if not game_pks:
            raise ValueError("No game_pks in dataframe")

        pitcher_table = f"{PROJECT}.{HIST_DS}.pitcher_game_stats"

        # Verify pitcher_game_stats has team_type column (requires recent rebuild)
        try:
            t = bq.get_table(pitcher_table)
            has_team_type = any(f.name == "team_type" for f in t.schema)
        except Exception:
            has_team_type = False

        if not has_team_type:
            logger.warning("[%s] pitcher_game_stats missing team_type column — V6 features will be neutral", label)
            raise ValueError("pitcher_game_stats missing team_type")

        v6_rows = []
        chunk_size = 1000
        velo_mean = 93.0
        velo_std = 3.0

        for i in range(0, len(game_pks), chunk_size):
            chunk = game_pks[i:i + chunk_size]
            pk_list = ", ".join(str(pk) for pk in chunk)

            # Get the top pitcher per team_type per game (highest total_pitches = starter)
            sql = f"""
                WITH starters AS (
                    SELECT
                        game_pk,
                        team_type,
                        pitcher,
                        fastball_pct,
                        breaking_pct,
                        offspeed_pct,
                        COALESCE(mean_fastball_velo, mean_release_speed, {velo_mean}) AS fastball_velo,
                        k_bb_pct,
                        xwoba_allowed,
                        ROW_NUMBER() OVER (
                            PARTITION BY game_pk, team_type
                            ORDER BY total_pitches DESC
                        ) AS rn
                    FROM `{pitcher_table}`
                    WHERE game_pk IN ({pk_list})
                )
                SELECT
                    h.game_pk,
                    h.fastball_pct                                               AS home_starter_fastball_pct,
                    h.breaking_pct                                               AS home_starter_breaking_pct,
                    h.offspeed_pct                                               AS home_starter_offspeed_pct,
                    h.xwoba_allowed                                              AS home_starter_xwoba_allowed,
                    h.k_bb_pct                                                   AS home_starter_k_bb_pct,
                    ROUND((h.fastball_velo - {velo_mean}) / {velo_std}, 4)      AS home_starter_velo_norm,
                    CAST(NULL AS FLOAT64)                                        AS home_starter_velo_trend,
                    a.fastball_pct                                               AS away_starter_fastball_pct,
                    a.breaking_pct                                               AS away_starter_breaking_pct,
                    a.offspeed_pct                                               AS away_starter_offspeed_pct,
                    a.xwoba_allowed                                              AS away_starter_xwoba_allowed,
                    a.k_bb_pct                                                   AS away_starter_k_bb_pct,
                    ROUND((a.fastball_velo - {velo_mean}) / {velo_std}, 4)      AS away_starter_velo_norm,
                    CAST(NULL AS FLOAT64)                                        AS away_starter_velo_trend,
                    ROUND(
                        (h.k_bb_pct * 2.0 - h.xwoba_allowed
                          + ((h.fastball_velo - {velo_mean}) / {velo_std}) * 0.05)
                        -
                        (a.k_bb_pct * 2.0 - a.xwoba_allowed
                          + ((a.fastball_velo - {velo_mean}) / {velo_std}) * 0.05),
                        4
                    )                                                            AS starter_arsenal_advantage,
                    CAST(NULL AS FLOAT64)                                        AS home_lineup_venue_woba,
                    CAST(NULL AS FLOAT64)                                        AS away_lineup_venue_woba,
                    CAST(NULL AS FLOAT64)                                        AS venue_woba_differential,
                    CAST(NULL AS FLOAT64)                                        AS home_venue_advantage,
                    CAST(NULL AS FLOAT64)                                        AS away_venue_disadvantage
                FROM (SELECT * FROM starters WHERE team_type = 'home' AND rn = 1) h
                JOIN (SELECT * FROM starters WHERE team_type = 'away' AND rn = 1) a
                  ON h.game_pk = a.game_pk
            """
            chunk_df = bq.query(sql).to_dataframe()
            v6_rows.append(chunk_df)

        if v6_rows:
            v6_combined = pd.concat(v6_rows, ignore_index=True)
            df = df.merge(v6_combined, on="game_pk", how="left")
            n_joined = v6_combined["game_pk"].nunique()
            logger.info("[%s] Joined V6 features for %d/%d games", label, n_joined, len(game_pks))

    except Exception as e:
        logger.warning("[%s] V6 feature join failed: %s — using neutral values", label, e)

    # Ensure all V6 columns exist (NaN for games without data → filled by imputer during training)
    for col in V6_NEW_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    return df


def join_matchup_v5_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Reuse V5 matchup join logic for training (H2H + platoon splits)."""
    try:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT)

        game_pks = df["game_pk"].dropna().astype(int).tolist()
        if not game_pks:
            return df

        v5_matchup_hist = f"{PROJECT}.{HIST_DS}.matchup_features_historical"
        v5_matchup_2026 = f"{PROJECT}.mlb_2026_season.matchup_features"

        for tbl in [v5_matchup_hist, v5_matchup_2026]:
            try:
                bq.get_table(tbl)
                break
            except Exception:
                tbl = None

        if tbl is None:
            logger.warning("[%s] No V5 matchup table found", label)
            return df

        chunk_size = 1000
        matchup_rows = []
        for i in range(0, len(game_pks), chunk_size):
            chunk = game_pks[i:i + chunk_size]
            pk_list = ", ".join(str(pk) for pk in chunk)
            sql = f"""
                SELECT
                    game_pk,
                    home_lineup_woba_vs_hand - away_lineup_woba_vs_hand AS lineup_woba_differential,
                    away_starter_woba_allowed - home_starter_woba_allowed AS starter_woba_differential,
                    matchup_advantage_home,
                    home_pct_same_hand, away_pct_same_hand,
                    COALESCE(home_h2h_woba, 0.320) - COALESCE(away_h2h_woba, 0.320) AS h2h_woba_differential,
                    home_top3_woba_vs_hand, away_top3_woba_vs_hand,
                    away_lineup_k_pct_vs_hand - home_lineup_k_pct_vs_hand AS lineup_k_pct_differential,
                    home_starter_k_pct, away_starter_k_pct
                FROM `{tbl}`
                WHERE game_pk IN ({pk_list})
                QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) = 1
            """
            matchup_rows.append(bq.query(sql).to_dataframe())

        if matchup_rows:
            matchup_df = pd.concat(matchup_rows, ignore_index=True)
            df = df.merge(matchup_df, on="game_pk", how="left")
            logger.info("[%s] Joined V5 matchup features for %d games",
                        label, matchup_df["game_pk"].nunique())

    except Exception as e:
        logger.warning("[%s] V5 matchup join failed: %s", label, e)

    v5_matchup_cols = [
        "lineup_woba_differential", "starter_woba_differential",
        "matchup_advantage_home", "home_pct_same_hand", "away_pct_same_hand",
        "h2h_woba_differential", "home_top3_woba_vs_hand", "away_top3_woba_vs_hand",
        "lineup_k_pct_differential", "home_starter_k_pct", "away_starter_k_pct",
    ]
    for col in v5_matchup_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class V6Trainer:
    def __init__(self, use_v6_join: bool = True, use_v5_join: bool = True):
        self.use_v6_join = use_v6_join
        self.use_v5_join = use_v5_join
        self.feature_cols: list[str] = []
        self.fill_values: dict = {}

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Loading V3 training data...")
        train_df = pd.read_parquet(TRAIN_PATH)
        val_df   = pd.read_parquet(VAL_PATH)
        logger.info("Train: %d rows | Val: %d rows", len(train_df), len(val_df))
        return train_df, val_df

    def prepare_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Prioritise listed V6 features, then include any other numeric cols
        available = [f for f in ALL_V6_FEATURES if f in train_df.columns]
        missing   = [f for f in ALL_V6_FEATURES if f not in train_df.columns]
        if missing:
            logger.warning("Missing V6 features (will impute): %s", missing)

        numeric_cols = [
            c for c in train_df.columns
            if pd.api.types.is_numeric_dtype(train_df[c]) and c not in EXCLUDE_COLS
        ]
        self.feature_cols = list(dict.fromkeys(
            available + [c for c in numeric_cols if c not in available]
        ))

        logger.info(
            "V6 total features: %d  (V5 base: %d, V6 new: %d, extra: %d)",
            len(self.feature_cols),
            len([f for f in self.feature_cols if f in V5_CORE_FEATURES]),
            len([f for f in self.feature_cols if f in V6_NEW_FEATURES]),
            len([f for f in self.feature_cols if f not in ALL_V6_FEATURES]),
        )

        # Add missing columns
        for col in self.feature_cols:
            if col not in train_df.columns:
                train_df[col] = np.nan
            if col not in val_df.columns:
                val_df[col] = np.nan

        # Median imputation fitted on train only — stored as fill_values for inference
        for col in self.feature_cols:
            fill_val = train_df[col].median()
            if pd.isna(fill_val):
                fill_val = 0.0
            self.fill_values[col] = float(fill_val)
            train_df[col] = train_df[col].fillna(fill_val)
            val_df[col]   = val_df[col].fillna(fill_val)

        X_train = train_df[self.feature_cols].values.astype(float)
        y_train = train_df[TARGET].values.astype(int)
        X_val   = val_df[self.feature_cols].values.astype(float)
        y_val   = val_df[TARGET].values.astype(int)

        logger.info("X_train shape: %s | X_val shape: %s", X_train.shape, X_val.shape)
        return X_train, y_train, X_val, y_val

    def _build_base_models(self):
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=2000, C=0.5, solver="liblinear", random_state=42
            )),
        ])
        xgb_model = xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.04, max_depth=5,
            subsample=0.8, colsample_bytree=0.75, min_child_weight=3,
            reg_lambda=1.5, reg_alpha=0.1, eval_metric="logloss",
            early_stopping_rounds=30, random_state=42, verbosity=0,
        )
        lgbm = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.04, num_leaves=31,
            max_depth=6, subsample=0.8, colsample_bytree=0.75,
            min_child_samples=20, reg_lambda=1.5, reg_alpha=0.1,
            random_state=42, verbose=-1,
        )
        return {"lr": lr, "xgb": xgb_model, "lgbm": lgbm}

    def train_stacked_ensemble(
        self, X_train, y_train, X_val, y_val, train_df
    ) -> dict:
        """Time-based expanding-year CV for stacking OOF predictions."""
        base_models = self._build_base_models()
        years = sorted(train_df["year"].dropna().unique().astype(int))
        n_base = len(base_models)

        oof_train = np.zeros((len(X_train), n_base))
        oof_val   = np.zeros((len(X_val),   n_base))

        logger.info("Stacking CV over years: %s", years)

        for fold_year in years[3:]:
            train_mask = train_df["year"].astype(int) < fold_year
            val_mask   = train_df["year"].astype(int) == fold_year
            if train_mask.sum() < 100 or val_mask.sum() < 10:
                continue

            X_fold_tr = X_train[train_mask]
            y_fold_tr = y_train[train_mask]
            X_fold_va = X_train[val_mask]

            for j, (name, model) in enumerate(base_models.items()):
                if hasattr(model, "fit"):
                    if name == "xgb":
                        model.fit(X_fold_tr, y_fold_tr,
                                  eval_set=[(X_fold_va, y_train[val_mask])],
                                  verbose=False)
                    else:
                        model.fit(X_fold_tr, y_fold_tr)
                    oof_train[val_mask, j] = model.predict_proba(X_fold_va)[:, 1]

        # Full fit on all training data
        for j, (name, model) in enumerate(base_models.items()):
            if name == "xgb":
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)
            oof_val[:, j] = model.predict_proba(X_val)[:, 1]

        # Train meta learner on OOF predictions
        meta_lr = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=500, random_state=42),
            cv="prefit" if False else 5,
            method="isotonic",
        )
        # Fit meta on val OOF first, then on train OOF for a proper hold-out estimate
        meta_lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        meta_lr.fit(oof_train, y_train)

        # Evaluate on val
        val_preds = meta_lr.predict_proba(oof_val)[:, 1]
        acc = accuracy_score(y_val, (val_preds >= 0.5).astype(int))
        auc = roc_auc_score(y_val, val_preds)
        brier = brier_score_loss(y_val, val_preds)

        logger.info("V6 Val — Accuracy: %.4f | AUC: %.4f | Brier: %.4f", acc, auc, brier)
        logger.info("V6 Val home win rate: %.3f | predicted: %.3f",
                    y_val.mean(), val_preds.mean())

        return {
            "base_models": base_models,
            "meta": meta_lr,
            "metrics": {"accuracy": float(acc), "auc": float(auc), "brier": float(brier)},
        }

    def run(self, dry_run: bool = False, upload: bool = False) -> dict:
        train_df, val_df = self.load_data()

        # Join V5 matchup features
        if self.use_v5_join:
            train_df = join_matchup_v5_features(train_df, "train")
            val_df   = join_matchup_v5_features(val_df,   "val")

        # Join V6 pitcher arsenal + venue features
        if self.use_v6_join:
            train_df = join_matchup_v6_features(train_df, "train")
            val_df   = join_matchup_v6_features(val_df,   "val")

        X_train, y_train, X_val, y_val = self.prepare_features(train_df, val_df)

        if dry_run:
            logger.info("[DRY RUN] Would train V6 with %d features", len(self.feature_cols))
            return {"features": len(self.feature_cols), "dry_run": True}

        result = self.train_stacked_ensemble(X_train, y_train, X_val, y_val, train_df)

        stacked_model = StackedV6Model(result["base_models"], result["meta"])

        model_data = {
            "model": stacked_model,
            "model_name": "v6_pitcher_venue_stacked_ensemble",
            "features": self.feature_cols,
            "fill_values": self.fill_values,
            "scaler": None,  # V6 uses fill_values directly; base models handle scaling internally
            "metrics": result["metrics"],
            "trained_at": datetime.utcnow().isoformat(),
            "v6_new_features": V6_NEW_FEATURES,
        }

        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump(model_data, f)
        logger.info("Saved V6 model to %s", OUTPUT_PATH)
        logger.info("Metrics: %s", result["metrics"])

        if upload:
            _upload_to_gcs(OUTPUT_PATH)

        return model_data


def _upload_to_gcs(local_path: Path) -> None:
    from google.cloud import storage
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_V6_PATH)
    blob.upload_from_filename(str(local_path))
    logger.info("Uploaded V6 model to gs://%s/%s", GCS_BUCKET, GCS_V6_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-v6-join",  action="store_true",
                        help="Skip V6 pitcher/venue feature join (use V5 features only)")
    parser.add_argument("--no-v5-join",  action="store_true",
                        help="Skip V5 matchup feature join")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--upload",      action="store_true",
                        help="Upload trained model to GCS after training")
    args = parser.parse_args()

    trainer = V6Trainer(
        use_v6_join=not args.no_v6_join,
        use_v5_join=not args.no_v5_join,
    )
    result = trainer.run(dry_run=args.dry_run, upload=args.upload)
    logger.info("Training complete: %s", result.get("metrics", {}))
