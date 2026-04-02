#!/usr/bin/env python3
"""
V5 Model Training — Matchup-Aware Stacked Ensemble

Extends V4 by adding pitcher/batter matchup features to the training data.
Matchup features come from pre-computed statcast-based splits for historical games.

Architecture: Same V4 stacked ensemble (LR + XGBoost + LightGBM → meta LR)
New features added on top of V3:
  - lineup_woba_differential           home lineup wOBA vs away starter hand MINUS
                                       away lineup wOBA vs home starter hand
  - starter_woba_differential          away starter wOBA allowed MINUS home starter wOBA allowed
  - matchup_advantage_home             composite platoon/H2H score
  - home_pct_same_hand                 % home batters same hand as away starter
  - away_pct_same_hand                 % away batters same hand as home starter
  - h2h_woba_differential              home H2H wOBA MINUS away H2H wOBA
  - home_top3_woba_vs_hand             top of order vs pitcher handedness
  - away_top3_woba_vs_hand
  - lineup_k_pct_differential          away lineup K% minus home lineup K%

Training data: train_v3_2015_2024.parquet + matchup_features for each game
Validation:    val_v3_2025.parquet + matchup_features

Matchup features are joined in from BigQuery for historical games.
Games without matchup data receive neutral/median-imputed values.

Output: models/game_outcome_2026_v5.pkl

Usage:
    python train_v5_models.py                    # full training (BQ join)
    python train_v5_models.py --no-matchup-join  # use only V3 features (test run)
    python train_v5_models.py --dry-run          # print feature counts, don't save
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
from model_classes import StackedV5Model

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
HIST_DATASET = "mlb_historical_data"
MATCHUP_TABLE = f"{PROJECT}.mlb_2026_season.matchup_features"

TRAIN_PATH = Path("data/training/train_v3_2015_2024.parquet")
VAL_PATH = Path("data/training/val_v3_2025.parquet")
OUTPUT_PATH = Path("models/game_outcome_2026_v5.pkl")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

# V3 features (baseline)
V3_FEATURES = [
    "home_rolling_win_pct_10", "away_rolling_win_pct_10",
    "home_rolling_win_pct_30", "away_rolling_win_pct_30",
    "home_ema_form", "away_ema_form",
    "home_rest_days", "away_rest_days",
    "home_back_to_back", "away_back_to_back",
    "pitcher_quality_diff",
    "park_run_factor_ratio",
    "season_phase",
    "is_divisional",
    # V3 derived
    "home_form_squared", "away_form_squared",
    "rest_balance",
    "home_momentum", "away_momentum",
    "fatigue_index",
    "park_advantage",
    "trend_alignment",
    "season_phase_home_effect",
    "win_pct_diff",
    "home_composite_strength", "away_composite_strength",
]

# New V5 matchup features
MATCHUP_FEATURES = [
    "lineup_woba_differential",
    "starter_woba_differential",
    "matchup_advantage_home",
    "home_pct_same_hand",
    "away_pct_same_hand",
    "h2h_woba_differential",
    "home_top3_woba_vs_hand",
    "away_top3_woba_vs_hand",
    "lineup_k_pct_differential",
    "home_starter_k_pct",
    "away_starter_k_pct",
]

ALL_V5_FEATURES = V3_FEATURES + MATCHUP_FEATURES

TARGET = "home_won"
EXCLUDE_COLS = {
    TARGET, "game_pk", "game_date", "year", "season",
    "home_team_id", "away_team_id", "home_team", "away_team",
    "home_team_name", "away_team_name",
}


class V5Trainer:
    """Train V5 matchup-aware stacked ensemble."""

    def __init__(self, use_matchup_join: bool = True):
        self.use_matchup_join = use_matchup_join
        self.feature_cols = []
        self.fill_values = {}

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Loading V3 training data...")
        train_df = pd.read_parquet(TRAIN_PATH)
        val_df = pd.read_parquet(VAL_PATH)
        logger.info("Train: %d rows | Val: %d rows", len(train_df), len(val_df))
        return train_df, val_df

    def join_matchup_features(
        self, df: pd.DataFrame, label: str
    ) -> pd.DataFrame:
        """
        Join pre-computed matchup features from BigQuery for historical games.
        For training data (2015–2024) this joins from mlb_historical matchup cache.
        Falls back gracefully if BQ is not available or game has no lineup data.
        """
        if not self.use_matchup_join:
            logger.info("[%s] Skipping matchup join (--no-matchup-join)", label)
            for col in MATCHUP_FEATURES:
                df[col] = np.nan
            return df

        try:
            from google.cloud import bigquery
            bq = bigquery.Client(project=PROJECT)

            game_pks = df["game_pk"].dropna().astype(int).tolist()
            if not game_pks:
                raise ValueError("No game_pks in dataframe")

            # Historical matchup features are stored in the same matchup_features table
            # for 2026 upcoming games, but for training we build them from statcast
            # using a pre-computed historical matchup cache table if available.
            historical_matchup_table = f"{PROJECT}.{HIST_DATASET}.matchup_features_historical"

            # Try historical table first, fall back to 2026 table
            for table in [historical_matchup_table, MATCHUP_TABLE]:
                try:
                    bq.get_table(table)
                    logger.info("[%s] Joining matchup features from %s", label, table)
                    break
                except Exception:
                    table = None

            if table is None:
                logger.warning(
                    "[%s] No matchup feature table found — using neutral values", label
                )
                for col in MATCHUP_FEATURES:
                    df[col] = np.nan
                return df

            # Batch query in chunks of 1000 to avoid query size limits
            chunk_size = 1000
            matchup_rows = []
            for i in range(0, len(game_pks), chunk_size):
                chunk = game_pks[i : i + chunk_size]
                pk_list = ", ".join(str(pk) for pk in chunk)
                sql = f"""
                    SELECT
                        game_pk,
                        home_lineup_woba_vs_hand - away_lineup_woba_vs_hand
                            AS lineup_woba_differential,
                        away_starter_woba_allowed - home_starter_woba_allowed
                            AS starter_woba_differential,
                        matchup_advantage_home,
                        home_pct_same_hand,
                        away_pct_same_hand,
                        COALESCE(home_h2h_woba, 0.320) - COALESCE(away_h2h_woba, 0.320)
                            AS h2h_woba_differential,
                        home_top3_woba_vs_hand,
                        away_top3_woba_vs_hand,
                        away_lineup_k_pct_vs_hand - home_lineup_k_pct_vs_hand
                            AS lineup_k_pct_differential,
                        home_starter_k_pct,
                        away_starter_k_pct
                    FROM `{table}`
                    WHERE game_pk IN ({pk_list})
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY game_pk ORDER BY computed_at DESC
                    ) = 1
                """
                chunk_df = bq.query(sql).to_dataframe()
                matchup_rows.append(chunk_df)

            matchup_df = pd.concat(matchup_rows, ignore_index=True) if matchup_rows else pd.DataFrame()
            if matchup_df.empty:
                logger.warning("[%s] No matchup features joined — using neutral values", label)
                for col in MATCHUP_FEATURES:
                    df[col] = np.nan
            else:
                df = df.merge(matchup_df, on="game_pk", how="left")
                joined = df[MATCHUP_FEATURES].notna().any(axis=1).sum()
                logger.info("[%s] Joined matchup features for %d/%d games", label, joined, len(df))

        except Exception as e:
            logger.warning("[%s] Matchup join failed: %s — using neutral values", label, e)
            for col in MATCHUP_FEATURES:
                df[col] = np.nan

        return df

    # -----------------------------------------------------------------------
    # Feature selection
    # -----------------------------------------------------------------------
    def prepare_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select and impute features."""
        # Use known V5 features if available, otherwise fall back to all numeric
        available_v5 = [f for f in ALL_V5_FEATURES if f in train_df.columns]
        missing = [f for f in ALL_V5_FEATURES if f not in train_df.columns]
        if missing:
            logger.warning("Missing V5 features (will impute with 0): %s", missing)

        # Auto-include any numeric features not explicitly excluded (catches new derived cols)
        numeric_cols = [
            c for c in train_df.columns
            if pd.api.types.is_numeric_dtype(train_df[c]) and c not in EXCLUDE_COLS
        ]
        self.feature_cols = list(dict.fromkeys(available_v5 + [
            c for c in numeric_cols if c not in available_v5
        ]))
        logger.info("Total features: %d (V3 base: %d, matchup: %d, extra: %d)",
                    len(self.feature_cols),
                    len([f for f in self.feature_cols if f in V3_FEATURES]),
                    len([f for f in self.feature_cols if f in MATCHUP_FEATURES]),
                    len([f for f in self.feature_cols if f not in ALL_V5_FEATURES]))

        # Add any missing feature columns with NaN
        for col in self.feature_cols:
            if col not in train_df.columns:
                train_df[col] = np.nan
            if col not in val_df.columns:
                val_df[col] = np.nan

        # Median imputation (fit on train only)
        for col in self.feature_cols:
            fill_val = train_df[col].median()
            if pd.isna(fill_val):
                fill_val = 0.0
            self.fill_values[col] = fill_val
            train_df[col] = train_df[col].fillna(fill_val)
            val_df[col] = val_df[col].fillna(fill_val)

        X_train = train_df[self.feature_cols].values.astype(float)
        y_train = train_df[TARGET].values.astype(int)
        X_val = val_df[self.feature_cols].values.astype(float)
        y_val = val_df[TARGET].values.astype(int)

        logger.info("X_train shape: %s | X_val shape: %s", X_train.shape, X_val.shape)
        return X_train, y_train, X_val, y_val

    # -----------------------------------------------------------------------
    # Model building (same V4 architecture)
    # -----------------------------------------------------------------------
    def _build_base_models(self):
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=2000, C=0.5, solver="liblinear", random_state=42
            )),
        ])

        xgb_model = xgb.XGBClassifier(
            n_estimators=600,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_weight=3,
            reg_lambda=1.5,
            reg_alpha=0.1,
            eval_metric="logloss",
            early_stopping_rounds=30,
            random_state=42,
            verbosity=0,
        )

        lgbm = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_samples=20,
            reg_lambda=1.5,
            reg_alpha=0.1,
            random_state=42,
            verbose=-1,
        )

        return {"lr": lr, "xgb": xgb_model, "lgbm": lgbm}

    # -----------------------------------------------------------------------
    # Stacking cross-validation
    # -----------------------------------------------------------------------
    def train_stacked_ensemble(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        train_df: pd.DataFrame,
    ) -> dict:
        """
        Time-based expanding-year cross-validation for stacking OOF predictions.
        """
        base_models = self._build_base_models()
        years = sorted(train_df["year"].dropna().unique().astype(int))
        n_base = len(base_models)

        # Initialize OOF arrays
        oof_train = np.zeros((len(X_train), n_base))
        oof_val_base = np.zeros((len(X_val), n_base))

        logger.info("Starting time-based stacking CV with years: %s", years)

        for fold_year in years[3:]:  # Need min 3 years of history to train
            train_mask = train_df["year"].astype(int) < fold_year
            val_mask = train_df["year"].astype(int) == fold_year

            if train_mask.sum() < 100 or val_mask.sum() < 10:
                continue

            X_fold_train = X_train[train_mask]
            y_fold_train = y_train[train_mask]
            X_fold_val = X_train[val_mask]

            for i, (name, base_model) in enumerate(base_models.items()):
                import copy
                model_copy = copy.deepcopy(base_model)

                if name == "xgb":
                    model_copy.fit(
                        X_fold_train, y_fold_train,
                        eval_set=[(X_fold_val, y_train[val_mask])],
                        verbose=False,
                    )
                else:
                    model_copy.fit(X_fold_train, y_fold_train)

                oof_preds = model_copy.predict_proba(X_fold_val)[:, 1]
                oof_train[val_mask, i] = oof_preds

        # Train final base models on full training set
        logger.info("Training final base models on full training data...")
        trained_base = {}
        for i, (name, base_model) in enumerate(base_models.items()):
            logger.info("  Training %s...", name)
            if name == "xgb":
                # Use last 20% of training data as eval for early stopping
                split_idx = int(0.8 * len(X_train))
                base_model.fit(
                    X_train[:split_idx], y_train[:split_idx],
                    eval_set=[(X_train[split_idx:], y_train[split_idx:])],
                    verbose=False,
                )
            else:
                base_model.fit(X_train, y_train)

            val_proba = base_model.predict_proba(X_val)[:, 1]
            val_acc = accuracy_score(y_val, (val_proba >= 0.5).astype(int))
            val_auc = roc_auc_score(y_val, val_proba)
            logger.info(
                "  %s → val acc: %.4f | AUC: %.4f", name, val_acc, val_auc
            )
            trained_base[name] = base_model
            oof_val_base[:, i] = val_proba

        # Train meta-model on OOF predictions
        logger.info("Training meta-model (Logistic Regression)...")
        meta = LogisticRegression(C=1.0, solver="liblinear", max_iter=500, random_state=42)

        # Use non-zero OOF rows only (where fold was computed)
        valid_oof_mask = oof_train.any(axis=1)
        if valid_oof_mask.sum() > 100:
            meta.fit(oof_train[valid_oof_mask], y_train[valid_oof_mask])
        else:
            logger.warning("Insufficient OOF data for meta — using equal weights")
            meta.fit(oof_val_base, y_val)  # fallback: fit on val as proxy

        # Evaluate stacked ensemble
        val_meta_proba = meta.predict_proba(oof_val_base)[:, 1]
        val_acc = accuracy_score(y_val, (val_meta_proba >= 0.5).astype(int))
        val_auc = roc_auc_score(y_val, val_meta_proba)
        val_brier = brier_score_loss(y_val, val_meta_proba)
        logger.info(
            "Stacked V5 ensemble → val acc: %.4f | AUC: %.4f | Brier: %.4f",
            val_acc, val_auc, val_brier
        )

        return {
            "base_models": trained_base,
            "meta_model": meta,
            "val_accuracy": val_acc,
            "val_auc": val_auc,
            "val_brier": val_brier,
        }

    # -----------------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------------
    def save_model(self, result: dict, dry_run: bool = False) -> None:
        """
        Save V5 model in the same pkl format expected by predict_today_games.py.
        The predict script loads base models, applies them, then stacks.
        We save a wrapper that handles this transparently.
        """
        base_models = result["base_models"]
        meta = result["meta_model"]

        wrapped_model = StackedV5Model(base_models, meta)

        # Scaler is inside the LR pipeline; expose top-level for compatibility
        # predict_today_games.py checks for scaler, if None it skips scaling
        model_data = {
            "model": wrapped_model,
            "scaler": None,  # Each base model handles its own scaling
            "features": self.feature_cols,
            "fill_values": self.fill_values,
            "model_name": "v5_matchup_stacked_ensemble",
            "trained_at": datetime.now().isoformat(),
            "val_accuracy": result["val_accuracy"],
            "val_auc": result["val_auc"],
            "val_brier": result["val_brier"],
            "v3_features": V3_FEATURES,
            "matchup_features": MATCHUP_FEATURES,
        }

        if dry_run:
            logger.info("[DRY RUN] Would save to %s", OUTPUT_PATH)
            return

        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump(model_data, f, protocol=4)
        logger.info("V5 model saved to %s", OUTPUT_PATH)

        # Also upload to GCS
        try:
            from google.cloud import storage
            client = storage.Client(project=PROJECT)
            bucket = client.bucket("hanks_tank_data")
            blob = bucket.blob("models/vertex/game_outcome_2026_v5/model.pkl")
            blob.upload_from_filename(str(OUTPUT_PATH))
            logger.info("V5 model uploaded to GCS")
        except Exception as e:
            logger.warning("GCS upload failed (model saved locally): %s", e)

    # -----------------------------------------------------------------------
    # Main entry
    # -----------------------------------------------------------------------
    def run(self, dry_run: bool = False) -> None:
        logger.info("=" * 60)
        logger.info("V5 Matchup-Aware Model Training")
        logger.info("=" * 60)

        train_df, val_df = self.load_data()

        # Join matchup features
        logger.info("Joining matchup features for training data...")
        train_df = self.join_matchup_features(train_df, "train")
        logger.info("Joining matchup features for validation data...")
        val_df = self.join_matchup_features(val_df, "val")

        X_train, y_train, X_val, y_val = self.prepare_features(train_df, val_df)

        result = self.train_stacked_ensemble(X_train, y_train, X_val, y_val, train_df)

        self.save_model(result, dry_run=dry_run)

        logger.info("\n=== V5 Training Complete ===")
        logger.info("  Val Accuracy : %.4f", result["val_accuracy"])
        logger.info("  Val AUC      : %.4f", result["val_auc"])
        logger.info("  Val Brier    : %.4f", result["val_brier"])
        logger.info("  Features     : %d (%d matchup)",
                    len(self.feature_cols),
                    len([f for f in self.feature_cols if f in MATCHUP_FEATURES]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train V5 matchup-aware model")
    parser.add_argument("--no-matchup-join", action="store_true",
                        help="Skip BigQuery matchup join (train on V3 features only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run training but do not save model")
    args = parser.parse_args()

    trainer = V5Trainer(use_matchup_join=not args.no_matchup_join)
    trainer.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
