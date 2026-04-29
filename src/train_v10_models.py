#!/usr/bin/env python3
"""
Train a deployable V10 model artifact using the existing V8 training parquet base
plus lineup/matchup joins that now exist in the live V10 feature table.

This is the practical bridge between the upgraded live feature store and model
retraining: it lets us retrain on slot-aware lineup signals immediately, without
waiting on a separate full historical V10 feature warehouse build.

Usage:
    python train_v10_models.py
    python train_v10_models.py --dry-run
    python train_v10_models.py --skip-upload
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
from google.cloud import bigquery, storage
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from build_v10_features_live import V10_MODEL_FEATURES

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
HIST_DS = "mlb_historical_data"
SEASON_DS = "mlb_2026_season"
BUCKET = "hanks_tank_data"

TRAIN_PATH = Path("data/training/train_v8_2015_2024.parquet")
VAL_PATH = Path("data/training/val_v8_2025.parquet")
OUTPUT_PATH = Path("models/game_outcome_2026_v10.pkl")
OUTPUT_PATH.parent.mkdir(exist_ok=True)
GCS_OUTPUT_PATH = "models/vertex/game_outcome_2026_v10/model.pkl"

MATCHUP_HIST_TABLE = f"{PROJECT}.{HIST_DS}.matchup_features_historical"
MATCHUP_SEASON_TABLE = f"{PROJECT}.{SEASON_DS}.matchup_features"
V10_SEASON_TABLE = f"{PROJECT}.{SEASON_DS}.game_v10_features"
SEASON_GAMES_TABLE = f"{PROJECT}.{SEASON_DS}.games"

TARGET = "home_won"
EXCLUDE_COLS = {
    TARGET, "game_pk", "game_date", "year", "season",
    "home_team_id", "away_team_id", "home_team", "away_team",
    "home_team_name", "away_team_name",
}

MATCHUP_JOIN_COLUMNS = [
    "home_lineup_woba_vs_hand", "away_lineup_woba_vs_hand",
    "home_lineup_k_pct_vs_hand", "away_lineup_k_pct_vs_hand",
    "home_top3_woba_vs_hand", "away_top3_woba_vs_hand",
    "home_middle4_woba_vs_hand", "away_middle4_woba_vs_hand",
    "home_bottom2_woba_vs_hand", "away_bottom2_woba_vs_hand",
    "home_pct_same_hand", "away_pct_same_hand",
    "home_h2h_woba", "away_h2h_woba",
    "home_starter_k_pct", "away_starter_k_pct",
    "matchup_advantage_home", "lineup_confirmed",
]


def evaluate(proba: np.ndarray, y_true: np.ndarray, label: str) -> dict:
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred)
    auc = roc_auc_score(y_true, proba)
    brier = brier_score_loss(y_true, proba)
    ll = log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6))

    confidence = {}
    for tier_name, offset in [("medium", 0.07), ("high", 0.14)]:
        mask = np.abs(proba - 0.5) >= offset
        if mask.any():
            confidence[tier_name] = {
                "coverage": round(float(mask.mean()), 4),
                "accuracy": round(float(accuracy_score(y_true[mask], pred[mask])), 4),
                "games": int(mask.sum()),
            }

    logger.info(
        "%-28s acc=%.4f auc=%.4f brier=%.4f",
        label,
        acc,
        auc,
        brier,
    )
    for tier_name, result in confidence.items():
        logger.info(
            "  %s tier: acc=%.4f coverage=%.1f%% (%d games)",
            tier_name,
            result["accuracy"],
            result["coverage"] * 100,
            result["games"],
        )

    return {
        "label": label,
        "accuracy": round(float(acc), 4),
        "auc": round(float(auc), 4),
        "brier": round(float(brier), 4),
        "log_loss": round(float(ll), 4),
        "confidence": confidence,
    }


class V10Trainer:
    def __init__(self):
        self.bq = bigquery.Client(project=PROJECT)
        self.storage = storage.Client(project=PROJECT)
        self.feature_cols: list[str] = []
        self.fill_values: dict[str, float] = {}

    def _fetch_parquet(self, local_path: Path, gcs_object: str) -> pd.DataFrame:
        if local_path.exists():
            return pd.read_parquet(local_path)

        logger.info(
            "%s not found locally — downloading from gs://%s/%s",
            local_path,
            BUCKET,
            gcs_object,
        )
        data = self.storage.bucket(BUCKET).blob(gcs_object).download_as_bytes()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return pd.read_parquet(local_path)

    def load_base_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train_df = self._fetch_parquet(TRAIN_PATH, "data/training/train_v8_2015_2024.parquet")
            val_df = self._fetch_parquet(VAL_PATH, "data/training/val_v8_2025.parquet")
            logger.info("Loaded base V8 parquet: train=%d | val=%d", len(train_df), len(val_df))
            return train_df, val_df
        except Exception as exc:
            logger.warning("Prebuilt V8 parquet unavailable: %s", exc)

        logger.info("Rebuilding V8 training parquet locally from V3 + BQ sources...")
        self._fetch_parquet(
            Path("data/training/train_v3_2015_2024.parquet"),
            "data/training/train_v3_2015_2024.parquet",
        )
        self._fetch_parquet(
            Path("data/training/val_v3_2025.parquet"),
            "data/training/val_v3_2025.parquet",
        )

        from build_v8_features import V8FeatureBuilder

        builder = V8FeatureBuilder()
        train_df, val_df = builder.build(save=True)
        logger.info("Rebuilt base V8 parquet: train=%d | val=%d", len(train_df), len(val_df))
        return train_df, val_df

    def _query_matchup_table(self, table: str, game_pks: list[int]) -> pd.DataFrame:
        if not game_pks:
            return pd.DataFrame()

        rows = []
        lineup_confirmed_sql = (
            "CAST(TRUE AS BOOL) AS lineup_confirmed"
            if table == MATCHUP_HIST_TABLE
            else "lineup_confirmed"
        )
        for index in range(0, len(game_pks), 1000):
            chunk = game_pks[index:index + 1000]
            pk_list = ", ".join(str(pk) for pk in chunk)
            sql = f"""
                SELECT
                    game_pk,
                    home_lineup_woba_vs_hand,
                    away_lineup_woba_vs_hand,
                    home_lineup_k_pct_vs_hand,
                    away_lineup_k_pct_vs_hand,
                    home_top3_woba_vs_hand,
                    away_top3_woba_vs_hand,
                    home_middle4_woba_vs_hand,
                    away_middle4_woba_vs_hand,
                    home_bottom2_woba_vs_hand,
                    away_bottom2_woba_vs_hand,
                    home_pct_same_hand,
                    away_pct_same_hand,
                    home_h2h_woba,
                    away_h2h_woba,
                    home_starter_k_pct,
                    away_starter_k_pct,
                    matchup_advantage_home,
                    {lineup_confirmed_sql}
                FROM `{table}`
                WHERE game_pk IN ({pk_list})
                QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) = 1
            """
            rows.append(self.bq.query(sql).to_dataframe())

        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def join_matchup_features(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        game_pks = df["game_pk"].dropna().astype(int).tolist()
        joined = pd.DataFrame()

        for table in [MATCHUP_HIST_TABLE, MATCHUP_SEASON_TABLE]:
            try:
                self.bq.get_table(table)
                joined = self._query_matchup_table(table, game_pks)
                if not joined.empty:
                    logger.info("[%s] Joined %d matchup rows from %s", label, len(joined), table)
                    break
            except Exception as exc:
                logger.warning("[%s] Could not join %s: %s", label, table, exc)

        if not joined.empty:
            df = df.merge(joined, on="game_pk", how="left")

        if "lineup_confirmed" not in df.columns:
            df["lineup_confirmed"] = 1.0
        else:
            df["lineup_confirmed"] = (
                df["lineup_confirmed"].fillna(True).astype(int)
            )

        derived_defaults = {
            "lineup_woba_differential": (
                df.get("home_lineup_woba_vs_hand", pd.Series(index=df.index, dtype=float))
                - df.get("away_lineup_woba_vs_hand", pd.Series(index=df.index, dtype=float))
            ),
            "lineup_k_pct_differential": (
                df.get("away_lineup_k_pct_vs_hand", pd.Series(index=df.index, dtype=float))
                - df.get("home_lineup_k_pct_vs_hand", pd.Series(index=df.index, dtype=float))
            ),
            "h2h_woba_differential": (
                df.get("home_h2h_woba", pd.Series(index=df.index, dtype=float))
                - df.get("away_h2h_woba", pd.Series(index=df.index, dtype=float))
            ),
        }
        for col, values in derived_defaults.items():
            df[col] = values

        for col in MATCHUP_JOIN_COLUMNS + [
            "lineup_woba_differential",
            "lineup_k_pct_differential",
            "h2h_woba_differential",
        ]:
            if col not in df.columns:
                df[col] = np.nan

        return df

    def prepare_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        candidate_features = [
            feature
            for feature in V10_MODEL_FEATURES
            if feature not in EXCLUDE_COLS and (feature in train_df.columns or feature in val_df.columns)
        ]
        self.feature_cols = []
        for feature in candidate_features:
            if feature in train_df.columns and pd.api.types.is_numeric_dtype(train_df[feature]):
                self.feature_cols.append(feature)
            elif feature in val_df.columns and pd.api.types.is_numeric_dtype(val_df[feature]):
                self.feature_cols.append(feature)

        missing = [feature for feature in V10_MODEL_FEATURES if feature not in self.feature_cols]
        logger.info(
            "Using %d V10-aligned features; %d canonical columns unavailable in offline base",
            len(self.feature_cols),
            len(missing),
        )
        if missing:
            logger.info("Unavailable offline V10 columns (first 12): %s", missing[:12])

        X_train = pd.DataFrame(index=train_df.index)
        X_val = pd.DataFrame(index=val_df.index)
        for feature in self.feature_cols:
            X_train[feature] = train_df.get(feature)
            X_val[feature] = val_df.get(feature)

        for frame in (X_train, X_val):
            for col in frame.select_dtypes(include=["bool"]).columns:
                frame[col] = frame[col].astype(int)
            for col in frame.select_dtypes(include=["object", "category"]).columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

        for feature in self.feature_cols:
            median = X_train[feature].median()
            fill_value = float(median) if pd.notna(median) else 0.0
            self.fill_values[feature] = fill_value
            X_train[feature] = X_train[feature].fillna(fill_value)
            X_val[feature] = X_val[feature].fillna(fill_value)

        y_train = train_df[TARGET].astype(int)
        y_val = val_df[TARGET].astype(int)
        return X_train, y_train, X_val, y_val

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[xgb.XGBClassifier, dict]:
        model = xgb.XGBClassifier(
            n_estimators=450,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.82,
            colsample_bytree=0.82,
            min_child_weight=4,
            reg_alpha=0.05,
            reg_lambda=1.5,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        metrics = evaluate(model.predict_proba(X_val)[:, 1], y_val.to_numpy(), "V10 2025 holdout")
        return model, metrics

    def load_completed_2026_features(self) -> pd.DataFrame:
        try:
            sql = f"""
                WITH latest_v10 AS (
                    SELECT *
                    FROM `{V10_SEASON_TABLE}`
                    QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) = 1
                )
                SELECT
                    g.game_pk,
                    g.game_date,
                    CAST(g.home_score > g.away_score AS INT64) AS home_won,
                    v10.* EXCEPT (game_pk, game_date, home_team_id, away_team_id, computed_at)
                FROM `{SEASON_GAMES_TABLE}` g
                JOIN latest_v10 v10 ON g.game_pk = v10.game_pk
                WHERE g.home_score IS NOT NULL
                  AND g.away_score IS NOT NULL
                  AND g.game_date < CURRENT_DATE()
                ORDER BY g.game_date
            """
            df = self.bq.query(sql).to_dataframe()
            logger.info("Loaded %d completed 2026 V10 rows for evaluation", len(df))
            return df
        except Exception as exc:
            logger.warning("Could not load completed 2026 V10 features: %s", exc)
            return pd.DataFrame()

    def run(self, dry_run: bool = False, upload: bool = True) -> dict:
        train_df, val_df = self.load_base_data()
        train_df = self.join_matchup_features(train_df, "train")
        val_df = self.join_matchup_features(val_df, "val")

        X_train, y_train, X_val, y_val = self.prepare_features(train_df, val_df)
        if dry_run:
            logger.info("[DRY RUN] Would train V10 with %d features", len(self.feature_cols))
            return {"features": len(self.feature_cols), "dry_run": True}

        model, val_metrics = self.train_model(X_train, y_train, X_val, y_val)

        eval_2026 = None
        df_2026 = self.load_completed_2026_features()
        if not df_2026.empty:
            X_2026 = pd.DataFrame(index=df_2026.index)
            for feature in self.feature_cols:
                X_2026[feature] = df_2026.get(feature, self.fill_values.get(feature, 0.0))
                X_2026[feature] = X_2026[feature].fillna(self.fill_values.get(feature, 0.0))
            eval_2026 = evaluate(
                model.predict_proba(X_2026)[:, 1],
                df_2026[TARGET].astype(int).to_numpy(),
                "V10 completed 2026 games",
            )

        payload = {
            "model": model,
            "version": "v10",
            "model_name": "V10_LineupMatchupXGB",
            "features": self.feature_cols,
            "fill_values": self.fill_values,
            "metrics": {
                "val_2025": val_metrics,
                "eval_2026": eval_2026,
            },
            "trained_at": datetime.utcnow().isoformat(),
        }

        with open(OUTPUT_PATH, "wb") as file_handle:
            pickle.dump(payload, file_handle)
        logger.info("Saved V10 model payload → %s", OUTPUT_PATH)

        if upload:
            blob = self.storage.bucket(BUCKET).blob(GCS_OUTPUT_PATH)
            blob.upload_from_filename(str(OUTPUT_PATH))
            logger.info("Uploaded V10 model → gs://%s/%s", BUCKET, GCS_OUTPUT_PATH)

        return payload


def main():
    parser = argparse.ArgumentParser(description="Train deployable V10 model artifact")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    trainer = V10Trainer()
    result = trainer.run(dry_run=args.dry_run, upload=not args.skip_upload)
    if not args.dry_run:
        logger.info("Result metrics: %s", result.get("metrics"))


if __name__ == "__main__":
    main()
