#!/usr/bin/env python3
"""
Train a game-outcome model on the 2026 feature store + historical features,
then upload to Vertex AI Model Registry for serving.

Training data:
  - Historical: mlb_historical_data.games_historical (2015-2025) with computed features
  - Current: mlb_2026_season.game_features (2026 spring training + regular season)

Model: XGBoost (best performer at 54.6% in v3 eval)

Usage:
    python train_vertex_model.py                 # train + register
    python train_vertex_model.py --local-only    # train, save locally, skip Vertex
    python train_vertex_model.py --dry-run       # preview data only
"""

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
HIST_DATASET = "mlb_historical_data"
SEASON_DATASET = "mlb_2026_season"
BUCKET = "hanks_tank_data"
MODEL_GCS_DIR = "models/vertex/game_outcome_2026"

# V3 feature list — the features our model uses
FEATURE_COLS = [
    "home_win_pct_10d", "away_win_pct_10d",
    "home_win_pct_30d", "away_win_pct_30d",
    "home_ema_form", "away_ema_form",
    "home_pitcher_quality", "away_pitcher_quality",
    "home_days_rest", "away_days_rest",
    "home_rest_advantage",
    "is_back_to_back",
    "month", "day_of_week",
    "is_early_season", "is_late_season",
    "is_home",
    "is_divisional_matchup",
    "home_form_squared", "away_form_squared",
    "rest_balance",
    "home_momentum", "away_momentum",
    "pitcher_quality_diff",
    "fatigue_index",
    "win_pct_diff",
    "home_composite_strength", "away_composite_strength",
]

TARGET = "home_won"


class VertexModelTrainer:
    def __init__(self, dry_run: bool = False, local_only: bool = False):
        self.bq = bigquery.Client(project=PROJECT)
        self.dry_run = dry_run
        self.local_only = local_only
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Build training data from historical games (replicate feature logic)
    # ------------------------------------------------------------------
    def load_historical_features(self) -> pd.DataFrame:
        """Compute features from historical games in BigQuery using SQL."""
        logger.info("Building historical features via BigQuery...")

        q = f"""
        WITH game_list AS (
            SELECT
                game_pk, game_date, game_type,
                home_team_id, home_team_name,
                away_team_id, away_team_name,
                home_score, away_score,
                CASE WHEN home_score > away_score THEN 1 ELSE 0 END AS home_won,
                EXTRACT(MONTH FROM game_date) AS month,
                EXTRACT(DAYOFWEEK FROM game_date) AS day_of_week,
                year
            FROM `{PROJECT}.{HIST_DATASET}.games_historical`
            WHERE status_code = 'F'
              AND game_type = 'R'
              AND year BETWEEN 2018 AND 2025
        ),
        -- Rolling win pct over last 10 games (approx via row_number)
        team_games AS (
            SELECT
                game_pk, game_date, team_id, is_home,
                CASE WHEN (is_home = 1 AND home_score > away_score)
                      OR  (is_home = 0 AND away_score > home_score)
                     THEN 1 ELSE 0 END AS won
            FROM (
                SELECT game_pk, game_date, home_team_id AS team_id,
                       1 AS is_home, home_score, away_score
                FROM game_list
                UNION ALL
                SELECT game_pk, game_date, away_team_id AS team_id,
                       0 AS is_home, home_score, away_score
                FROM game_list
            )
        ),
        rolling AS (
            SELECT
                game_pk, team_id,
                AVG(won) OVER (
                    PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                ) AS win_pct_10,
                AVG(won) OVER (
                    PARTITION BY team_id ORDER BY game_date
                    ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
                ) AS win_pct_30
            FROM team_games
        )
        SELECT
            g.game_pk, g.game_date, g.home_won,
            COALESCE(rh.win_pct_10, 0.5) AS home_win_pct_10d,
            COALESCE(ra.win_pct_10, 0.5) AS away_win_pct_10d,
            COALESCE(rh.win_pct_30, 0.5) AS home_win_pct_30d,
            COALESCE(ra.win_pct_30, 0.5) AS away_win_pct_30d,
            COALESCE(rh.win_pct_10, 0.5) AS home_ema_form,
            COALESCE(ra.win_pct_10, 0.5) AS away_ema_form,
            0.5 AS home_pitcher_quality,
            0.5 AS away_pitcher_quality,
            1 AS home_days_rest,
            1 AS away_days_rest,
            0 AS home_rest_advantage,
            0 AS is_back_to_back,
            g.month,
            g.day_of_week,
            CASE WHEN g.month <= 4 THEN 1 ELSE 0 END AS is_early_season,
            CASE WHEN g.month >= 9 THEN 1 ELSE 0 END AS is_late_season,
            1 AS is_home,
            0 AS is_divisional_matchup,
            POWER(COALESCE(rh.win_pct_10, 0.5), 2) AS home_form_squared,
            POWER(COALESCE(ra.win_pct_10, 0.5), 2) AS away_form_squared,
            0.0 AS rest_balance,
            COALESCE(rh.win_pct_10, 0.5) - COALESCE(rh.win_pct_30, 0.5) AS home_momentum,
            COALESCE(ra.win_pct_10, 0.5) - COALESCE(ra.win_pct_30, 0.5) AS away_momentum,
            0.0 AS pitcher_quality_diff,
            0 AS fatigue_index,
            COALESCE(rh.win_pct_30, 0.5) - COALESCE(ra.win_pct_30, 0.5) AS win_pct_diff,
            0.4*COALESCE(rh.win_pct_10,0.5) + 0.3*COALESCE(rh.win_pct_30,0.5)
                + 0.2*0.5 + 0.1*COALESCE(rh.win_pct_30,0.5) AS home_composite_strength,
            0.4*COALESCE(ra.win_pct_10,0.5) + 0.3*COALESCE(ra.win_pct_30,0.5)
                + 0.2*0.5 + 0.1*COALESCE(ra.win_pct_30,0.5) AS away_composite_strength
        FROM game_list g
        LEFT JOIN rolling rh ON g.game_pk = rh.game_pk
            AND g.home_team_id = rh.team_id
        LEFT JOIN rolling ra ON g.game_pk = ra.game_pk
            AND g.away_team_id = ra.team_id
        ORDER BY g.game_date
        """
        df = self.bq.query(q).to_dataframe()
        logger.info("Historical features: %d games", len(df))
        return df

    def load_2026_features(self) -> pd.DataFrame:
        q = f"""
        SELECT * FROM `{PROJECT}.{SEASON_DATASET}.game_features`
        WHERE game_type IN ('R', 'S')
        ORDER BY game_date
        """
        try:
            df = self.bq.query(q).to_dataframe()
            logger.info("2026 features: %d games", len(df))
            return df
        except Exception as e:
            logger.warning("No 2026 features yet: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train(self):
        logger.info("=" * 60)
        logger.info("VERTEX AI MODEL TRAINING")
        logger.info("=" * 60)

        hist = self.load_historical_features()
        current = self.load_2026_features()

        # Combine
        if not current.empty:
            # Align columns
            for c in FEATURE_COLS:
                if c not in current.columns:
                    current[c] = 0
            combined = pd.concat([
                hist[FEATURE_COLS + [TARGET]],
                current[FEATURE_COLS + [TARGET]],
            ], ignore_index=True)
        else:
            combined = hist[FEATURE_COLS + [TARGET]]

        combined = combined.dropna(subset=[TARGET])
        logger.info("Total training samples: %d", len(combined))

        if self.dry_run:
            logger.info("[DRY RUN] Would train on %d samples with %d features",
                        len(combined), len(FEATURE_COLS))
            return None

        X = combined[FEATURE_COLS].fillna(0).values.astype(np.float32)
        y = combined[TARGET].values.astype(int)

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Time-series CV
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="logloss",
                early_stopping_rounds=20,
                use_label_encoder=False,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            val_pred = model.predict(X_val)
            val_prob = model.predict_proba(X_val)[:, 1]
            acc = accuracy_score(y_val, val_pred)
            auc = roc_auc_score(y_val, val_prob)
            ll = log_loss(y_val, val_prob)
            cv_scores.append({"fold": fold, "accuracy": acc, "auc": auc, "log_loss": ll})
            logger.info("  Fold %d: acc=%.4f  auc=%.4f  ll=%.4f", fold, acc, auc, ll)

        avg_acc = np.mean([s["accuracy"] for s in cv_scores])
        avg_auc = np.mean([s["auc"] for s in cv_scores])
        logger.info("CV Mean: acc=%.4f  auc=%.4f", avg_acc, avg_auc)

        # Final model on all data
        logger.info("Training final model on all %d samples...", len(X_scaled))
        final_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        final_model.fit(X_scaled, y)

        # Pack artifacts
        artifacts = {
            "model": final_model,
            "scaler": scaler,
            "features": FEATURE_COLS,
            "model_name": "XGBoost_2026_vertex",
            "trained_at": datetime.utcnow().isoformat(),
            "training_samples": len(X_scaled),
            "cv_accuracy": float(avg_acc),
            "cv_auc": float(avg_auc),
            "cv_scores": cv_scores,
        }

        # Save locally
        local_path = self.model_dir / "game_outcome_2026_vertex.pkl"
        with open(local_path, "wb") as f:
            pickle.dump(artifacts, f)
        logger.info("✓ Model saved to %s", local_path)

        # Feature importance
        imp = dict(zip(FEATURE_COLS, final_model.feature_importances_.tolist()))
        imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top 10 features:")
        for feat, score in imp_sorted[:10]:
            logger.info("  %-30s %.4f", feat, score)

        if not self.local_only:
            self._upload_to_gcs(local_path)
            self._register_vertex_model()

        return artifacts

    # ------------------------------------------------------------------
    # GCS + Vertex registration
    # ------------------------------------------------------------------
    def _upload_to_gcs(self, local_path: Path):
        logger.info("Uploading model to GCS...")
        client = storage.Client(project=PROJECT)
        bucket = client.bucket(BUCKET)

        # Upload pickle
        blob = bucket.blob(f"{MODEL_GCS_DIR}/model.pkl")
        blob.upload_from_filename(str(local_path))
        logger.info("✓ Uploaded to gs://%s/%s/model.pkl", BUCKET, MODEL_GCS_DIR)

        # Upload metadata
        meta = {
            "framework": "xgboost",
            "features": FEATURE_COLS,
            "target": TARGET,
            "uploaded_at": datetime.utcnow().isoformat(),
        }
        meta_blob = bucket.blob(f"{MODEL_GCS_DIR}/metadata.json")
        meta_blob.upload_from_string(json.dumps(meta, indent=2),
                                      content_type="application/json")

    def _register_vertex_model(self):
        """Register model in Vertex AI Model Registry."""
        try:
            from google.cloud import aiplatform

            aiplatform.init(project=PROJECT, location="us-central1")

            model = aiplatform.Model.upload(
                display_name="mlb-game-outcome-2026",
                artifact_uri=f"gs://{BUCKET}/{MODEL_GCS_DIR}",
                serving_container_image_uri=(
                    "us-docker.pkg.dev/vertex-ai/prediction/"
                    "sklearn-cpu.1-3:latest"
                ),
                description=(
                    "XGBoost game outcome model for 2026 MLB season. "
                    "Predicts home team win probability."
                ),
                labels={"sport": "mlb", "season": "2026", "version": "v1"},
            )
            logger.info("✓ Registered Vertex AI model: %s", model.resource_name)
        except ImportError:
            logger.warning("google-cloud-aiplatform not installed — skipping Vertex registration")
        except Exception as e:
            logger.error("Vertex registration failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Train + register Vertex AI model")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trainer = VertexModelTrainer(dry_run=args.dry_run, local_only=args.local_only)
    trainer.train()


if __name__ == "__main__":
    main()
