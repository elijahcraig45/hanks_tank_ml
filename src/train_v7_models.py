#!/usr/bin/env python3
"""
V7 Model Training — Bullpen Health + Moon Phase + Pitcher Venue Splits

Extends V6 (pitcher arsenal + venue history) with:
  - Bullpen fatigue / depth (7-day rolling pitch count, leverage-weighted)
  - Moon phase & circadian offset
  - Starter historical performance at specific ballpark
  - Recalibrated home-advantage weight (reduced from V6)

Architecture: Same V5/V6 stacked ensemble (LR + XGBoost + LightGBM → meta LR)
New V7 features:
  Bullpen health (home + away):
    *_bullpen_pitches_7d, *_bullpen_games_7d, *_bullpen_ip_7d
    *_bullpen_fatigue_score, *_closer_days_rest, *_bullpen_depth_score
    bullpen_fatigue_differential
  Moon / circadian:
    moon_phase, moon_illumination, is_full_moon, is_new_moon, moon_waxing
    home_circadian_offset, away_circadian_offset, circadian_differential
  Pitcher venue:
    home_starter_venue_era, *_whip, *_k9, *_pa_total
    away_starter_venue_era, *_whip, *_k9, *_pa_total
    starter_venue_era_differential
  Home advantage recalibration:
    park_ha_recalibrated  (replaces raw is_home constant; ~28% reduced weight,
                           moon-phase modulated)

Training data: train_v3_2015_2024.parquet + v5/v6/v7 feature joins from BQ
Validation:    val_v3_2025.parquet

Output:
    models/game_outcome_2026_v7.pkl          (local)
    gs://hanks_tank_data/models/vertex/game_outcome_2026_v7/model.pkl  (GCS)

Usage:
    python train_v7_models.py                    # full training + GCS upload
    python train_v7_models.py --no-v7-join       # V6 features only (fast test)
    python train_v7_models.py --dry-run          # feature count preview, no save
    python train_v7_models.py --skip-upload      # train but don't push to GCS
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

from sklearn.calibration import CalibratedClassifierCV  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from model_classes import StackedV7Model  # noqa: F401 — needed for pickle

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT   = "hankstank"
HIST_DS   = "mlb_historical_data"
SEASON_DS = "mlb_2026_season"

TRAIN_PATH  = Path("data/training/train_v3_2015_2024.parquet")
VAL_PATH    = Path("data/training/val_v3_2025.parquet")
OUTPUT_PATH = Path("models/game_outcome_2026_v7.pkl")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

GCS_BUCKET  = "hanks_tank_data"
GCS_V7_PATH = "models/vertex/game_outcome_2026_v7/model.pkl"

TARGET = "home_won"
EXCLUDE_COLS = {
    TARGET, "game_pk", "game_date", "year", "season",
    "home_team_id", "away_team_id", "home_team", "away_team",
    "home_team_name", "away_team_name",
}

# ---------------------------------------------------------------------------
# V6 feature set (baseline carried through)
# ---------------------------------------------------------------------------
V6_CORE_FEATURES = [
    # Rolling form
    "home_ema_form", "away_ema_form",
    "home_form_squared", "away_form_squared",
    "home_momentum", "away_momentum",
    "win_pct_diff", "form_difference", "form_interaction", "trend_alignment",
    "home_trend_direction", "away_trend_direction",
    "home_win_pct_10d", "away_win_pct_10d",
    "home_win_pct_30d", "away_win_pct_30d",
    # Rest and travel
    "home_team_rest_days", "away_team_rest_days",
    "rest_balance", "is_back_to_back", "fatigue_index",
    "travel_distance_km",
    # Pitcher / park
    "home_pitcher_quality", "away_pitcher_quality",
    "pitcher_quality_diff",
    "home_park_run_factor", "away_park_run_factor",
    "park_advantage",
    # Composite strength
    "home_composite_strength", "away_composite_strength",
    # Temporal
    "month", "day_of_week", "is_home",
    "month_3", "month_4", "month_5", "month_6",
    "month_7", "month_8", "month_9", "month_10", "month_11",
    "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6", "dow_7",
    "season_phase_home_effect", "month_home_effect",
    # V5 matchup
    "lineup_woba_differential", "starter_woba_differential",
    "matchup_advantage_home", "home_pct_same_hand", "away_pct_same_hand",
    "h2h_woba_differential", "home_top3_woba_vs_hand", "away_top3_woba_vs_hand",
    "lineup_k_pct_differential", "home_starter_k_pct", "away_starter_k_pct",
    # V6 pitcher arsenal
    "home_starter_fastball_pct", "away_starter_fastball_pct",
    "home_starter_breaking_pct", "away_starter_breaking_pct",
    "home_starter_offspeed_pct", "away_starter_offspeed_pct",
    "home_starter_xwoba_allowed", "away_starter_xwoba_allowed",
    "home_starter_k_bb_pct", "away_starter_k_bb_pct",
    "home_starter_velo_norm", "away_starter_velo_norm",
    "home_starter_velo_trend", "away_starter_velo_trend",
    "starter_arsenal_advantage",
    # V6 venue history (batting)
    "home_lineup_venue_woba", "away_lineup_venue_woba",
    "venue_woba_differential", "home_venue_advantage", "away_venue_disadvantage",
]

# New V7 features
V7_NEW_FEATURES = [
    # Bullpen health
    "home_bullpen_pitches_7d", "away_bullpen_pitches_7d",
    "home_bullpen_games_7d",   "away_bullpen_games_7d",
    "home_bullpen_ip_7d",      "away_bullpen_ip_7d",
    "home_bullpen_fatigue_score", "away_bullpen_fatigue_score",
    "home_closer_days_rest",      "away_closer_days_rest",
    "home_bullpen_depth_score",   "away_bullpen_depth_score",
    "bullpen_fatigue_differential",
    # Moon / circadian
    "moon_phase", "moon_illumination",
    "is_full_moon", "is_new_moon", "moon_waxing",
    "home_circadian_offset", "away_circadian_offset",
    "circadian_differential",
    # Pitcher venue splits
    "home_starter_venue_era",  "home_starter_venue_whip",
    "home_starter_venue_k9",   "home_starter_venue_pa_total",
    "away_starter_venue_era",  "away_starter_venue_whip",
    "away_starter_venue_k9",   "away_starter_venue_pa_total",
    "starter_venue_era_differential",
    # HA recalibration
    "park_ha_recalibrated",
]

ALL_V7_FEATURES = V6_CORE_FEATURES + V7_NEW_FEATURES


# ---------------------------------------------------------------------------
# BQ join helpers (reuse V5 + V6 style, add V7 table)
# ---------------------------------------------------------------------------

def _join_from_table(
    bq, df: pd.DataFrame, table: str, cols: list[str], label: str
) -> pd.DataFrame:
    """Generic BQ join by game_pk over chunked batches."""
    game_pks = df["game_pk"].dropna().astype(int).tolist()
    if not game_pks:
        return df
    rows = []
    for i in range(0, len(game_pks), 1000):
        chunk = game_pks[i:i + 1000]
        pk_list = ", ".join(str(pk) for pk in chunk)
        select_cols = ", ".join(["game_pk"] + cols)
        sql = f"""
            SELECT {select_cols}
            FROM `{table}`
            WHERE game_pk IN ({pk_list})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) = 1
        """
        try:
            rows.append(bq.query(sql).to_dataframe())
        except Exception as e:
            logger.warning("[%s] chunk failed: %s", label, e)
    if rows:
        joined = pd.concat(rows, ignore_index=True)
        df = df.merge(joined, on="game_pk", how="left")
        logger.info("[%s] Joined %d/%d games from %s", label, joined["game_pk"].nunique(), len(game_pks), table)
    return df


def join_v5_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    try:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT)
        for tbl in [
            f"{PROJECT}.{HIST_DS}.matchup_features_historical",
            f"{PROJECT}.{SEASON_DS}.matchup_features",
        ]:
            try:
                bq.get_table(tbl)
                cols = [
                    "home_lineup_woba_vs_hand - away_lineup_woba_vs_hand AS lineup_woba_differential",
                    "away_starter_woba_allowed - home_starter_woba_allowed AS starter_woba_differential",
                    "matchup_advantage_home", "home_pct_same_hand", "away_pct_same_hand",
                    "COALESCE(home_h2h_woba,0.320)-COALESCE(away_h2h_woba,0.320) AS h2h_woba_differential",
                    "home_top3_woba_vs_hand", "away_top3_woba_vs_hand",
                    "away_lineup_k_pct_vs_hand-home_lineup_k_pct_vs_hand AS lineup_k_pct_differential",
                    "home_starter_k_pct", "away_starter_k_pct",
                ]
                game_pks = df["game_pk"].dropna().astype(int).tolist()
                rows = []
                for i in range(0, len(game_pks), 1000):
                    chunk = game_pks[i:i + 1000]
                    pk_list = ", ".join(str(pk) for pk in chunk)
                    sql = f"SELECT game_pk, {', '.join(cols)} FROM `{tbl}` WHERE game_pk IN ({pk_list}) QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) = 1"
                    rows.append(bq.query(sql).to_dataframe())
                if rows:
                    joined = pd.concat(rows, ignore_index=True)
                    df = df.merge(joined, on="game_pk", how="left")
                    logger.info("[%s] Joined V5 features: %d games", label, joined["game_pk"].nunique())
                break
            except Exception:
                continue
    except Exception as e:
        logger.warning("[%s] V5 join failed: %s", label, e)
    return df


def join_v6_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    try:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT)
        v6_tbl = f"{PROJECT}.{SEASON_DS}.matchup_v6_features"
        bq.get_table(v6_tbl)
        v6_cols = [
            "home_starter_fastball_pct", "away_starter_fastball_pct",
            "home_starter_breaking_pct", "away_starter_breaking_pct",
            "home_starter_offspeed_pct", "away_starter_offspeed_pct",
            "home_starter_xwoba_allowed", "away_starter_xwoba_allowed",
            "home_starter_k_bb_pct", "away_starter_k_bb_pct",
            "home_starter_velo_norm", "away_starter_velo_norm",
            "home_starter_velo_trend", "away_starter_velo_trend",
            "starter_arsenal_advantage",
            "home_lineup_venue_woba", "away_lineup_venue_woba",
            "venue_woba_differential", "home_venue_advantage", "away_venue_disadvantage",
        ]
        df = _join_from_table(bq, df, v6_tbl, v6_cols, f"{label}/v6")
    except Exception as e:
        logger.warning("[%s] V6 join failed: %s", label, e)
    for col in [
        "home_starter_fastball_pct", "away_starter_fastball_pct",
        "home_starter_breaking_pct", "away_starter_breaking_pct",
        "home_starter_offspeed_pct", "away_starter_offspeed_pct",
        "home_starter_xwoba_allowed", "away_starter_xwoba_allowed",
        "home_starter_k_bb_pct", "away_starter_k_bb_pct",
        "home_starter_velo_norm", "away_starter_velo_norm",
        "home_starter_velo_trend", "away_starter_velo_trend",
        "starter_arsenal_advantage",
        "home_lineup_venue_woba", "away_lineup_venue_woba",
        "venue_woba_differential", "home_venue_advantage", "away_venue_disadvantage",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def join_v7_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Join V7 bullpen/moon/pitcher-venue from matchup_v7_features.
    For historical training data, moon features are computed locally;
    bullpen and pitcher-venue will be mostly NaN → neutral imputed."""
    try:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT)
        v7_tbl = f"{PROJECT}.{SEASON_DS}.matchup_v7_features"
        bq.get_table(v7_tbl)
        df = _join_from_table(bq, df, v7_tbl, [c for c in V7_NEW_FEATURES], f"{label}/v7")
    except Exception as e:
        logger.warning("[%s] V7 table join skipped: %s — computing moon locally", label, e)

    # Compute moon phase locally for training rows that couldn't be joined
    if "moon_phase" not in df.columns or df["moon_phase"].isna().all():
        logger.info("[%s] Computing moon features from game_date...", label)
        from build_v7_features import compute_moon_features
        if "game_date" in df.columns:
            import datetime as dt
            moon_rows = []
            for gd in df["game_date"]:
                try:
                    d = gd if isinstance(gd, dt.date) else pd.Timestamp(gd).date()
                    moon_rows.append(compute_moon_features(d))
                except Exception:
                    moon_rows.append({
                        "moon_phase": 0.25, "moon_illumination": 50.0,
                        "is_full_moon": 0, "is_new_moon": 0, "moon_waxing": 1,
                    })
            moon_df = pd.DataFrame(moon_rows)
            for col in moon_df.columns:
                df[col] = moon_df[col].values
        else:
            for col in ["moon_phase", "moon_illumination", "is_full_moon", "is_new_moon", "moon_waxing"]:
                df[col] = np.nan

    # park_ha_recalibrated: compute from park_run_factor + moon_phase if not joined
    if "park_ha_recalibrated" not in df.columns or df["park_ha_recalibrated"].isna().all():
        prf = df.get("home_park_run_factor", pd.Series(np.ones(len(df))))
        mp  = df.get("moon_phase", pd.Series(np.full(len(df), 0.25)))
        df["park_ha_recalibrated"] = (prf * 0.72 * (1.0 - 0.05 * mp)).round(4)

    # Ensure all V7 columns exist
    for col in V7_NEW_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class V7Trainer:
    def __init__(
        self,
        use_v5_join: bool = True,
        use_v6_join: bool = True,
        use_v7_join: bool = True,
    ):
        self.use_v5_join = use_v5_join
        self.use_v6_join = use_v6_join
        self.use_v7_join = use_v7_join
        self.feature_cols: list[str] = []
        self.fill_values: dict = {}

    def _fetch_parquet(self, local_path: Path, gcs_object: str) -> pd.DataFrame:
        """Load parquet from local disk, falling back to GCS download if not found."""
        if local_path.exists():
            return pd.read_parquet(local_path)
        logger.info("%s not found locally — downloading from gs://%s/%s",
                    local_path, GCS_BUCKET, gcs_object)
        from google.cloud import storage as gcs_storage
        client = gcs_storage.Client(project=PROJECT)
        data = client.bucket(GCS_BUCKET).blob(gcs_object).download_as_bytes()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return pd.read_parquet(local_path)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Loading V3 base training data...")
        train_df = self._fetch_parquet(TRAIN_PATH, "data/training/train_v3_2015_2024.parquet")
        val_df   = self._fetch_parquet(VAL_PATH,   "data/training/val_v3_2025.parquet")
        logger.info("Train: %d rows | Val: %d rows", len(train_df), len(val_df))
        return train_df, val_df

    def prepare_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        available = [f for f in ALL_V7_FEATURES if f in train_df.columns]
        missing   = [f for f in ALL_V7_FEATURES if f not in train_df.columns]
        if missing:
            logger.warning("Missing V7 features (will impute with train median): %s", missing[:10])

        # Also include any extra numeric columns in the training data
        numeric_cols = [
            c for c in train_df.columns
            if pd.api.types.is_numeric_dtype(train_df[c]) and c not in EXCLUDE_COLS
        ]
        self.feature_cols = list(dict.fromkeys(
            available + [c for c in numeric_cols if c not in available]
        ))

        # Down-weight home advantage features: scale is_home & park_ha_recalibrated
        # by 0.72 so that the model doesn't over-rely on home field constant.
        for col in ("is_home",):
            if col in train_df.columns:
                train_df[col] = train_df[col] * 0.72
                val_df[col]   = val_df[col] * 0.72

        logger.info(
            "V7 features: %d total  (V6 base: %d, V7 new: %d, extra: %d)",
            len(self.feature_cols),
            len([f for f in self.feature_cols if f in V6_CORE_FEATURES]),
            len([f for f in self.feature_cols if f in V7_NEW_FEATURES]),
            len([f for f in self.feature_cols if f not in ALL_V7_FEATURES]),
        )

        for col in self.feature_cols:
            if col not in train_df.columns:
                train_df[col] = np.nan
            if col not in val_df.columns:
                val_df[col] = np.nan

        # Training-time median imputation (NOT zero — avoids signal contamination)
        for col in self.feature_cols:
            med = train_df[col].median()
            fill_val = float(med) if pd.notna(med) else 0.0
            self.fill_values[col] = fill_val
            train_df[col] = train_df[col].fillna(fill_val)
            val_df[col]   = val_df[col].fillna(fill_val)

        X_train = train_df[self.feature_cols].values.astype(float)
        y_train = train_df[TARGET].values.astype(int)
        X_val   = val_df[self.feature_cols].values.astype(float)
        y_val   = val_df[TARGET].values.astype(int)

        logger.info("X_train: %s | X_val: %s", X_train.shape, X_val.shape)
        return X_train, y_train, X_val, y_val

    def _base_models(self) -> dict:
        return {
            "lr": Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    max_iter=2000, C=0.5, solver="liblinear", random_state=42
                )),
            ]),
            "xgb": xgb.XGBClassifier(
                n_estimators=700, learning_rate=0.035, max_depth=5,
                subsample=0.8, colsample_bytree=0.75, min_child_weight=3,
                reg_lambda=1.5, reg_alpha=0.1, eval_metric="logloss",
                early_stopping_rounds=30, random_state=42, verbosity=0,
            ),
            "lgbm": lgb.LGBMClassifier(
                n_estimators=700, learning_rate=0.035, num_leaves=31,
                max_depth=6, subsample=0.8, colsample_bytree=0.75,
                min_child_samples=20, reg_lambda=1.5, reg_alpha=0.1,
                random_state=42, verbose=-1,
            ),
        }

    def train(
        self, X_train, y_train, X_val, y_val, train_df
    ) -> dict:
        base_models = self._base_models()
        years = sorted(train_df["year"].dropna().unique().astype(int))
        n_base = len(base_models)

        oof_train = np.zeros((len(X_train), n_base))
        oof_val   = np.zeros((len(X_val),   n_base))

        logger.info("Expanding-year CV over: %s", years)

        for fold_year in years[3:]:
            tr_mask = train_df["year"].astype(int) < fold_year
            va_mask = train_df["year"].astype(int) == fold_year
            if tr_mask.sum() < 100 or va_mask.sum() < 10:
                continue
            X_fold_tr = X_train[tr_mask]
            y_fold_tr = y_train[tr_mask]
            X_fold_va = X_train[va_mask]

            for j, (name, model) in enumerate(base_models.items()):
                if name == "xgb":
                    model.fit(X_fold_tr, y_fold_tr,
                              eval_set=[(X_fold_va, y_train[va_mask])],
                              verbose=False)
                else:
                    model.fit(X_fold_tr, y_fold_tr)
                oof_train[va_mask, j] = model.predict_proba(X_fold_va)[:, 1]

        # Final fit on full training set
        for j, (name, model) in enumerate(base_models.items()):
            if name == "xgb":
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)
            oof_val[:, j] = model.predict_proba(X_val)[:, 1]

        meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        meta.fit(oof_train, y_train)

        val_preds = meta.predict_proba(oof_val)[:, 1]
        acc    = accuracy_score(y_val, (val_preds >= 0.5).astype(int))
        auc    = roc_auc_score(y_val, val_preds)
        brier  = brier_score_loss(y_val, val_preds)
        logger.info("V7 Val — Acc: %.4f | AUC: %.4f | Brier: %.4f", acc, auc, brier)
        logger.info("V7 Val home win rate: %.3f | predicted: %.3f",
                    y_val.mean(), val_preds.mean())

        return {
            "base_models": base_models,
            "meta": meta,
            "metrics": {"accuracy": float(acc), "auc": float(auc), "brier": float(brier)},
        }

    def run(self, dry_run: bool = False, upload: bool = True) -> dict:
        train_df, val_df = self.load_data()

        if self.use_v5_join:
            train_df = join_v5_features(train_df, "train")
            val_df   = join_v5_features(val_df,   "val")

        if self.use_v6_join:
            train_df = join_v6_features(train_df, "train")
            val_df   = join_v6_features(val_df,   "val")

        if self.use_v7_join:
            train_df = join_v7_features(train_df, "train")
            val_df   = join_v7_features(val_df,   "val")

        X_train, y_train, X_val, y_val = self.prepare_features(train_df, val_df)

        if dry_run:
            logger.info("[DRY RUN] Would train V7 with %d features", len(self.feature_cols))
            return {"features": len(self.feature_cols), "dry_run": True}

        result = self.train(X_train, y_train, X_val, y_val, train_df)

        # Serialize
        stacked = StackedV7Model(
            base_models=result["base_models"],
            meta=result["meta"],
        )
        payload = {
            "model":        stacked,
            "model_name":   "V7_BullpenMoonVenueSplit",
            "version":      "v7",
            "feature_cols": self.feature_cols,
            "fill_values":  self.fill_values,
            "metrics":      result["metrics"],
            "trained_at":   datetime.utcnow().isoformat(),
        }
        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Saved V7 model → %s", OUTPUT_PATH)

        if upload:
            from google.cloud import storage
            client = storage.Client(project=PROJECT)
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(GCS_V7_PATH)
            blob.upload_from_filename(str(OUTPUT_PATH))
            logger.info("Uploaded V7 model → gs://%s/%s", GCS_BUCKET, GCS_V7_PATH)

        return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-v7-join",   action="store_true")
    parser.add_argument("--no-v6-join",   action="store_true")
    parser.add_argument("--no-v5-join",   action="store_true")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--skip-upload",  action="store_true")
    args = parser.parse_args()

    trainer = V7Trainer(
        use_v5_join=not args.no_v5_join,
        use_v6_join=not args.no_v6_join,
        use_v7_join=not args.no_v7_join,
    )
    result = trainer.run(dry_run=args.dry_run, upload=not args.skip_upload)
    if not args.dry_run:
        m = result.get("metrics", {})
        print(f"\nV7 Results — Accuracy: {m.get('accuracy', 0):.4f} | AUC: {m.get('auc', 0):.4f}")


if __name__ == "__main__":
    main()
