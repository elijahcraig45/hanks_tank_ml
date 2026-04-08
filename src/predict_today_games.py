#!/usr/bin/env python3
"""
Daily Game Predictor with Matchup Features

Triggered per-game (via Cloud Task) ~90 minutes before first pitch.
Combines classic team-level rolling features (V3/V4) with the new
pitcher/batter matchup features to predict each game's outcome.

Reads from:
  - mlb_2026_season.game_features    (team rolling stats, rest, park factors)
  - mlb_2026_season.matchup_features (H2H, splits, lineup-level)
  - models/game_outcome_2026_matchup.pkl (V5 model, trained weekly)

Writes to:
  - mlb_2026_season.game_predictions  (per-game, per-day predictions)

Usage:
    python predict_today_games.py                    # today's games
    python predict_today_games.py --date 2026-04-02
    python predict_today_games.py --game-pk 745612,745613
    python predict_today_games.py --dry-run
    python predict_today_games.py --fallback-v4      # use V4 if V5 not yet trained
"""

import argparse
import logging
import pickle
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import urllib3
from google.cloud import bigquery, storage
from model_classes import StackedV5Model  # noqa: F401 — needed for pickle deserialization

# ---------------------------------------------------------------------------
# Repo root — works regardless of working directory (local dev or Cloud Function)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _REPO_ROOT / "models"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
DATASET = "mlb_2026_season"
BUCKET = "hanks_tank_data"
MLB_API = "https://statsapi.mlb.com/api/v1"

GAME_FEATURES_TABLE  = f"{PROJECT}.{DATASET}.game_features"
MATCHUP_TABLE        = f"{PROJECT}.{DATASET}.matchup_features"
MATCHUP_V7_TABLE     = f"{PROJECT}.{DATASET}.matchup_v7_features"
V8_FEATURES_TABLE    = f"{PROJECT}.{DATASET}.game_v8_features"
PREDICTIONS_TABLE    = f"{PROJECT}.{DATASET}.game_predictions"
LINEUPS_TABLE        = f"{PROJECT}.{DATASET}.lineups"

# ---------------------------------------------------------------------------
# Model resolution chain: V8 → V7 → V6 → V5 → V4
# Each model is tried in order (local path first, then GCS) until one loads.
# V8 is the current best (57.65% overall, 65.4% at ≥60% confidence).
# V7 is the previous production model. V4 is the final fallback.
# ---------------------------------------------------------------------------
V8_MODEL_GCS = "models/vertex/game_outcome_2026_v8/model.pkl"
V7_MODEL_GCS = "models/vertex/game_outcome_2026_v7/model.pkl"
V6_MODEL_GCS = "models/vertex/game_outcome_2026_v6/model.pkl"
V5_MODEL_GCS = "models/vertex/game_outcome_2026_v5/model.pkl"
V4_MODEL_GCS = "models/vertex/game_outcome_2026/model.pkl"
V8_LOCAL = _MODELS_DIR / "game_outcome_2026_v8_final.pkl"
V7_LOCAL = _MODELS_DIR / "game_outcome_2026_v7.pkl"
V6_LOCAL = _MODELS_DIR / "game_outcome_2026_v6.pkl"
V5_LOCAL = _MODELS_DIR / "game_outcome_2026_v5.pkl"
V4_LOCAL = _MODELS_DIR / "game_outcome_2026_vertex.pkl"

# V8 uses updated confidence thresholds based on validation experiments:
#   high   (≥60%): 65.4% accuracy on 11.7% of games — actionable picks
#   medium (≥55%): 59.9% accuracy on 43% of games  — solid signal
#   low    (<55%): essentially a coin flip            — show but flag
CONFIDENCE_TIERS = {"high": 0.60, "medium": 0.55, "low": 0.50}

PREDICTIONS_SCHEMA = [
    bigquery.SchemaField("game_pk", "INTEGER"),
    bigquery.SchemaField("game_date", "DATE"),
    bigquery.SchemaField("home_team_id", "INTEGER"),
    bigquery.SchemaField("home_team_name", "STRING"),
    bigquery.SchemaField("away_team_id", "INTEGER"),
    bigquery.SchemaField("away_team_name", "STRING"),
    bigquery.SchemaField("home_starter_id", "INTEGER"),
    bigquery.SchemaField("home_starter_name", "STRING"),
    bigquery.SchemaField("away_starter_id", "INTEGER"),
    bigquery.SchemaField("away_starter_name", "STRING"),
    bigquery.SchemaField("home_win_probability", "FLOAT"),
    bigquery.SchemaField("away_win_probability", "FLOAT"),
    bigquery.SchemaField("predicted_winner", "STRING"),
    bigquery.SchemaField("confidence_tier", "STRING"),
    bigquery.SchemaField("model_version", "STRING"),
    bigquery.SchemaField("lineup_confirmed", "BOOLEAN"),
    bigquery.SchemaField("matchup_advantage_home", "FLOAT"),
    bigquery.SchemaField("home_lineup_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_lineup_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_starter_hand", "STRING"),
    bigquery.SchemaField("away_starter_hand", "STRING"),
    bigquery.SchemaField("h2h_data_available", "BOOLEAN"),
    bigquery.SchemaField("game_time_utc", "TIMESTAMP"),
    bigquery.SchemaField("predicted_at", "TIMESTAMP"),
    # V7 pitcher arsenal (2026 statcast)
    bigquery.SchemaField("home_starter_mean_velo", "FLOAT"),
    bigquery.SchemaField("away_starter_mean_velo", "FLOAT"),
    bigquery.SchemaField("home_starter_velo_norm", "FLOAT"),
    bigquery.SchemaField("away_starter_velo_norm", "FLOAT"),
    bigquery.SchemaField("home_starter_k_bb_pct", "FLOAT"),
    bigquery.SchemaField("away_starter_k_bb_pct", "FLOAT"),
    bigquery.SchemaField("home_starter_xwoba_allowed", "FLOAT"),
    bigquery.SchemaField("away_starter_xwoba_allowed", "FLOAT"),
    bigquery.SchemaField("starter_arsenal_advantage", "FLOAT"),
    # V7 bullpen health
    bigquery.SchemaField("home_bullpen_fatigue_score", "FLOAT"),
    bigquery.SchemaField("away_bullpen_fatigue_score", "FLOAT"),
    bigquery.SchemaField("bullpen_fatigue_differential", "FLOAT"),
    bigquery.SchemaField("home_closer_days_rest", "FLOAT"),
    bigquery.SchemaField("away_closer_days_rest", "FLOAT"),
    # V7 moon phase
    bigquery.SchemaField("moon_illumination", "FLOAT"),
    bigquery.SchemaField("is_full_moon", "INTEGER"),
    # V7 pitcher venue splits
    bigquery.SchemaField("home_starter_venue_era", "FLOAT"),
    bigquery.SchemaField("away_starter_venue_era", "FLOAT"),
    bigquery.SchemaField("starter_venue_era_differential", "FLOAT"),
    # V7 venue wOBA
    bigquery.SchemaField("venue_woba_differential", "FLOAT"),
]



class V8EnsemblePredictor:
    """Adapter that wraps the V8 multi-model bundle for sklearn-style inference.

    The V8 bundle stores three sub-models (CatBoost cat_tuned, CatBoost cat_wide,
    LightGBM lgb, MLP mlp) each with its own feature set and optional scaler.
    This class routes inference through all sub-models and blends the home-win
    probabilities using the Optuna-optimised weights from training time.
    """

    def __init__(self, bundle: dict):
        self.models = bundle["models"]
        self.scalers = bundle.get("scalers", {})
        self.weights = bundle["weights"]
        self.feature_sets = bundle["feature_sets"]
        self.fill_values = bundle.get("fill_values", {})
        # CatBoost categorical feature names (team IDs) for CatBoost sub-models
        cat_idx = bundle.get("cat_feature_indices", [])
        # Resolve indices → column names from the cat_tuned feature set
        cat_cols = self.feature_sets.get("cat_tuned", [])
        self._cat_feature_names = [cat_cols[i] for i in cat_idx if i < len(cat_cols)]
        # Stable ordered feature union for external callers
        seen: set = set()
        self.all_features: list[str] = []
        for fs in self.feature_sets.values():
            for f in fs:
                if f not in seen:
                    seen.add(f)
                    self.all_features.append(f)

    def _is_catboost(self, model) -> bool:
        return type(model).__module__.startswith("catboost")

    def _proba_one(self, name: str, model, X_df: pd.DataFrame) -> np.ndarray:
        """Home-win probability from a single sub-model."""
        cols = self.feature_sets[name]
        X_sub = pd.DataFrame(index=X_df.index)
        for col in cols:
            X_sub[col] = X_df[col] if col in X_df.columns else self.fill_values.get(col, 0.0)
        X_sub = X_sub.fillna(0.0)

        if self._is_catboost(model):
            # CatBoost requires categorical columns to be int/str, not float.
            # Pass as DataFrame so CatBoost resolves cat_features by column name.
            cat_present = [c for c in self._cat_feature_names if c in X_sub.columns]
            for col in cat_present:
                X_sub[col] = X_sub[col].astype(int)
            return model.predict_proba(X_sub, ntree_start=0)[:, 1]

        scaler = self.scalers.get(name)
        X_arr = scaler.transform(X_sub.values) if scaler else X_sub.values
        return model.predict_proba(X_arr)[:, 1]

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        """Weighted ensemble probability. Accepts a DataFrame; returns shape (n, 2)."""
        total_w = sum(self.weights.values()) or 1.0
        blended = np.zeros(len(X_df))
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0) / total_w
            if w == 0:
                continue
            blended += w * self._proba_one(name, model, X_df)
        return np.column_stack([1.0 - blended, blended])


class DailyPredictor:
    def __init__(self, dry_run: bool = False, fallback_v4: bool = False):
        self.bq = bigquery.Client(project=PROJECT)
        self.dry_run = dry_run
        self.fallback_v4 = fallback_v4
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = None
        self._is_v8 = False  # Set to True after load_model() if V8 is found
        self._ensure_table()

    # -----------------------------------------------------------------------
    # Table setup
    # -----------------------------------------------------------------------
    def _ensure_table(self) -> None:
        table_ref = bigquery.Table(PREDICTIONS_TABLE, schema=PREDICTIONS_SCHEMA)
        table_ref.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="game_date",
        )
        table_ref.clustering_fields = ["game_pk"]
        try:
            self.bq.get_table(PREDICTIONS_TABLE)
        except Exception:
            if not self.dry_run:
                self.bq.create_table(table_ref, exists_ok=True)
                logger.info("Created table %s", PREDICTIONS_TABLE)

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------
    def load_model(self) -> None:
        """
        Load best available model from local cache or GCS.
        Resolution chain: V8 → V7 → V6 → V5 → V4 (oldest fallback).

        V8 is the active production model as of 2026-04-08:
          - 57.65% overall accuracy (val 2025), best single model
          - 65.4% accuracy at ≥60% confidence threshold (11.7% of games)
          - Uses CatBoost with team ID embeddings + 85-feature V8 set
        """
        _versions = [
            ("v8", V8_LOCAL, V8_MODEL_GCS),
            ("v7", V7_LOCAL, V7_MODEL_GCS),
            ("v6", V6_LOCAL, V6_MODEL_GCS),
            ("v5", V5_LOCAL, V5_MODEL_GCS),
        ]
        data = None

        if not self.fallback_v4:
            for label, local_path, gcs_path in _versions:
                if local_path.exists():
                    logger.info("Loading %s model from local: %s", label, local_path)
                    with open(local_path, "rb") as f:
                        data = pickle.load(f)
                    break
                try:
                    logger.info("Attempting to download %s model from GCS...", label)
                    client = storage.Client(project=PROJECT)
                    bucket_obj = client.bucket(BUCKET)
                    blob = bucket_obj.blob(gcs_path)
                    data = pickle.loads(blob.download_as_bytes())
                    # Cache locally to avoid repeated GCS downloads
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(local_path, "wb") as f:
                        pickle.dump(data, f)
                    logger.info("%s model downloaded and cached locally", label)
                    break
                except Exception as e:
                    logger.warning("%s model unavailable: %s — trying next version", label, e)

        # V4 final fallback — always available
        if data is None:
            if V4_LOCAL.exists():
                logger.info("Falling back to V4 model: %s", V4_LOCAL)
                with open(V4_LOCAL, "rb") as f:
                    data = pickle.load(f)
            else:
                logger.info("Downloading V4 fallback model from GCS...")
                client = storage.Client(project=PROJECT)
                bucket_obj = client.bucket(BUCKET)
                blob = bucket_obj.blob(V4_MODEL_GCS)
                data = pickle.loads(blob.download_as_bytes())

        # V8 bundles use a different structure: multi-model ensemble with per-model
        # feature sets, scalers, and Optuna-tuned blend weights.
        if isinstance(data.get("models"), dict) and "weights" in data:
            ensemble = V8EnsemblePredictor(data)
            self.model = ensemble
            self.scaler = None          # each sub-model handles its own scaling
            self.feature_names = ensemble.all_features
            self.model_version = data.get("version", "v8_final")
            self._is_v8 = True
        else:
            self.model = data["model"]
            self.scaler = data.get("scaler")
            self.feature_names = data["features"]
            self.model_version = data.get("model_name", "v4_fallback")
            self._is_v8 = (
                self.model_version.startswith("V8")
                or "v8" in self.model_version.lower()
            )
        logger.info(
            "Model loaded: %s (%d features) | V8 mode: %s",
            self.model_version, len(self.feature_names), self._is_v8,
        )    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------
    def fetch_schedule(self, target_date: date, game_pks: Optional[list[int]] = None) -> list[dict]:
        """Fetch today's scheduled games from MLB API."""
        url = f"{MLB_API}/schedule"
        params = {
            "date": target_date.isoformat(),
            "sportId": 1,
            "hydrate": "team,probablePitcher,venue",
            "gameType": "R,F,D,L,W",
        }
        resp = requests.get(url, params=params, headers={"User-Agent": "HanksTank/2.0"},
                            timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()

        games = []
        for day in data.get("dates", []):
            for g in day.get("games", []):
                pk = g["gamePk"]
                if game_pks and pk not in game_pks:
                    continue
                if g.get("status", {}).get("abstractGameState") == "Final":
                    continue

                game_datetime_str = g.get("gameDate", "")
                try:
                    game_time_utc = datetime.fromisoformat(
                        game_datetime_str.replace("Z", "+00:00")
                    ).isoformat()
                except ValueError:
                    game_time_utc = None

                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})
                games.append({
                    "game_pk": pk,
                    "game_date": day["date"],
                    "game_time_utc": game_time_utc,
                    "home_team_id": home.get("team", {}).get("id"),
                    "home_team_name": home.get("team", {}).get("name", ""),
                    "away_team_id": away.get("team", {}).get("id"),
                    "away_team_name": away.get("team", {}).get("name", ""),
                    "home_probable_pitcher_id": home.get("probablePitcher", {}).get("id"),
                    "home_probable_pitcher_name": home.get("probablePitcher", {}).get("fullName", ""),
                    "away_probable_pitcher_id": away.get("probablePitcher", {}).get("id"),
                    "away_probable_pitcher_name": away.get("probablePitcher", {}).get("fullName", ""),
                    "venue_name": g.get("venue", {}).get("name", ""),
                })
        return games

    def load_game_features(self, game_pks: list[int]) -> pd.DataFrame:
        """Load team-level rolling features from game_features table."""
        pk_list = ", ".join(str(pk) for pk in game_pks)
        sql = f"""
            SELECT *
            FROM `{GAME_FEATURES_TABLE}`
            WHERE game_pk IN ({pk_list})
        """
        try:
            return self.bq.query(sql).to_dataframe()
        except Exception as e:
            logger.warning("Could not load game_features: %s", e)
            return pd.DataFrame()

    def load_matchup_features(self, game_pks: list[int]) -> pd.DataFrame:
        """Load pitcher/batter matchup features."""
        pk_list = ", ".join(str(pk) for pk in game_pks)
        sql = f"""
            SELECT *
            FROM `{MATCHUP_TABLE}`
            WHERE game_pk IN ({pk_list})
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY game_pk ORDER BY computed_at DESC
            ) = 1
        """
        try:
            return self.bq.query(sql).to_dataframe()
        except Exception as e:
            logger.warning("Could not load matchup_features: %s", e)
            return pd.DataFrame()

    def load_lineup_starters(self, game_pks: list[int]) -> pd.DataFrame:
        """Load starter pitcher names from lineups table."""
        pk_list = ", ".join(str(pk) for pk in game_pks)
        sql = f"""
            SELECT game_pk, team_type, player_id, player_name, pitch_hand
            FROM `{LINEUPS_TABLE}`
            WHERE game_pk IN ({pk_list})
              AND is_probable_pitcher = TRUE
        """
        try:
            return self.bq.query(sql).to_dataframe()
        except Exception as e:
            logger.debug("Could not load lineup starters: %s", e)
            return pd.DataFrame()

    def load_v7_features(self, game_pks: list[int]) -> pd.DataFrame:
        """Load V7 matchup features (bullpen health, moon phase, pitcher venue splits)."""
        pk_list = ", ".join(str(pk) for pk in game_pks)
        sql = f"""
            SELECT *
            FROM `{MATCHUP_V7_TABLE}`
            WHERE game_pk IN ({pk_list})
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY game_pk ORDER BY computed_at DESC
            ) = 1
        """
        try:
            return self.bq.query(sql).to_dataframe()
        except Exception as e:
            logger.debug("Could not load v7 features (table may not exist yet): %s", e)
            return pd.DataFrame()

    def load_v8_features(self, game_pks: list[int]) -> pd.DataFrame:
        """
        Load V8 features (Elo, Pythagorean, run differential, streaks, H2H).
        Populated by build_v8_features_live.py in the daily pipeline.

        Falls back gracefully to an empty DataFrame if the table does not yet
        contain rows for the requested games — the V8 model will use neutral
        imputed values in that case.
        """
        pk_list = ", ".join(str(pk) for pk in game_pks)
        sql = f"""
            SELECT *
            FROM `{V8_FEATURES_TABLE}`
            WHERE game_pk IN ({pk_list})
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY game_pk ORDER BY computed_at DESC
            ) = 1
        """
        try:
            return self.bq.query(sql).to_dataframe()
        except Exception as e:
            logger.warning(
                "V8 features unavailable (game_v8_features may not exist or backfill "
                "not yet run): %s — predictions will use neutral Elo/Pythagorean defaults",
                e,
            )
            return pd.DataFrame()

    # -----------------------------------------------------------------------
    # Feature assembly
    # -----------------------------------------------------------------------
    def assemble_features(
        self, game: dict, game_feat_row: Optional[pd.Series],
        matchup_row: Optional[pd.Series],
        v7_row: Optional[pd.Series] = None,
        v8_row: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Combine team-level, matchup, V7, and V8 features into the model input vector.
        Priority: V8 features override V3/V4 equivalents when both are present.
        Handles missing features gracefully with league-average defaults.
        """
        feat = {}

        # --- Team-level features (from game_features, same as V4) ---
        if game_feat_row is not None and not game_feat_row.empty:
            for col in game_feat_row.index:
                if col not in ("game_pk", "game_date", "home_team_id", "away_team_id"):
                    feat[col] = game_feat_row[col]
        else:
            # Provide safe defaults (model will use these when features missing)
            # These are approximate league-average values
            feat.update({
                "home_rolling_win_pct_10": 0.50,
                "away_rolling_win_pct_10": 0.50,
                "home_rolling_win_pct_30": 0.50,
                "away_rolling_win_pct_30": 0.50,
                "home_ema_form": 0.50,
                "away_ema_form": 0.50,
                "home_rest_days": 1,
                "away_rest_days": 1,
                "home_back_to_back": 0,
                "away_back_to_back": 0,
                "pitcher_quality_diff": 0.0,
                "park_run_factor_ratio": 1.0,
                "season_phase": 1,
                "is_divisional": 0,
            })

        # --- Matchup features (new V5 additions) ---
        if matchup_row is not None and not matchup_row.empty:
            matchup_cols = [
                "home_lineup_woba_vs_hand", "home_lineup_k_pct_vs_hand",
                "home_lineup_bb_pct_vs_hand", "home_lineup_iso_vs_hand",
                "home_top3_woba_vs_hand", "home_middle4_woba_vs_hand",
                "home_bottom2_woba_vs_hand", "home_pct_same_hand",
                "home_h2h_woba", "home_h2h_k_pct", "home_h2h_hr_rate",
                "away_lineup_woba_vs_hand", "away_lineup_k_pct_vs_hand",
                "away_lineup_bb_pct_vs_hand", "away_lineup_iso_vs_hand",
                "away_top3_woba_vs_hand", "away_middle4_woba_vs_hand",
                "away_bottom2_woba_vs_hand", "away_pct_same_hand",
                "away_h2h_woba", "away_h2h_k_pct", "away_h2h_hr_rate",
                "home_starter_woba_allowed", "home_starter_k_pct",
                "away_starter_woba_allowed", "away_starter_k_pct",
                "matchup_advantage_home",
            ]
            for col in matchup_cols:
                if col in matchup_row.index:
                    val = matchup_row[col]
                    feat[col] = float(val) if pd.notna(val) else None

            # Derived features
            home_woba = feat.get("home_lineup_woba_vs_hand")
            away_woba = feat.get("away_lineup_woba_vs_hand")
            if home_woba is not None and away_woba is not None:
                feat["lineup_woba_differential"] = home_woba - away_woba

            home_k = feat.get("home_lineup_k_pct_vs_hand")
            away_k = feat.get("away_lineup_k_pct_vs_hand")
            if home_k is not None and away_k is not None:
                feat["lineup_k_pct_differential"] = away_k - home_k  # more K for away = better for home pitcher

            home_starter_woba = feat.get("home_starter_woba_allowed")
            away_starter_woba = feat.get("away_starter_woba_allowed")
            if home_starter_woba is not None and away_starter_woba is not None:
                feat["starter_woba_differential"] = away_starter_woba - home_starter_woba  # higher = home pitcher better

            home_h2h = feat.get("home_h2h_woba")
            away_h2h = feat.get("away_h2h_woba")
            if home_h2h is not None and away_h2h is not None:
                feat["h2h_woba_differential"] = home_h2h - away_h2h

        else:
            # Fill matchup features with neutral values
            neutral_matchup = {
                "home_lineup_woba_vs_hand": 0.320,
                "away_lineup_woba_vs_hand": 0.320,
                "home_starter_woba_allowed": 0.320,
                "away_starter_woba_allowed": 0.320,
                "matchup_advantage_home": 0.0,
                "lineup_woba_differential": 0.0,
                "lineup_k_pct_differential": 0.0,
                "starter_woba_differential": 0.0,
                "h2h_woba_differential": 0.0,
            }
            feat.update(neutral_matchup)

        # V7 features (bullpen health, moon phase, pitcher venue splits)
        if v7_row is not None and not v7_row.empty:
            skip_cols = {"game_pk", "game_date", "home_team_id", "away_team_id", "computed_at"}
            for col in v7_row.index:
                if col not in skip_cols:
                    val = v7_row[col]
                    feat[col] = float(val) if pd.notna(val) else None

        # V8 features (Elo, Pythagorean, run differential, streaks, H2H)
        # These override any V3/V4 equivalents (e.g., V8 win_pct_season supersedes
        # V3's rolling_win_pct_30 since V8 uses the actual game history directly).
        if v8_row is not None and not v8_row.empty:
            skip_cols = {
                "game_pk", "game_date", "home_team_id", "away_team_id",
                "computed_at", "data_completeness",
            }
            for col in v8_row.index:
                if col not in skip_cols:
                    val = v8_row[col]
                    if pd.notna(val):
                        feat[col] = float(val) if not isinstance(val, (bool, np.bool_)) else int(val)
        else:
            # V8 features not yet available — use neutral (league-average) defaults.
            # The model handles these gracefully via median imputation.
            # NOTE: Elo defaults to 0.5 win probability (no information) — predictions
            # at these values will be lower confidence and correctly classified as "low" tier.
            feat.update({
                "home_elo": 1500.0, "away_elo": 1500.0, "elo_differential": 0.0,
                "elo_home_win_prob": 0.532, "elo_win_prob_differential": 0.032,
                "home_pythag_season": 0.5, "away_pythag_season": 0.5,
                "home_pythag_last30": 0.5, "away_pythag_last30": 0.5,
                "pythag_differential": 0.0, "home_luck_factor": 0.0,
                "away_luck_factor": 0.0, "luck_differential": 0.0,
                "home_run_diff_10g": 0.0, "away_run_diff_10g": 0.0,
                "home_run_diff_30g": 0.0, "away_run_diff_30g": 0.0,
                "run_diff_differential": 0.0,
                "home_era_proxy_10g": 4.5, "away_era_proxy_10g": 4.5,
                "home_era_proxy_30g": 4.5, "away_era_proxy_30g": 4.5,
                "era_proxy_differential": 0.0,
                "home_win_pct_season": 0.5, "away_win_pct_season": 0.5,
                "home_scoring_momentum": 0.0, "away_scoring_momentum": 0.0,
                "home_current_streak": 0, "away_current_streak": 0,
                "home_win_pct_7g": 0.5, "away_win_pct_7g": 0.5,
                "home_win_pct_14g": 0.5, "away_win_pct_14g": 0.5,
                "streak_differential": 0,
                "home_on_winning_streak": 0, "away_on_winning_streak": 0,
                "home_on_losing_streak": 0, "away_on_losing_streak": 0,
                "home_streak_direction": 0, "away_streak_direction": 0,
                "h2h_win_pct_season": 0.5, "h2h_win_pct_3yr": 0.5,
                "h2h_advantage_season": 0.0, "h2h_advantage_3yr": 0.0,
                "h2h_games_3yr": 0, "is_divisional": 0,
                "season_pct_complete": 0.1, "season_stage": 0,
                "home_games_played_season": 5,
                "season_stage_late": 0, "season_stage_early": 1,
            })

        # Build final feature vector aligned with model's expected features
        row = {}
        for fname in self.feature_names:
            val = feat.get(fname)
            row[fname] = float(val) if val is not None else np.nan

        return pd.DataFrame([row])

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------
    def predict_game(
        self, game: dict, feat_df: pd.DataFrame,
        matchup_row: Optional[pd.Series] = None
    ) -> dict:
        """Generate a prediction for one game."""
        # Impute NaN with column medians (same as training)
        X = feat_df.copy()
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(0.0)

        # V8EnsemblePredictor handles its own feature selection and scaling
        # internally and expects a DataFrame; older models expect a numpy array.
        if isinstance(self.model, V8EnsemblePredictor):
            proba = self.model.predict_proba(X)[0]
        elif self.scaler:
            proba = self.model.predict_proba(self.scaler.transform(X))[0]
        else:
            proba = self.model.predict_proba(X.values)[0]

        # proba[1] = home win probability
        home_win_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        away_win_prob = 1.0 - home_win_prob

        predicted_winner = (
            game.get("home_team_name", "Home")
            if home_win_prob > 0.5
            else game.get("away_team_name", "Away")
        )

        max_prob = max(home_win_prob, away_win_prob)
        confidence_tier = "low"
        if max_prob >= CONFIDENCE_TIERS["high"]:
            confidence_tier = "high"
        elif max_prob >= CONFIDENCE_TIERS["medium"]:
            confidence_tier = "medium"

        return {
            "home_win_probability": round(home_win_prob, 4),
            "away_win_probability": round(away_win_prob, 4),
            "predicted_winner": predicted_winner,
            "confidence_tier": confidence_tier,
        }

    # -----------------------------------------------------------------------
    # Orchestrate
    # -----------------------------------------------------------------------
    def run_for_date(self, target_date: date) -> dict:
        """Predict all games on a given date."""
        games = self.fetch_schedule(target_date)
        if not games:
            logger.info("No upcoming games on %s", target_date)
            return {"games_predicted": 0}
        return self._run(games, target_date)

    def run_for_game_pks(
        self, game_pks: list[int], game_date: Optional[date] = None
    ) -> dict:
        target_date = game_date or date.today()
        games = self.fetch_schedule(target_date, game_pks)
        return self._run(games, target_date)

    def _run(self, games: list[dict], target_date: date) -> dict:
        if self.model is None:
            self.load_model()

        game_pks = [g["game_pk"] for g in games]
        game_features_df = self.load_game_features(game_pks)
        matchup_df = self.load_matchup_features(game_pks)
        v7_df = self.load_v7_features(game_pks)
        v8_df = self.load_v8_features(game_pks) if self._is_v8 else pd.DataFrame()
        starter_df = self.load_lineup_starters(game_pks)

        pred_rows = []
        for game in games:
            pk = game["game_pk"]

            # Game-level team features
            gf_row = None
            if not game_features_df.empty and "game_pk" in game_features_df.columns:
                gf_subset = game_features_df[game_features_df["game_pk"] == pk]
                if not gf_subset.empty:
                    gf_row = gf_subset.iloc[0]

            # Matchup features
            mf_row = None
            if not matchup_df.empty and "game_pk" in matchup_df.columns:
                mf_subset = matchup_df[matchup_df["game_pk"] == pk]
                if not mf_subset.empty:
                    mf_row = mf_subset.iloc[0]

            # V7 features
            v7_row = None
            if not v7_df.empty and "game_pk" in v7_df.columns:
                v7_subset = v7_df[v7_df["game_pk"] == pk]
                if not v7_subset.empty:
                    v7_row = v7_subset.iloc[0]

            # V8 features (Elo, Pythagorean, run differential, streaks, H2H)
            v8_row = None
            if not v8_df.empty and "game_pk" in v8_df.columns:
                v8_subset = v8_df[v8_df["game_pk"] == pk]
                if not v8_subset.empty:
                    v8_row = v8_subset.iloc[0]

            feat_df = self.assemble_features(game, gf_row, mf_row, v7_row, v8_row)
            pred = self.predict_game(game, feat_df, mf_row)

            # Starter info from lineup table (prefer over schedule)
            home_starter_id = game.get("home_probable_pitcher_id")
            home_starter_name = game.get("home_probable_pitcher_name", "")
            away_starter_id = game.get("away_probable_pitcher_id")
            away_starter_name = game.get("away_probable_pitcher_name", "")
            home_starter_hand = None
            away_starter_hand = None

            if not starter_df.empty:
                home_row = starter_df[
                    (starter_df["game_pk"] == pk) & (starter_df["team_type"] == "home")
                ]
                away_row = starter_df[
                    (starter_df["game_pk"] == pk) & (starter_df["team_type"] == "away")
                ]
                if not home_row.empty:
                    home_starter_id = int(home_row["player_id"].iloc[0])
                    home_starter_name = home_row["player_name"].iloc[0]
                    home_starter_hand = str(home_row["pitch_hand"].iloc[0]) if pd.notna(home_row["pitch_hand"].iloc[0]) else None
                if not away_row.empty:
                    away_starter_id = int(away_row["player_id"].iloc[0])
                    away_starter_name = away_row["player_name"].iloc[0]
                    away_starter_hand = str(away_row["pitch_hand"].iloc[0]) if pd.notna(away_row["pitch_hand"].iloc[0]) else None

            if mf_row is not None and not mf_row.empty:
                home_starter_hand = home_starter_hand or str(mf_row.get("home_starter_hand", ""))
                away_starter_hand = away_starter_hand or str(mf_row.get("away_starter_hand", ""))

            lineup_confirmed = bool(
                mf_row.get("lineup_confirmed") if mf_row is not None else False
            )
            matchup_advantage = float(
                mf_row.get("matchup_advantage_home", 0.0)
            ) if mf_row is not None else 0.0
            home_woba = float(
                mf_row.get("home_lineup_woba_vs_hand")
            ) if mf_row is not None and pd.notna(mf_row.get("home_lineup_woba_vs_hand")) else None
            away_woba = float(
                mf_row.get("away_lineup_woba_vs_hand")
            ) if mf_row is not None and pd.notna(mf_row.get("away_lineup_woba_vs_hand")) else None
            h2h_available = bool(
                mf_row.get("home_h2h_pa_total", 0) > 0
                or mf_row.get("away_h2h_pa_total", 0) > 0
            ) if mf_row is not None else False

            pred_row = {
                "game_pk": pk,
                "game_date": game["game_date"],
                "home_team_id": game.get("home_team_id"),
                "home_team_name": game.get("home_team_name", ""),
                "away_team_id": game.get("away_team_id"),
                "away_team_name": game.get("away_team_name", ""),
                "home_starter_id": home_starter_id,
                "home_starter_name": home_starter_name,
                "away_starter_id": away_starter_id,
                "away_starter_name": away_starter_name,
                "home_win_probability": pred["home_win_probability"],
                "away_win_probability": pred["away_win_probability"],
                "predicted_winner": pred["predicted_winner"],
                "confidence_tier": pred["confidence_tier"],
                "model_version": self.model_version,
                "lineup_confirmed": lineup_confirmed,
                "matchup_advantage_home": matchup_advantage,
                "home_lineup_woba_vs_hand": home_woba,
                "away_lineup_woba_vs_hand": away_woba,
                "home_starter_hand": home_starter_hand,
                "away_starter_hand": away_starter_hand,
                "h2h_data_available": h2h_available,
                "game_time_utc": game.get("game_time_utc"),
                "predicted_at": datetime.now(tz=timezone.utc).isoformat(),
                # V7 pitcher arsenal (from matchup_v7_features)
                "home_starter_mean_velo": float(v7_row.get("home_starter_mean_velo")) if v7_row is not None and pd.notna(v7_row.get("home_starter_mean_velo")) else None,
                "away_starter_mean_velo": float(v7_row.get("away_starter_mean_velo")) if v7_row is not None and pd.notna(v7_row.get("away_starter_mean_velo")) else None,
                "home_starter_velo_norm": float(v7_row.get("home_starter_velo_norm")) if v7_row is not None and pd.notna(v7_row.get("home_starter_velo_norm")) else None,
                "away_starter_velo_norm": float(v7_row.get("away_starter_velo_norm")) if v7_row is not None and pd.notna(v7_row.get("away_starter_velo_norm")) else None,
                "home_starter_k_bb_pct": float(v7_row.get("home_starter_k_bb_pct")) if v7_row is not None and pd.notna(v7_row.get("home_starter_k_bb_pct")) else None,
                "away_starter_k_bb_pct": float(v7_row.get("away_starter_k_bb_pct")) if v7_row is not None and pd.notna(v7_row.get("away_starter_k_bb_pct")) else None,
                "home_starter_xwoba_allowed": float(v7_row.get("home_starter_xwoba_allowed")) if v7_row is not None and pd.notna(v7_row.get("home_starter_xwoba_allowed")) else None,
                "away_starter_xwoba_allowed": float(v7_row.get("away_starter_xwoba_allowed")) if v7_row is not None and pd.notna(v7_row.get("away_starter_xwoba_allowed")) else None,
                "starter_arsenal_advantage": float(v7_row.get("starter_arsenal_advantage")) if v7_row is not None and pd.notna(v7_row.get("starter_arsenal_advantage")) else None,
                # V7 bullpen health
                "home_bullpen_fatigue_score": float(v7_row.get("home_bullpen_fatigue_score")) if v7_row is not None and pd.notna(v7_row.get("home_bullpen_fatigue_score")) else None,
                "away_bullpen_fatigue_score": float(v7_row.get("away_bullpen_fatigue_score")) if v7_row is not None and pd.notna(v7_row.get("away_bullpen_fatigue_score")) else None,
                "bullpen_fatigue_differential": float(v7_row.get("bullpen_fatigue_differential")) if v7_row is not None and pd.notna(v7_row.get("bullpen_fatigue_differential")) else None,
                "home_closer_days_rest": float(v7_row.get("home_closer_days_rest")) if v7_row is not None and pd.notna(v7_row.get("home_closer_days_rest")) else None,
                "away_closer_days_rest": float(v7_row.get("away_closer_days_rest")) if v7_row is not None and pd.notna(v7_row.get("away_closer_days_rest")) else None,
                # V7 moon phase
                "moon_illumination": float(v7_row.get("moon_illumination")) if v7_row is not None and pd.notna(v7_row.get("moon_illumination")) else None,
                "is_full_moon": int(v7_row.get("is_full_moon")) if v7_row is not None and pd.notna(v7_row.get("is_full_moon")) else None,
                # V7 pitcher venue splits
                "home_starter_venue_era": float(v7_row.get("home_starter_venue_era")) if v7_row is not None and pd.notna(v7_row.get("home_starter_venue_era")) else None,
                "away_starter_venue_era": float(v7_row.get("away_starter_venue_era")) if v7_row is not None and pd.notna(v7_row.get("away_starter_venue_era")) else None,
                "starter_venue_era_differential": float(v7_row.get("starter_venue_era_differential")) if v7_row is not None and pd.notna(v7_row.get("starter_venue_era_differential")) else None,
                # V7 venue wOBA
                "venue_woba_differential": float(v7_row.get("venue_woba_differential")) if v7_row is not None and pd.notna(v7_row.get("venue_woba_differential")) else None,
            }
            pred_rows.append(pred_row)
            logger.info(
                "Game %s: %s %.1f%% vs %s %.1f%% [%s] (lineup=%s, matchup=%.3f)",
                pk,
                pred_row["home_team_name"], pred["home_win_probability"] * 100,
                pred_row["away_team_name"], pred["away_win_probability"] * 100,
                pred["confidence_tier"],
                lineup_confirmed,
                matchup_advantage,
            )

        if pred_rows and not self.dry_run:
            pk_list = ", ".join(str(r["game_pk"]) for r in pred_rows)
            # Delete prior predictions for these game PKs (idempotent upsert).
            # Falls back to insert-only when the streaming buffer blocks DML.
            try:
                self.bq.query(
                    f"DELETE FROM `{PREDICTIONS_TABLE}` "
                    f"WHERE game_pk IN ({pk_list}) "
                    f"AND DATE(predicted_at) = '{target_date.isoformat()}'"
                ).result()
            except Exception as exc:
                if "streaming buffer" in str(exc).lower():
                    logger.warning(
                        "DELETE skipped (streaming buffer active) — inserting new predictions alongside existing"
                    )
                else:
                    raise
            errors = self.bq.insert_rows_json(PREDICTIONS_TABLE, pred_rows)
            if errors:
                logger.error("BQ insert errors: %s", errors)
            else:
                logger.info("Wrote %d predictions to BigQuery", len(pred_rows))
        elif pred_rows and self.dry_run:
            logger.info("[DRY RUN] Would write %d predictions", len(pred_rows))

        return {
            "games_predicted": len(pred_rows),
            "model_version": self.model_version,
            "lineup_confirmed_count": sum(1 for r in pred_rows if r["lineup_confirmed"]),
            "h2h_data_count": sum(1 for r in pred_rows if r["h2h_data_available"]),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict today's MLB games with matchup features")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--game-pk", help="Comma-separated game PKs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fallback-v4", action="store_true",
                        help="Force use of V4 model (before V5 is trained)")
    args = parser.parse_args()

    predictor = DailyPredictor(dry_run=args.dry_run, fallback_v4=args.fallback_v4)

    if args.game_pk:
        pks = [int(pk.strip()) for pk in args.game_pk.split(",")]
        target = date.fromisoformat(args.date) if args.date else date.today()
        result = predictor.run_for_game_pks(pks, target)
    else:
        target = date.fromisoformat(args.date) if args.date else date.today()
        result = predictor.run_for_date(target)

    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()
