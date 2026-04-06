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
PREDICTIONS_TABLE    = f"{PROJECT}.{DATASET}.game_predictions"
LINEUPS_TABLE        = f"{PROJECT}.{DATASET}.lineups"

# Model GCS paths — predictor tries V7 → V6 → V5 → V4
V7_MODEL_GCS = "models/vertex/game_outcome_2026_v7/model.pkl"
V6_MODEL_GCS = "models/vertex/game_outcome_2026_v6/model.pkl"
V5_MODEL_GCS = "models/vertex/game_outcome_2026_v5/model.pkl"
V4_MODEL_GCS = "models/vertex/game_outcome_2026/model.pkl"
V7_LOCAL = Path("models/game_outcome_2026_v7.pkl")
V6_LOCAL = Path("models/game_outcome_2026_v6.pkl")
V5_LOCAL = Path("models/game_outcome_2026_v5.pkl")
V4_LOCAL = Path("models/game_outcome_2026_vertex.pkl")

CONFIDENCE_TIERS = {"high": 0.62, "medium": 0.57, "low": 0.53}

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
]


class DailyPredictor:
    def __init__(self, dry_run: bool = False, fallback_v4: bool = False):
        self.bq = bigquery.Client(project=PROJECT)
        self.dry_run = dry_run
        self.fallback_v4 = fallback_v4
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = None
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
        """Try V7 → V6 → V5 → V4 in order, using local cache when available."""
        _versions = [
            ("v7", V7_LOCAL, V7_MODEL_GCS),
            ("v6", V6_LOCAL, V6_MODEL_GCS),
            ("v5", V5_LOCAL, V5_MODEL_GCS),
        ]
        data = None

        if not self.fallback_v4:
            for label, local_path, gcs_path in _versions:
                if local_path.exists():
                    logger.info("Loading %s model from %s", label, local_path)
                    with open(local_path, "rb") as f:
                        data = pickle.load(f)
                    break
                try:
                    logger.info("Downloading %s model from GCS...", label)
                    client = storage.Client(project=PROJECT)
                    bucket = client.bucket(BUCKET)
                    blob = bucket.blob(gcs_path)
                    data = pickle.loads(blob.download_as_bytes())
                    break
                except Exception as e:
                    logger.warning("%s model not available: %s — trying next version", label, e)

        # V4 fallback
        if data is None:
            if V4_LOCAL.exists():
                logger.info("Loading V4 model from %s", V4_LOCAL)
                with open(V4_LOCAL, "rb") as f:
                    data = pickle.load(f)
            else:
                logger.info("Downloading V4 model from GCS...")
                client = storage.Client(project=PROJECT)
                bucket = client.bucket(BUCKET)
                blob = bucket.blob(V4_MODEL_GCS)
                data = pickle.loads(blob.download_as_bytes())

        self.model = data["model"]
        self.scaler = data.get("scaler")
        self.feature_names = data["features"]
        self.model_version = data.get("model_name", "v4_fallback")
        logger.info("Model loaded: %s (%d features)", self.model_version, len(self.feature_names))

    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Feature assembly
    # -----------------------------------------------------------------------
    def assemble_features(
        self, game: dict, game_feat_row: Optional[pd.Series],
        matchup_row: Optional[pd.Series],
        v7_row: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Combine team-level and matchup features into the model input vector.
        Handles missing features gracefully with median imputation placeholders.
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

        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        proba = self.model.predict_proba(X_scaled)[0]
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

            feat_df = self.assemble_features(game, gf_row, mf_row, v7_row)
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
            self.bq.query(
                f"DELETE FROM `{PREDICTIONS_TABLE}` "
                f"WHERE game_pk IN ({pk_list}) "
                f"AND DATE(predicted_at) = '{target_date.isoformat()}'"
            ).result()
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
