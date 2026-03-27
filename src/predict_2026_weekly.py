#!/usr/bin/env python3
"""
Weekly Batch Prediction Pipeline

Runs every Friday to predict the next week of games (Friday → Thursday).
Loads the trained model, fetches scheduled games from the MLB API,
computes live features, generates predictions with confidence levels,
and writes results to BigQuery.

Output table: mlb_2026_season.weekly_predictions

Usage:
    python predict_2026_weekly.py                  # next Fri-Thu
    python predict_2026_weekly.py --start 2026-03-27 --end 2026-04-02
    python predict_2026_weekly.py --dry-run
"""

import argparse
import logging
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3
from google.cloud import bigquery, storage
from sklearn.preprocessing import StandardScaler

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
DATASET = "mlb_2026_season"
BUCKET = "hanks_tank_data"
MODEL_GCS_PATH = "models/vertex/game_outcome_2026/model.pkl"
LOCAL_MODEL_PATH = Path("models/game_outcome_2026_vertex.pkl")
MLB_API = "https://statsapi.mlb.com/api/v1"
PREDICTIONS_TABLE = f"{PROJECT}.{DATASET}.weekly_predictions"
SEASON = 2026

CONFIDENCE_TIERS = {
    "high": 0.60,
    "medium": 0.55,
    "low": 0.50,
}


class WeeklyPredictor:
    def __init__(self, dry_run: bool = False):
        self.bq = bigquery.Client(project=PROJECT)
        self.dry_run = dry_run
        self.model = None
        self.scaler = None
        self.features = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self):
        """Load model from local file or GCS."""
        if LOCAL_MODEL_PATH.exists():
            logger.info("Loading model from %s", LOCAL_MODEL_PATH)
            with open(LOCAL_MODEL_PATH, "rb") as f:
                data = pickle.load(f)
        else:
            logger.info("Downloading model from GCS...")
            client = storage.Client(project=PROJECT)
            bucket = client.bucket(BUCKET)
            blob = bucket.blob(MODEL_GCS_PATH)
            pkl_bytes = blob.download_as_bytes()
            data = pickle.loads(pkl_bytes)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.features = data["features"]
        logger.info("Model loaded: %s (%d features)",
                     data.get("model_name", "unknown"), len(self.features))

    # ------------------------------------------------------------------
    # Fetch upcoming schedule
    # ------------------------------------------------------------------
    def fetch_schedule(self, start: date, end: date) -> pd.DataFrame:
        logger.info("Fetching schedule %s → %s", start, end)
        resp = requests.get(
            f"{MLB_API}/schedule",
            params={
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "sportId": 1,
                "hydrate": "team,probablePitcher,venue",
            },
            headers={"User-Agent": "HanksTank/2.0"},
            timeout=30,
            verify=False,
        )
        resp.raise_for_status()
        data = resp.json()

        rows = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})
                rows.append({
                    "game_pk": g["gamePk"],
                    "game_date": d["date"],
                    "game_type": g.get("gameType", "R"),
                    "home_team_id": home.get("team", {}).get("id"),
                    "home_team_name": home.get("team", {}).get("name"),
                    "away_team_id": away.get("team", {}).get("id"),
                    "away_team_name": away.get("team", {}).get("name"),
                    "venue_name": g.get("venue", {}).get("name"),
                    "home_probable_pitcher": home.get("probablePitcher", {}).get("fullName"),
                    "away_probable_pitcher": away.get("probablePitcher", {}).get("fullName"),
                })
        df = pd.DataFrame(rows)
        logger.info("Found %d scheduled games", len(df))
        return df

    # ------------------------------------------------------------------
    # Compute live features for upcoming games
    # ------------------------------------------------------------------
    def compute_features(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Compute features for each upcoming game using latest 2026 data."""
        # Load rolling context from BQ
        all_games = self._load_recent_games()
        team_stats = self._load_latest_team_stats()
        standings = self._load_latest_standings()

        standings_wpct = {}
        if not standings.empty:
            for _, r in standings.iterrows():
                standings_wpct[int(r["team_id"])] = float(r.get("win_percentage", 0.5) or 0.5)

        feature_rows = []
        for _, g in schedule.iterrows():
            gd = pd.Timestamp(g["game_date"]).date()
            hid = g["home_team_id"]
            aid = g["away_team_id"]

            h_wp10 = self._rolling_wp(all_games, hid, gd, 10)
            a_wp10 = self._rolling_wp(all_games, aid, gd, 10)
            h_wp30 = self._rolling_wp(all_games, hid, gd, 30)
            a_wp30 = self._rolling_wp(all_games, aid, gd, 30)
            h_ema = self._ema(all_games, hid, gd)
            a_ema = self._ema(all_games, aid, gd)

            h_rest = self._rest(all_games, hid, gd)
            a_rest = self._rest(all_games, aid, gd)
            h_b2b = 1 if h_rest == 0 else 0

            h_era = self._team_era(team_stats, hid)
            a_era = self._team_era(team_stats, aid)
            h_pq = max(0, 1 - h_era / 10) if h_era else 0.5
            a_pq = max(0, 1 - a_era / 10) if a_era else 0.5

            month = gd.month
            dow = gd.weekday()

            h_div = standings.loc[standings["team_id"] == hid, "division_id"]
            a_div = standings.loc[standings["team_id"] == aid, "division_id"]
            is_div = 1 if (not h_div.empty and not a_div.empty and
                           int(h_div.iloc[0]) == int(a_div.iloc[0])) else 0

            feature_rows.append({
                "home_win_pct_10d": h_wp10,
                "away_win_pct_10d": a_wp10,
                "home_win_pct_30d": h_wp30,
                "away_win_pct_30d": a_wp30,
                "home_ema_form": h_ema,
                "away_ema_form": a_ema,
                "home_pitcher_quality": h_pq,
                "away_pitcher_quality": a_pq,
                "home_days_rest": h_rest,
                "away_days_rest": a_rest,
                "home_rest_advantage": h_rest - a_rest,
                "is_back_to_back": h_b2b,
                "month": month,
                "day_of_week": dow,
                "is_early_season": 1 if month <= 4 else 0,
                "is_late_season": 1 if month >= 9 else 0,
                "is_home": 1,
                "is_divisional_matchup": is_div,
                "home_form_squared": h_ema ** 2,
                "away_form_squared": a_ema ** 2,
                "rest_balance": (h_rest - a_rest) / 5.0,
                "home_momentum": h_wp10 - h_ema,
                "away_momentum": a_wp10 - a_ema,
                "pitcher_quality_diff": h_pq - a_pq,
                "fatigue_index": h_b2b * 2,
                "win_pct_diff": h_wp30 - a_wp30,
                "home_composite_strength": 0.4 * h_wp10 + 0.3 * h_ema + 0.2 * h_pq + 0.1 * h_wp30,
                "away_composite_strength": 0.4 * a_wp10 + 0.3 * a_ema + 0.2 * a_pq + 0.1 * a_wp30,
            })

        return pd.DataFrame(feature_rows)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, schedule: pd.DataFrame,
                features_df: pd.DataFrame) -> pd.DataFrame:
        X = features_df[self.features].fillna(0).values.astype(np.float32)
        X_scaled = self.scaler.transform(X)

        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)

        results = schedule.copy()
        results["home_win_probability"] = np.round(probs, 4)
        results["away_win_probability"] = np.round(1 - probs, 4)
        results["predicted_winner"] = np.where(
            preds == 1,
            results["home_team_name"],
            results["away_team_name"],
        )
        results["prediction_confidence"] = np.round(
            np.maximum(probs, 1 - probs), 4
        )
        results["confidence_tier"] = results["prediction_confidence"].apply(
            self._tier
        )
        results["prediction_date"] = date.today().isoformat()
        results["model_version"] = "xgb_2026_v1"

        return results

    @staticmethod
    def _tier(conf: float) -> str:
        if conf >= CONFIDENCE_TIERS["high"]:
            return "HIGH"
        if conf >= CONFIDENCE_TIERS["medium"]:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_predictions(self, df: pd.DataFrame):
        if df.empty:
            return
        if self.dry_run:
            logger.info("[DRY RUN] %d predictions", len(df))
            for _, r in df.iterrows():
                logger.info(
                    "  %s: %s @ %s → %s (%.1f%% %s)",
                    r["game_date"], r["away_team_name"], r["home_team_name"],
                    r["predicted_winner"],
                    r["prediction_confidence"] * 100,
                    r["confidence_tier"],
                )
            return

        # Date columns
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["prediction_date"] = pd.to_datetime(df["prediction_date"])

        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            autodetect=True,
        )
        job = self.bq.load_table_from_dataframe(df, PREDICTIONS_TABLE,
                                                  job_config=job_config)
        job.result()
        logger.info("✓ Saved %d predictions to %s", job.output_rows, PREDICTIONS_TABLE)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        self.load_model()

        if start is None:
            # Next Friday through Thursday
            today = date.today()
            days_until_friday = (4 - today.weekday()) % 7
            start = today + timedelta(days=days_until_friday)
            end = start + timedelta(days=6)

        schedule = self.fetch_schedule(start, end)
        if schedule.empty:
            logger.info("No games scheduled for %s → %s", start, end)
            return pd.DataFrame()

        features = self.compute_features(schedule)
        predictions = self.predict(schedule, features)
        self.save_predictions(predictions)

        # Summary
        logger.info("=" * 60)
        logger.info("WEEKLY PREDICTIONS: %s → %s", start, end)
        logger.info("=" * 60)
        for tier in ["HIGH", "MEDIUM", "LOW"]:
            count = (predictions["confidence_tier"] == tier).sum()
            logger.info("  %s confidence: %d games", tier, count)
        logger.info("  Total: %d games", len(predictions))

        return predictions

    # ------------------------------------------------------------------
    # Helper: data loaders for feature computation
    # ------------------------------------------------------------------
    def _load_recent_games(self) -> pd.DataFrame:
        q = f"""
        (SELECT game_pk, game_date, home_team_id, away_team_id,
                home_score, away_score
         FROM `{PROJECT}.{DATASET}.games`
         WHERE status IN ('Final', 'Completed Early'))
        UNION ALL
        (SELECT game_pk, game_date, home_team_id, away_team_id,
                CAST(home_score AS INT64) AS home_score,
                CAST(away_score AS INT64) AS away_score
         FROM `{PROJECT}.mlb_historical_data.games_historical`
         WHERE year = 2025 AND status_code = 'F')
        ORDER BY game_date
        """
        df = self.bq.query(q).to_dataframe()
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        for c in ["home_score", "away_score"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        return df

    def _load_latest_team_stats(self) -> pd.DataFrame:
        q = f"""
        SELECT * EXCEPT(rn) FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY team_id, stat_type ORDER BY snapshot_date DESC
            ) rn FROM `{PROJECT}.{DATASET}.team_stats`
        ) WHERE rn = 1
        """
        try:
            return self.bq.query(q).to_dataframe()
        except Exception:
            return pd.DataFrame()

    def _load_latest_standings(self) -> pd.DataFrame:
        q = f"""
        SELECT * EXCEPT(rn) FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY team_id ORDER BY snapshot_date DESC
            ) rn FROM `{PROJECT}.{DATASET}.standings`
        ) WHERE rn = 1
        """
        try:
            return self.bq.query(q).to_dataframe()
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _rolling_wp(games: pd.DataFrame, team_id: int,
                    before: date, window: int) -> float:
        mask = (
            ((games["home_team_id"] == team_id) |
             (games["away_team_id"] == team_id)) &
            (games["game_date"] < before)
        )
        recent = games.loc[mask].tail(window)
        if recent.empty:
            return 0.5
        wins = (
            ((recent["home_team_id"] == team_id) &
             (recent["home_score"] > recent["away_score"])) |
            ((recent["away_team_id"] == team_id) &
             (recent["away_score"] > recent["home_score"]))
        ).sum()
        return wins / len(recent)

    @staticmethod
    def _ema(games: pd.DataFrame, team_id: int, before: date,
             span: int = 10) -> float:
        mask = (
            ((games["home_team_id"] == team_id) |
             (games["away_team_id"] == team_id)) &
            (games["game_date"] < before)
        )
        recent = games.loc[mask].tail(30)
        if recent.empty:
            return 0.5
        outcomes = (
            ((recent["home_team_id"] == team_id) &
             (recent["home_score"] > recent["away_score"])) |
            ((recent["away_team_id"] == team_id) &
             (recent["away_score"] > recent["home_score"]))
        ).astype(float)
        return float(outcomes.ewm(span=span, min_periods=1).mean().iloc[-1])

    @staticmethod
    def _rest(games: pd.DataFrame, team_id: int, game_date: date) -> int:
        mask = (
            ((games["home_team_id"] == team_id) |
             (games["away_team_id"] == team_id)) &
            (games["game_date"] < game_date)
        )
        prev = games.loc[mask]
        if prev.empty:
            return 3
        last = pd.Timestamp(prev["game_date"].iloc[-1])
        return max(int((pd.Timestamp(game_date) - last).days), 0)

    @staticmethod
    def _team_era(team_stats: pd.DataFrame, team_id: int) -> float:
        if team_stats.empty:
            return 4.5
        row = team_stats.loc[
            (team_stats["team_id"] == team_id) &
            (team_stats["stat_type"] == "pitching")
        ]
        if row.empty:
            return 4.5
        return float(row.iloc[0].get("era", 4.5) or 4.5)


def main():
    parser = argparse.ArgumentParser(description="Weekly MLB batch predictions")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    predictor = WeeklyPredictor(dry_run=args.dry_run)
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    predictor.run(start, end)


if __name__ == "__main__":
    main()
