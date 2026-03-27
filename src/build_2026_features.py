#!/usr/bin/env python3
"""
Feature Engineering Pipeline for 2026 Season

Reads raw data from mlb_2026_season, computes the v3-style feature set for
every completed game, and writes the result to mlb_2026_season.game_features.

Features computed (matching v3 model expectations):
  - Rolling win % (10-day, 30-day)
  - EMA form
  - Pitcher quality proxy (team ERA / K9 rankings)
  - Rest days & back-to-back flags
  - Park factor ratio
  - Momentum / trend
  - Season-phase interaction
  - Composite strength
  - Home indicator + temporal

Usage:
    python build_2026_features.py                      # all games so far
    python build_2026_features.py --since 2026-03-20   # from a specific date
    python build_2026_features.py --dry-run
"""

import argparse
import logging
from datetime import date, timedelta
from typing import Dict

import numpy as np
import pandas as pd
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
DATASET = "mlb_2026_season"
FEATURES_TABLE = f"{PROJECT}.{DATASET}.game_features"


class FeatureBuilder:
    """Build game-level ML features from the 2026 raw tables."""

    def __init__(self, dry_run: bool = False):
        self.bq = bigquery.Client(project=PROJECT)
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_games(self, since: date | None = None) -> pd.DataFrame:
        where = ""
        if since:
            where = f"AND game_date >= '{since.isoformat()}'"
        q = f"""
        SELECT *
        FROM `{PROJECT}.{DATASET}.games`
        WHERE status IN ('Final', 'Completed Early')
          {where}
        ORDER BY game_date, game_pk
        """
        df = self.bq.query(q).to_dataframe()
        logger.info("Loaded %d completed games", len(df))
        return df

    def load_team_stats_latest(self) -> pd.DataFrame:
        """Latest team stats snapshot (most recent snapshot_date per team)."""
        q = f"""
        SELECT * EXCEPT(rn) FROM (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY team_id, stat_type
                                    ORDER BY snapshot_date DESC) rn
            FROM `{PROJECT}.{DATASET}.team_stats`
        ) WHERE rn = 1
        """
        return self.bq.query(q).to_dataframe()

    def load_standings_latest(self) -> pd.DataFrame:
        q = f"""
        SELECT * EXCEPT(rn) FROM (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY team_id
                                    ORDER BY snapshot_date DESC) rn
            FROM `{PROJECT}.{DATASET}.standings`
        ) WHERE rn = 1
        """
        return self.bq.query(q).to_dataframe()

    def load_historical_games(self) -> pd.DataFrame:
        """Pull recent historical games from the historical dataset for
        rolling baseline (pre-2026 context for early-season features)."""
        q = f"""
        SELECT game_pk, game_date, game_type,
               home_team_id, home_team_name,
               away_team_id, away_team_name,
               CAST(home_score AS INT64) AS home_score,
               CAST(away_score AS INT64) AS away_score
        FROM `{PROJECT}.mlb_historical_data.games_historical`
        WHERE year = 2025 AND status_code = 'F'
        ORDER BY game_date DESC
        LIMIT 5000
        """
        try:
            return self.bq.query(q).to_dataframe()
        except Exception as e:
            logger.warning("Could not load historical games for baseline: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Rolling features per team
    # ------------------------------------------------------------------
    @staticmethod
    def _rolling_win_pct(games_sorted: pd.DataFrame, team_id: int,
                         before_date, window: int) -> float:
        """Win pct for `team_id` in the last `window` games before `before_date`."""
        mask = (
            ((games_sorted["home_team_id"] == team_id) |
             (games_sorted["away_team_id"] == team_id)) &
            (games_sorted["game_date"] < before_date)
        )
        recent = games_sorted.loc[mask].tail(window)
        if len(recent) == 0:
            return 0.5
        wins = (
            ((recent["home_team_id"] == team_id) & (recent["home_score"] > recent["away_score"])) |
            ((recent["away_team_id"] == team_id) & (recent["away_score"] > recent["home_score"]))
        ).sum()
        return wins / len(recent)

    @staticmethod
    def _ema_form(games_sorted: pd.DataFrame, team_id: int,
                  before_date, span: int = 10) -> float:
        mask = (
            ((games_sorted["home_team_id"] == team_id) |
             (games_sorted["away_team_id"] == team_id)) &
            (games_sorted["game_date"] < before_date)
        )
        recent = games_sorted.loc[mask].tail(30)
        if len(recent) == 0:
            return 0.5
        outcomes = (
            ((recent["home_team_id"] == team_id) & (recent["home_score"] > recent["away_score"])) |
            ((recent["away_team_id"] == team_id) & (recent["away_score"] > recent["home_score"]))
        ).astype(float)
        return float(outcomes.ewm(span=span, min_periods=1).mean().iloc[-1])

    @staticmethod
    def _rest_days(games_sorted: pd.DataFrame, team_id: int,
                   game_date) -> int:
        mask = (
            ((games_sorted["home_team_id"] == team_id) |
             (games_sorted["away_team_id"] == team_id)) &
            (games_sorted["game_date"] < game_date)
        )
        prev = games_sorted.loc[mask]
        if prev.empty:
            return 3  # default
        last = pd.Timestamp(prev["game_date"].iloc[-1])
        current = pd.Timestamp(game_date)
        return max(int((current - last).days), 0)

    @staticmethod
    def _back_to_back(games_sorted: pd.DataFrame, team_id: int,
                      game_date) -> int:
        yesterday = pd.Timestamp(game_date) - timedelta(days=1)
        mask = (
            ((games_sorted["home_team_id"] == team_id) |
             (games_sorted["away_team_id"] == team_id)) &
            (games_sorted["game_date"] == yesterday.date() if hasattr(yesterday, 'date') else games_sorted["game_date"] == yesterday)
        )
        return 1 if mask.any() else 0

    # ------------------------------------------------------------------
    # Team-level metrics from stats snapshots
    # ------------------------------------------------------------------
    def _team_pitching_metric(self, team_stats: pd.DataFrame,
                               team_id: int, metric: str) -> float:
        row = team_stats.loc[
            (team_stats["team_id"] == team_id) &
            (team_stats["stat_type"] == "pitching")
        ]
        if row.empty:
            return 0.0
        return float(row.iloc[0].get(metric, 0) or 0)

    # ------------------------------------------------------------------
    # Build feature rows
    # ------------------------------------------------------------------
    def build_features(self, since: date | None = None) -> pd.DataFrame:
        games = self.load_games(since)
        if games.empty:
            logger.info("No completed games to featurize")
            return pd.DataFrame()

        # Combine with historical for rolling context
        hist = self.load_historical_games()
        if not hist.empty:
            hist["game_date"] = pd.to_datetime(hist["game_date"]).dt.date
        games["game_date"] = pd.to_datetime(games["game_date"]).dt.date

        all_games = pd.concat([hist, games], ignore_index=True).sort_values("game_date")
        # Coerce scores
        for col in ["home_score", "away_score"]:
            all_games[col] = pd.to_numeric(all_games[col], errors="coerce").fillna(0)

        team_stats = self.load_team_stats_latest()
        standings = self.load_standings_latest()

        # Build lookup for standings win%
        standings_wpct: Dict[int, float] = {}
        if not standings.empty:
            for _, r in standings.iterrows():
                standings_wpct[r["team_id"]] = float(r.get("win_percentage", 0.5) or 0.5)

        rows = []
        for _, g in games.iterrows():
            gd = g["game_date"]
            hid = g["home_team_id"]
            aid = g["away_team_id"]

            # Rolling
            h_wp10 = self._rolling_win_pct(all_games, hid, gd, 10)
            a_wp10 = self._rolling_win_pct(all_games, aid, gd, 10)
            h_wp30 = self._rolling_win_pct(all_games, hid, gd, 30)
            a_wp30 = self._rolling_win_pct(all_games, aid, gd, 30)
            h_ema = self._ema_form(all_games, hid, gd, span=10)
            a_ema = self._ema_form(all_games, aid, gd, span=10)

            # Rest
            h_rest = self._rest_days(all_games, hid, gd)
            a_rest = self._rest_days(all_games, aid, gd)
            h_b2b = self._back_to_back(all_games, hid, gd)

            # Pitcher quality from team ERA (lower = better, so invert)
            h_era = self._team_pitching_metric(team_stats, hid, "era")
            a_era = self._team_pitching_metric(team_stats, aid, "era")
            h_pq = max(0, 1 - (h_era / 10.0)) if h_era else 0.5
            a_pq = max(0, 1 - (a_era / 10.0)) if a_era else 0.5

            # Temporal
            month = gd.month if hasattr(gd, "month") else pd.Timestamp(gd).month
            dow = gd.weekday() if hasattr(gd, "weekday") else pd.Timestamp(gd).weekday()
            is_early = 1 if month <= 4 else 0
            is_late = 1 if month >= 9 else 0

            # Division matchup
            h_div = standings.loc[standings["team_id"] == hid, "division_id"]
            a_div = standings.loc[standings["team_id"] == aid, "division_id"]
            is_div = 1 if (not h_div.empty and not a_div.empty and
                           h_div.iloc[0] == a_div.iloc[0]) else 0

            # Outcome
            home_won = 1 if g["home_score"] > g["away_score"] else 0

            # v3 derived
            h_form_sq = h_ema ** 2
            a_form_sq = a_ema ** 2
            rest_balance = (h_rest - a_rest) / 5.0
            h_momentum = h_wp10 - h_ema
            a_momentum = a_wp10 - a_ema
            pq_diff = h_pq - a_pq
            fatigue = h_b2b * 2
            win_pct_diff = h_wp30 - a_wp30
            h_composite = 0.4 * h_wp10 + 0.3 * h_ema + 0.2 * h_pq + 0.1 * h_wp30
            a_composite = 0.4 * a_wp10 + 0.3 * a_ema + 0.2 * a_pq + 0.1 * a_wp30

            rows.append({
                "game_pk": g["game_pk"],
                "game_date": gd,
                "game_type": g.get("game_type", "R"),
                "home_team_id": hid,
                "home_team_name": g.get("home_team_name"),
                "away_team_id": aid,
                "away_team_name": g.get("away_team_name"),
                "home_score": int(g["home_score"]),
                "away_score": int(g["away_score"]),
                "home_won": home_won,
                # rolling
                "home_win_pct_10d": round(h_wp10, 4),
                "away_win_pct_10d": round(a_wp10, 4),
                "home_win_pct_30d": round(h_wp30, 4),
                "away_win_pct_30d": round(a_wp30, 4),
                "home_ema_form": round(h_ema, 4),
                "away_ema_form": round(a_ema, 4),
                # pitcher
                "home_pitcher_quality": round(h_pq, 4),
                "away_pitcher_quality": round(a_pq, 4),
                # rest
                "home_days_rest": h_rest,
                "away_days_rest": a_rest,
                "home_rest_advantage": h_rest - a_rest,
                "is_back_to_back": h_b2b,
                # temporal
                "month": month,
                "day_of_week": dow,
                "is_early_season": is_early,
                "is_late_season": is_late,
                "is_home": 1,
                # matchup
                "is_divisional_matchup": is_div,
                # v3 composite
                "home_form_squared": round(h_form_sq, 4),
                "away_form_squared": round(a_form_sq, 4),
                "rest_balance": round(rest_balance, 4),
                "home_momentum": round(h_momentum, 4),
                "away_momentum": round(a_momentum, 4),
                "pitcher_quality_diff": round(pq_diff, 4),
                "fatigue_index": fatigue,
                "win_pct_diff": round(win_pct_diff, 4),
                "home_composite_strength": round(h_composite, 4),
                "away_composite_strength": round(a_composite, 4),
                "feature_date": date.today().isoformat(),
            })

        df = pd.DataFrame(rows)
        logger.info("Built features for %d games", len(df))
        return df

    # ------------------------------------------------------------------
    # Load to BQ
    # ------------------------------------------------------------------
    def save_features(self, df: pd.DataFrame):
        if df.empty:
            return
        if self.dry_run:
            logger.info("[DRY RUN] would write %d feature rows", len(df))
            logger.info("Columns: %s", list(df.columns))
            return

        # Convert date columns
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["feature_date"] = pd.to_datetime(df["feature_date"])

        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            autodetect=True,
        )
        job = self.bq.load_table_from_dataframe(df, FEATURES_TABLE,
                                                  job_config=job_config)
        job.result()
        logger.info("✓ Wrote %d rows to %s", job.output_rows, FEATURES_TABLE)


def main():
    parser = argparse.ArgumentParser(description="Build 2026 game features")
    parser.add_argument("--since", help="Only games on or after YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    since = date.fromisoformat(args.since) if args.since else None
    builder = FeatureBuilder(dry_run=args.dry_run)
    features = builder.build_features(since=since)
    builder.save_features(features)


if __name__ == "__main__":
    main()
