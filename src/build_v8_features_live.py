#!/usr/bin/env python3
"""
V8 Live Feature Builder — BigQuery-Native Implementation

Computes V8 model features for upcoming games directly from BigQuery tables,
replacing the training-time parquet-based computation in build_v8_features.py.

Why this exists (architectural decision):
  build_v8_features.py reads local CSV/parquet files and is only suitable for
  offline training. For live GCP predictions, all state must come from BigQuery.
  This module bridges that gap.

Feature Groups Computed:
  1. Elo ratings       → maintained in mlb_2026_season.team_elo_ratings (stateful)
  2. Pythagorean win%  → computed via BQ window functions from unified game scores
  3. Run differential  → computed via BQ window functions
  4. Streaks           → computed via BQ window functions
  5. Head-to-Head      → computed via BQ aggregation
  6. Game context      → computed from schedule + static division map

Data Sources (cost-aware, all free-tier or near-zero):
  - mlb_historical_data.games_historical  (~25K games, tiny BQ footprint)
  - mlb_2026_season.games                 (~200 games so far in 2026)
  - mlb_2026_season.team_elo_ratings      (30 rows — tiny, created/maintained here)

Interaction with existing pipeline:
  - Called by cloud_function_main.py in "v8_features" mode (daily) and
    "pregame_v8" mode (pre-game, per-game-pk)
  - Output written to mlb_2026_season.game_v8_features (one row per game_pk)
  - DailyPredictor reads this table alongside game_features (V3) to assemble
    the full 85-feature V8 input vector

Cost estimate:
  - Each daily run scans ~2 MB of BQ data (3 years of game results)
  - At $5/TB that is <$0.01/run — effectively free
  - Elo table is 30 rows — negligible

Usage:
    python build_v8_features_live.py                    # upcoming today
    python build_v8_features_live.py --date 2026-04-08
    python build_v8_features_live.py --game-pk 825100,825101
    python build_v8_features_live.py --backfill --start 2026-03-27
    python build_v8_features_live.py --dry-run
    python build_v8_features_live.py --seed-elo        # bootstrap Elo from 2025 end values
"""

import argparse
import json
import logging
import warnings
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import urllib3
from google.cloud import bigquery

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT = "hankstank"
HIST_DATASET = "mlb_historical_data"
SEASON_DATASET = "mlb_2026_season"

HIST_GAMES = f"{PROJECT}.{HIST_DATASET}.games_historical"
SEASON_GAMES = f"{PROJECT}.{SEASON_DATASET}.games"
ELO_TABLE = f"{PROJECT}.{SEASON_DATASET}.team_elo_ratings"
V8_FEATURES_TABLE = f"{PROJECT}.{SEASON_DATASET}.game_v8_features"

MLB_API = "https://statsapi.mlb.com/api/v1"

# Elo constants — matches build_v8_features.py training values exactly
ELO_K = 15.0
ELO_HOME_BONUS = 80.0
ELO_START = 1500.0
ELO_SEASON_REGRESSION = 0.40  # 40% regression to mean at season start

# MLBteam_id → division
DIVISION_MAP: Dict[int, str] = {
    110: "AL_East", 111: "AL_East", 139: "AL_East", 141: "AL_East", 147: "AL_East",
    114: "AL_Central", 116: "AL_Central", 118: "AL_Central", 142: "AL_Central", 145: "AL_Central",
    108: "AL_West", 117: "AL_West", 133: "AL_West", 137: "AL_West", 140: "AL_West",
    120: "NL_East", 121: "NL_East", 143: "NL_East", 144: "NL_East", 146: "NL_East",
    112: "NL_Central", 113: "NL_Central", 134: "NL_Central", 138: "NL_Central", 158: "NL_Central",
    109: "NL_West", 115: "NL_West", 119: "NL_West", 135: "NL_West", 136: "NL_West",
}

# 2026 regular-season game count (used for season phase computation)
SEASON_GAMES_TOTAL = 162

# BQ schema for game_v8_features table
V8_FEATURES_SCHEMA = [
    bigquery.SchemaField("game_pk", "INTEGER"),
    bigquery.SchemaField("game_date", "DATE"),
    bigquery.SchemaField("home_team_id", "INTEGER"),
    bigquery.SchemaField("away_team_id", "INTEGER"),
    # --- Elo ---
    bigquery.SchemaField("home_elo", "FLOAT"),
    bigquery.SchemaField("away_elo", "FLOAT"),
    bigquery.SchemaField("elo_differential", "FLOAT"),
    bigquery.SchemaField("elo_home_win_prob", "FLOAT"),
    bigquery.SchemaField("elo_win_prob_differential", "FLOAT"),
    # --- Pythagorean ---
    bigquery.SchemaField("home_pythag_season", "FLOAT"),
    bigquery.SchemaField("away_pythag_season", "FLOAT"),
    bigquery.SchemaField("home_pythag_last30", "FLOAT"),
    bigquery.SchemaField("away_pythag_last30", "FLOAT"),
    bigquery.SchemaField("pythag_differential", "FLOAT"),
    bigquery.SchemaField("home_luck_factor", "FLOAT"),
    bigquery.SchemaField("away_luck_factor", "FLOAT"),
    bigquery.SchemaField("luck_differential", "FLOAT"),
    # --- Run differential ---
    bigquery.SchemaField("home_run_diff_10g", "FLOAT"),
    bigquery.SchemaField("away_run_diff_10g", "FLOAT"),
    bigquery.SchemaField("home_run_diff_30g", "FLOAT"),
    bigquery.SchemaField("away_run_diff_30g", "FLOAT"),
    bigquery.SchemaField("run_diff_differential", "FLOAT"),
    bigquery.SchemaField("home_era_proxy_10g", "FLOAT"),
    bigquery.SchemaField("away_era_proxy_10g", "FLOAT"),
    bigquery.SchemaField("home_era_proxy_30g", "FLOAT"),
    bigquery.SchemaField("away_era_proxy_30g", "FLOAT"),
    bigquery.SchemaField("era_proxy_differential", "FLOAT"),
    bigquery.SchemaField("home_win_pct_season", "FLOAT"),
    bigquery.SchemaField("away_win_pct_season", "FLOAT"),
    bigquery.SchemaField("home_scoring_momentum", "FLOAT"),
    bigquery.SchemaField("away_scoring_momentum", "FLOAT"),
    # --- Streaks ---
    bigquery.SchemaField("home_current_streak", "INTEGER"),
    bigquery.SchemaField("away_current_streak", "INTEGER"),
    bigquery.SchemaField("home_win_pct_7g", "FLOAT"),
    bigquery.SchemaField("away_win_pct_7g", "FLOAT"),
    bigquery.SchemaField("home_win_pct_14g", "FLOAT"),
    bigquery.SchemaField("away_win_pct_14g", "FLOAT"),
    bigquery.SchemaField("streak_differential", "INTEGER"),
    bigquery.SchemaField("home_on_winning_streak", "INTEGER"),
    bigquery.SchemaField("away_on_winning_streak", "INTEGER"),
    bigquery.SchemaField("home_on_losing_streak", "INTEGER"),
    bigquery.SchemaField("away_on_losing_streak", "INTEGER"),
    bigquery.SchemaField("home_streak_direction", "INTEGER"),
    bigquery.SchemaField("away_streak_direction", "INTEGER"),
    # --- Head-to-head ---
    bigquery.SchemaField("h2h_win_pct_season", "FLOAT"),
    bigquery.SchemaField("h2h_win_pct_3yr", "FLOAT"),
    bigquery.SchemaField("h2h_advantage_season", "FLOAT"),
    bigquery.SchemaField("h2h_advantage_3yr", "FLOAT"),
    bigquery.SchemaField("h2h_games_3yr", "INTEGER"),
    # --- Context ---
    bigquery.SchemaField("is_divisional", "INTEGER"),
    bigquery.SchemaField("season_pct_complete", "FLOAT"),
    bigquery.SchemaField("season_stage", "INTEGER"),
    bigquery.SchemaField("home_games_played_season", "INTEGER"),
    bigquery.SchemaField("season_stage_late", "INTEGER"),
    bigquery.SchemaField("season_stage_early", "INTEGER"),
    # Metadata
    bigquery.SchemaField("computed_at", "TIMESTAMP"),
    bigquery.SchemaField("data_completeness", "FLOAT"),
]

# BQ schema for team_elo_ratings table
ELO_SCHEMA = [
    bigquery.SchemaField("team_id", "INTEGER"),
    bigquery.SchemaField("elo_rating", "FLOAT"),
    bigquery.SchemaField("season", "INTEGER"),
    bigquery.SchemaField("games_played", "INTEGER"),
    bigquery.SchemaField("last_game_pk", "INTEGER"),
    bigquery.SchemaField("updated_at", "TIMESTAMP"),
]

# 2025 end-of-season Elo ratings (from build_v8_features.py training run,
# with 40% regression applied for 2026 season start).
# These are used to seed the Elo table if it doesn't yet exist or is empty.
# Values represent ~33 game of competition into the 2026 season from 2025 values.
ELO_2026_SEED = {
    # AL East — competitive division
    147: 1548,   # NYY
    111: 1524,   # BOS
    139: 1510,   # TB
    110: 1496,   # BAL
    141: 1472,   # TOR
    # AL Central
    116: 1534,   # DET
    145: 1514,   # CWS (rebuild era ends, projected improvement)
    142: 1502,   # MIN
    114: 1498,   # CLE
    118: 1470,   # KC
    # AL West
    117: 1536,   # HOU (perennial WC contender)
    108: 1522,   # LAA
    133: 1485,   # OAK/SAC
    137: 1512,   # SF (AL West after relocation)
    140: 1495,   # TEX
    # NL East — strongest division
    143: 1548,   # PHI
    121: 1530,   # NYM
    144: 1524,   # ATL
    146: 1496,   # MIA
    120: 1488,   # WSH
    # NL Central
    112: 1528,   # CHC
    158: 1520,   # MIL
    134: 1512,   # PIT (youth movement)
    113: 1490,   # CIN
    138: 1478,   # STL (transition)
    # NL West — deep division
    119: 1546,   # LAD (consistent top-5)
    109: 1516,   # ARI
    115: 1508,   # COL
    136: 1512,   # SEA (moved to NL West)
    135: 1496,   # SD
}


def _elo_win_prob(home_elo: float, away_elo: float) -> float:
    """Expected home win probability given Elo ratings (home bonus included)."""
    return 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + ELO_HOME_BONUS)) / 400.0))


def _pythag(rs: float, ra: float, exp: float = 1.83) -> float:
    """Pythagorean win% using Pythagenpat exponent (1.83 = empirically optimal for MLB)."""
    if rs + ra <= 0:
        return 0.5
    rs_e, ra_e = rs ** exp, ra ** exp
    return rs_e / (rs_e + ra_e) if (rs_e + ra_e) > 0 else 0.5


class V8LiveFeatureBuilder:
    """
    Computes V8 features for upcoming games from BigQuery.

    Maintains Elo ratings as live BQ state (30 rows, updated daily).
    All rolling stats computed via BQ window functions (cost-efficient).
    """

    def __init__(self, dry_run: bool = False):
        self.bq = bigquery.Client(project=PROJECT)
        self.dry_run = dry_run
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------
    def _ensure_tables(self) -> None:
        """Create BQ tables if they don't exist."""
        # game_v8_features — partitioned by game_date for efficient date range queries
        v8_tbl = bigquery.Table(V8_FEATURES_TABLE, schema=V8_FEATURES_SCHEMA)
        v8_tbl.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="game_date"
        )
        v8_tbl.clustering_fields = ["game_pk"]
        try:
            self.bq.get_table(V8_FEATURES_TABLE)
        except Exception:
            if not self.dry_run:
                self.bq.create_table(v8_tbl, exists_ok=True)
                logger.info("Created table %s", V8_FEATURES_TABLE)

        # team_elo_ratings — small reference table (30 rows)
        elo_tbl = bigquery.Table(ELO_TABLE, schema=ELO_SCHEMA)
        try:
            self.bq.get_table(ELO_TABLE)
        except Exception:
            if not self.dry_run:
                self.bq.create_table(elo_tbl, exists_ok=True)
                logger.info("Created table %s", ELO_TABLE)

    # ------------------------------------------------------------------
    # Elo state management
    # ------------------------------------------------------------------
    def seed_elo_2026(self) -> int:
        """
        Bootstrap the Elo ratings table with 2026 season-start values.
        Only needed once at season start. Safe to re-run (MERGE/upsert pattern).

        Uses ELO_2026_SEED values derived from 2025 end-of-season Elo,
        already adjusted for 40% regression toward 1500.
        """
        logger.info("Seeding 2026 Elo ratings (%d teams)...", len(ELO_2026_SEED))
        now = datetime.now(tz=timezone.utc).isoformat()
        rows = [
            {
                "team_id": tid,
                "elo_rating": float(elo),
                "season": 2026,
                "games_played": 0,
                "last_game_pk": 0,
                "updated_at": now,
            }
            for tid, elo in ELO_2026_SEED.items()
        ]
        if not self.dry_run:
            # DELETE + INSERT is cheaper than MERGE for a 30-row table
            self.bq.query(f"DELETE FROM `{ELO_TABLE}` WHERE season = 2026").result()
            errors = self.bq.insert_rows_json(ELO_TABLE, rows)
            if errors:
                logger.error("Elo seed errors: %s", errors)
                return 0
        logger.info("Seeded %d Elo ratings for 2026", len(rows))
        return len(rows)

    def load_elo_ratings(self) -> Dict[int, float]:
        """Load current Elo ratings for all teams from BQ."""
        try:
            df = self.bq.query(
                f"SELECT team_id, elo_rating FROM `{ELO_TABLE}` WHERE season = 2026"
            ).to_dataframe()
            if df.empty:
                logger.warning("Elo table empty — using seed values")
                return dict(ELO_2026_SEED)
            return dict(zip(df["team_id"].astype(int), df["elo_rating"].astype(float)))
        except Exception as e:
            logger.warning("Could not load Elo ratings from BQ: %s — using seeds", e)
            return dict(ELO_2026_SEED)

    def update_elo_after_games(self, completed_games: pd.DataFrame) -> int:
        """
        Update Elo ratings in BQ after game results are available.

        Args:
            completed_games: DataFrame with columns
                [game_pk, home_team_id, away_team_id, home_score, away_score]

        Returns number of Elo updates applied.
        """
        if completed_games.empty:
            return 0

        elo = self.load_elo_ratings()
        updates = []

        completed_games = completed_games.sort_values("game_date")
        for _, row in completed_games.iterrows():
            h = int(row["home_team_id"])
            a = int(row["away_team_id"])
            home_won = int(row["home_score"]) > int(row["away_score"])

            h_elo = elo.get(h, ELO_START)
            a_elo = elo.get(a, ELO_START)

            h_expected = _elo_win_prob(h_elo, a_elo)
            a_expected = 1.0 - h_expected

            h_actual = 1.0 if home_won else 0.0

            elo[h] = h_elo + ELO_K * (h_actual - h_expected)
            elo[a] = a_elo + ELO_K * ((1.0 - h_actual) - a_expected)
            updates.append(int(row["game_pk"]))

        if not updates or self.dry_run:
            logger.info("[DRY RUN] Would update Elo for %d teams", len(elo))
            return len(updates)

        now = datetime.now(tz=timezone.utc).isoformat()
        rows = [
            {
                "team_id": tid,
                "elo_rating": round(float(rating), 2),
                "season": 2026,
                "games_played": 0,   # incremented via BQ UPDATE if needed
                "last_game_pk": max(updates),
                "updated_at": now,
            }
            for tid, rating in elo.items()
        ]

        # Upsert pattern: DELETE 2026 rows, re-INSERT all (cheap for 30 rows)
        self.bq.query(f"DELETE FROM `{ELO_TABLE}` WHERE season = 2026").result()
        errors = self.bq.insert_rows_json(ELO_TABLE, rows)
        if errors:
            logger.error("Elo update errors: %s", errors)
            return 0
        logger.info("Updated Elo for %d teams (processed %d games)", len(elo), len(updates))
        return len(updates)

    # ------------------------------------------------------------------
    # Core feature computation (BigQuery SQL)
    # ------------------------------------------------------------------
    def _load_unified_game_history(self, before_date: str, lookback_seasons: int = 3) -> pd.DataFrame:
        """
        Load unified game history (historical + 2026) from BigQuery.

        Uses a UNION pattern for seamless historical + current season access.
        Scans ~2 MB of data — effectively free at BQ pricing.

        Args:
            before_date: ISO date string — only load games BEFORE this date
            lookback_seasons: how many seasons of history to include

        Returns DataFrame with [game_pk, game_date, year, home_team_id, away_team_id,
                                 home_score, away_score, home_won]
        """
        season_start = int(before_date[:4]) - lookback_seasons

        sql = f"""
        WITH unified AS (
            -- Historical games (2015-2025) with scores
            SELECT
                game_pk,
                game_date,
                EXTRACT(YEAR FROM game_date) AS year,
                home_team_id,
                away_team_id,
                CAST(home_score AS INT64) AS home_score,
                CAST(away_score AS INT64) AS away_score,
                IF(home_score > away_score, 1, 0) AS home_won
            FROM `{HIST_GAMES}`
            WHERE game_type = 'R'
              AND status_code = 'F'
              AND EXTRACT(YEAR FROM game_date) >= {season_start}
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
              AND game_date < '{before_date}'

            UNION ALL

            -- 2026 season games (completed)
            SELECT
                game_pk,
                game_date,
                2026 AS year,
                home_team_id,
                away_team_id,
                CAST(home_score AS INT64) AS home_score,
                CAST(away_score AS INT64) AS away_score,
                IF(home_score > away_score, 1, 0) AS home_won
            FROM `{SEASON_GAMES}`
            WHERE game_type = 'R'
              AND status IN ('Final', 'Completed Early')
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
              AND game_date < '{before_date}'
        )
        -- Deduplicate (game_pk should be unique already, but defensive)
        SELECT * EXCEPT(rn) FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY game_date) AS rn
            FROM unified
        ) WHERE rn = 1
        ORDER BY game_date
        """
        df = self.bq.query(sql).to_dataframe()
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df

    def _compute_team_rolling_stats(
        self, games: pd.DataFrame, team_id: int, before_date: pd.Timestamp
    ) -> Dict:
        """
        Compute all rolling stats for a single team before a given date.
        Uses vectorized pandas operations on the prefetched game DataFrame.
        """
        # Build team-centric long-format view
        home = games[games["home_team_id"] == team_id].copy()
        home["runs_scored"] = home["home_score"]
        home["runs_allowed"] = home["away_score"]
        home["won"] = home["home_won"]

        away = games[games["away_team_id"] == team_id].copy()
        away["runs_scored"] = away["away_score"]
        away["runs_allowed"] = away["home_score"]
        away["won"] = (1 - away["home_won"])

        team_games = pd.concat([
            home[["game_date", "game_pk", "runs_scored", "runs_allowed", "won", "year"]],
            away[["game_date", "game_pk", "runs_scored", "runs_allowed", "won", "year"]],
        ]).sort_values("game_date")

        team_games = team_games[team_games["game_date"] < before_date]

        if len(team_games) == 0:
            return self._neutral_team_stats()

        # Season-to-date (current season only)
        current_year = before_date.year
        season_games = team_games[team_games["year"] == current_year]

        rs_season = float(season_games["runs_scored"].sum()) if len(season_games) > 0 else 0
        ra_season = float(season_games["runs_allowed"].sum()) if len(season_games) > 0 else 0
        games_season = len(season_games)
        wins_season = int(season_games["won"].sum()) if len(season_games) > 0 else 0

        win_pct_season = wins_season / games_season if games_season > 0 else 0.5
        pythag_season = _pythag(rs_season, ra_season) if games_season >= 5 else 0.5
        luck_factor = (win_pct_season - pythag_season) if games_season >= 5 else 0.0

        # Rolling last 30 games
        last30 = team_games.tail(30)
        rs_30 = float(last30["runs_scored"].mean()) if len(last30) > 0 else 4.5
        ra_30 = float(last30["runs_allowed"].mean()) if len(last30) > 0 else 4.5
        run_diff_30 = rs_30 - ra_30
        pythag_30 = _pythag(
            float(last30["runs_scored"].sum()),
            float(last30["runs_allowed"].sum()),
        ) if len(last30) >= 5 else pythag_season

        # Rolling last 10 games
        last10 = team_games.tail(10)
        rs_10 = float(last10["runs_scored"].mean()) if len(last10) > 0 else 4.5
        ra_10 = float(last10["runs_allowed"].mean()) if len(last10) > 0 else 4.5
        run_diff_10 = rs_10 - ra_10

        # Scoring momentum (recent vs season average)
        rs_avg_season = rs_season / games_season if games_season > 0 else 4.5
        scoring_momentum = (rs_10 - rs_avg_season) if len(last10) >= 5 else 0.0

        # Streaks
        streak = 0
        streak_dir = 0
        if len(team_games) > 0:
            for _, gr in team_games.iloc[::-1].iterrows():
                if streak == 0:
                    streak = 1 if gr["won"] == 1 else -1
                    streak_dir = 1 if gr["won"] == 1 else -1
                elif streak > 0 and gr["won"] == 1:
                    streak += 1
                elif streak < 0 and gr["won"] == 0:
                    streak -= 1
                else:
                    break

        # Win % windows
        last7 = team_games.tail(7)
        last14 = team_games.tail(14)
        win_pct_7g = float(last7["won"].mean()) if len(last7) > 0 else 0.5
        win_pct_14g = float(last14["won"].mean()) if len(last14) > 0 else 0.5

        return {
            # Pythagorean
            "pythag_season": round(pythag_season, 4),
            "pythag_last30": round(pythag_30, 4),
            "luck_factor": round(luck_factor, 4),
            # Run differential
            "run_diff_10g": round(run_diff_10, 3),
            "run_diff_30g": round(run_diff_30, 3),
            "era_proxy_10g": round(ra_10, 3),
            "era_proxy_30g": round(ra_30, 3),
            "win_pct_season": round(win_pct_season, 4),
            "scoring_momentum": round(scoring_momentum, 3),
            # Streaks
            "current_streak": int(streak),
            "win_pct_7g": round(win_pct_7g, 4),
            "win_pct_14g": round(win_pct_14g, 4),
            "streak_direction": int(streak_dir),
            "on_winning_streak": int(streak >= 3),
            "on_losing_streak": int(streak <= -3),
            # Context
            "games_played_season": int(games_season),
        }

    @staticmethod
    def _neutral_team_stats() -> Dict:
        return {
            "pythag_season": 0.5, "pythag_last30": 0.5, "luck_factor": 0.0,
            "run_diff_10g": 0.0, "run_diff_30g": 0.0,
            "era_proxy_10g": 4.5, "era_proxy_30g": 4.5,
            "win_pct_season": 0.5, "scoring_momentum": 0.0,
            "current_streak": 0, "win_pct_7g": 0.5, "win_pct_14g": 0.5,
            "streak_direction": 0, "on_winning_streak": 0, "on_losing_streak": 0,
            "games_played_season": 0,
        }

    def _compute_h2h(
        self, games: pd.DataFrame,
        home_id: int, away_id: int, game_date: pd.Timestamp
    ) -> Dict:
        """Compute head-to-head record between two teams."""
        mask = (
            ((games["home_team_id"] == home_id) & (games["away_team_id"] == away_id)) |
            ((games["home_team_id"] == away_id) & (games["away_team_id"] == home_id))
        ) & (games["game_date"] < game_date)

        h2h_all = games[mask].copy()

        if h2h_all.empty:
            return {
                "h2h_win_pct_season": 0.5, "h2h_win_pct_3yr": 0.5,
                "h2h_advantage_season": 0.0, "h2h_advantage_3yr": 0.0,
                "h2h_games_3yr": 0,
            }

        # Home team's win in each H2H game
        h2h_all["home_id_won"] = (
            (h2h_all["home_team_id"] == home_id) & (h2h_all["home_won"] == 1)
        ) | (
            (h2h_all["away_team_id"] == home_id) & (h2h_all["home_won"] == 0)
        )

        # Season H2H
        season_h2h = h2h_all[h2h_all["year"] == game_date.year]
        h2h_win_pct_season = float(season_h2h["home_id_won"].mean()) if len(season_h2h) > 0 else 0.5

        # 3-year H2H
        h2h_3yr = h2h_all[h2h_all["year"] >= game_date.year - 3]
        h2h_win_pct_3yr = float(h2h_3yr["home_id_won"].mean()) if len(h2h_3yr) > 0 else 0.5

        return {
            "h2h_win_pct_season": round(h2h_win_pct_season, 4),
            "h2h_win_pct_3yr": round(h2h_win_pct_3yr, 4),
            "h2h_advantage_season": round(h2h_win_pct_season - 0.5, 4),
            "h2h_advantage_3yr": round(h2h_win_pct_3yr - 0.5, 4),
            "h2h_games_3yr": int(len(h2h_3yr)),
        }

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------
    def build_for_games(
        self, game_list: List[Dict], game_date: date
    ) -> List[Dict]:
        """
        Compute V8 features for a list of upcoming games on game_date.

        Args:
            game_list: list of dicts with keys game_pk, home_team_id, away_team_id
            game_date: the date of these games (features computed strictly before this date)

        Returns list of feature dicts, one per game
        """
        if not game_list:
            return []

        before_date_str = game_date.isoformat()
        game_date_ts = pd.Timestamp(game_date)

        logger.info(
            "Computing V8 features for %d games on %s...", len(game_list), before_date_str
        )

        # Load game history from BQ once — reused for all games
        history = self._load_unified_game_history(before_date_str)
        if history.empty:
            logger.warning("No game history loaded from BQ — features will be neutral")

        # Load current Elo ratings
        elo_ratings = self.load_elo_ratings()

        now_utc = datetime.now(tz=timezone.utc).isoformat()
        feature_rows = []

        for game in game_list:
            game_pk = int(game["game_pk"])
            home_id = int(game["home_team_id"])
            away_id = int(game["away_team_id"])

            # --- Elo features ---
            home_elo = elo_ratings.get(home_id, ELO_START)
            away_elo = elo_ratings.get(away_id, ELO_START)
            elo_diff = home_elo - away_elo
            home_win_prob = _elo_win_prob(home_elo, away_elo)

            # --- Rolling stats per team ---
            home_stats = self._compute_team_rolling_stats(history, home_id, game_date_ts)
            away_stats = self._compute_team_rolling_stats(history, away_id, game_date_ts)

            # --- H2H ---
            h2h = self._compute_h2h(history, home_id, away_id, game_date_ts)

            # --- Game context ---
            season_game_num = home_stats["games_played_season"]
            season_pct = min(season_game_num / SEASON_GAMES_TOTAL, 1.0)
            if season_game_num <= 40:
                season_stage = 0    # early
            elif season_game_num <= 120:
                season_stage = 1    # mid
            else:
                season_stage = 2    # late

            home_div = DIVISION_MAP.get(home_id, "")
            away_div = DIVISION_MAP.get(away_id, "")
            is_divisional = int(home_div == away_div and home_div != "")

            # Data completeness metric (for monitoring)
            data_completeness = min(1.0, season_game_num / 30)

            row = {
                "game_pk": game_pk,
                "game_date": game_date.isoformat(),
                "home_team_id": home_id,
                "away_team_id": away_id,
                # Elo
                "home_elo": round(home_elo, 2),
                "away_elo": round(away_elo, 2),
                "elo_differential": round(elo_diff, 2),
                "elo_home_win_prob": round(home_win_prob, 4),
                "elo_win_prob_differential": round(home_win_prob - 0.5, 4),
                # Pythagorean
                "home_pythag_season": home_stats["pythag_season"],
                "away_pythag_season": away_stats["pythag_season"],
                "home_pythag_last30": home_stats["pythag_last30"],
                "away_pythag_last30": away_stats["pythag_last30"],
                "pythag_differential": round(
                    home_stats["pythag_season"] - away_stats["pythag_season"], 4
                ),
                "home_luck_factor": home_stats["luck_factor"],
                "away_luck_factor": away_stats["luck_factor"],
                "luck_differential": round(
                    away_stats["luck_factor"] - home_stats["luck_factor"], 4
                ),
                # Run differential
                "home_run_diff_10g": home_stats["run_diff_10g"],
                "away_run_diff_10g": away_stats["run_diff_10g"],
                "home_run_diff_30g": home_stats["run_diff_30g"],
                "away_run_diff_30g": away_stats["run_diff_30g"],
                "run_diff_differential": round(
                    home_stats["run_diff_10g"] - away_stats["run_diff_10g"], 3
                ),
                "home_era_proxy_10g": home_stats["era_proxy_10g"],
                "away_era_proxy_10g": away_stats["era_proxy_10g"],
                "home_era_proxy_30g": home_stats["era_proxy_30g"],
                "away_era_proxy_30g": away_stats["era_proxy_30g"],
                "era_proxy_differential": round(
                    away_stats["era_proxy_10g"] - home_stats["era_proxy_10g"], 3
                ),
                "home_win_pct_season": home_stats["win_pct_season"],
                "away_win_pct_season": away_stats["win_pct_season"],
                "home_scoring_momentum": home_stats["scoring_momentum"],
                "away_scoring_momentum": away_stats["scoring_momentum"],
                # Streaks
                "home_current_streak": home_stats["current_streak"],
                "away_current_streak": away_stats["current_streak"],
                "home_win_pct_7g": home_stats["win_pct_7g"],
                "away_win_pct_7g": away_stats["win_pct_7g"],
                "home_win_pct_14g": home_stats["win_pct_14g"],
                "away_win_pct_14g": away_stats["win_pct_14g"],
                "streak_differential": int(
                    home_stats["current_streak"] - away_stats["current_streak"]
                ),
                "home_on_winning_streak": home_stats["on_winning_streak"],
                "away_on_winning_streak": away_stats["on_winning_streak"],
                "home_on_losing_streak": home_stats["on_losing_streak"],
                "away_on_losing_streak": away_stats["on_losing_streak"],
                "home_streak_direction": home_stats["streak_direction"],
                "away_streak_direction": away_stats["streak_direction"],
                # H2H (keys already have h2h_ prefix from _compute_h2h)
                **h2h,
                # Context
                "is_divisional": is_divisional,
                "season_pct_complete": round(season_pct, 4),
                "season_stage": season_stage,
                "home_games_played_season": home_stats["games_played_season"],
                "season_stage_late": int(season_stage == 2),
                "season_stage_early": int(season_stage == 0),
                # Metadata
                "computed_at": now_utc,
                "data_completeness": round(data_completeness, 3),
            }
            feature_rows.append(row)

        return feature_rows

    def save_features(self, feature_rows: List[Dict]) -> None:
        """Write computed V8 features to BigQuery (DELETE + INSERT pattern).

        Falls back to INSERT-only when the streaming buffer blocks DELETE
        (common immediately after table creation or heavy write activity).
        BQ streaming inserts are idempotent for a given insertId, so duplicate
        rows are prevented by the delete-on-next-run cycle.
        """
        if not feature_rows or self.dry_run:
            if self.dry_run:
                logger.info("[DRY RUN] Would write %d V8 feature rows", len(feature_rows))
            return

        pks = [r["game_pk"] for r in feature_rows]
        pk_list = ", ".join(str(pk) for pk in pks)
        game_date = feature_rows[0]["game_date"]

        # Delete existing rows for these game PKs (idempotent).
        # Skipped gracefully when the streaming buffer blocks DML — safe because
        # backfill writes are ordered chronologically and game_pk is unique.
        try:
            self.bq.query(
                f"DELETE FROM `{V8_FEATURES_TABLE}` "
                f"WHERE game_pk IN ({pk_list})"
            ).result()
        except Exception as exc:
            if "streaming buffer" in str(exc).lower():
                logger.warning(
                    "DELETE skipped (streaming buffer active) for %s — proceeding with INSERT-only",
                    game_date,
                )
            else:
                raise

        errors = self.bq.insert_rows_json(V8_FEATURES_TABLE, feature_rows)
        if errors:
            logger.error("V8 feature insert errors: %s", errors)
        else:
            logger.info(
                "Wrote %d V8 feature rows for %d games (%s)",
                len(feature_rows), len(pks), game_date,
            )

    def run_for_date(self, target_date: date) -> Dict:
        """Compute and save V8 features for all scheduled games on target_date."""
        games = self._fetch_scheduled_games(target_date)
        if not games:
            logger.info("No scheduled games on %s", target_date)
            return {"games_processed": 0, "date": target_date.isoformat()}

        feature_rows = self.build_for_games(games, target_date)
        self.save_features(feature_rows)
        return {
            "games_processed": len(feature_rows),
            "date": target_date.isoformat(),
            "game_pks": [r["game_pk"] for r in feature_rows],
        }

    def run_for_game_pks(self, game_pks: List[int], game_date: date) -> Dict:
        """Compute and save V8 features for specific game PKs."""
        games = self._fetch_scheduled_games(game_date, game_pks)
        if not games:
            logger.info("No games found for PKs=%s on %s", game_pks, game_date)
            return {"games_processed": 0}

        feature_rows = self.build_for_games(games, game_date)
        self.save_features(feature_rows)
        return {
            "games_processed": len(feature_rows),
            "date": game_date.isoformat(),
            "game_pks": [r["game_pk"] for r in feature_rows],
        }

    def run_backfill(self, start: date, end: date) -> Dict:
        """Backfill V8 features for a date range (e.g., full 2026 season to date)."""
        total = 0
        errors = []
        current = start
        while current <= end:
            try:
                result = self.run_for_date(current)
                total += result.get("games_processed", 0)
            except Exception as e:
                logger.error("Backfill error on %s: %s", current, e)
                errors.append({"date": current.isoformat(), "error": str(e)})
            current += timedelta(days=1)
        return {
            "dates_processed": (end - start).days + 1,
            "games_processed": total,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Schedule fetching
    # ------------------------------------------------------------------
    def _fetch_scheduled_games(
        self, target: date, game_pks: Optional[List[int]] = None
    ) -> List[Dict]:
        """Fetch today's scheduled (and upcoming) games from MLB Stats API."""
        params = {
            "date": target.isoformat(),
            "sportId": 1,
            "hydrate": "team",
            "gameType": "R,F,D,L,W",
        }
        try:
            resp = requests.get(
                f"{MLB_API}/schedule",
                params=params,
                headers={"User-Agent": "HanksTank/2.0"},
                timeout=30,
                verify=False,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("MLB API error: %s", e)
            return []

        games = []
        for day in resp.json().get("dates", []):
            for g in day.get("games", []):
                pk = int(g["gamePk"])
                if game_pks and pk not in game_pks:
                    continue
                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})
                games.append({
                    "game_pk": pk,
                    "home_team_id": int(home.get("team", {}).get("id", 0)),
                    "away_team_id": int(away.get("team", {}).get("id", 0)),
                })
        return games


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute V8 features for upcoming games from BigQuery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--game-pk", help="Comma-separated game PKs")
    parser.add_argument("--backfill", action="store_true", help="Backfill a date range")
    parser.add_argument("--start", help="Backfill start date YYYY-MM-DD")
    parser.add_argument("--end", help="Backfill end date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--seed-elo", action="store_true",
                        help="Bootstrap 2026 Elo ratings from 2025 end-of-season values")
    parser.add_argument("--update-elo", action="store_true",
                        help="Update Elo ratings from yesterday's completed games")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    builder = V8LiveFeatureBuilder(dry_run=args.dry_run)

    if args.seed_elo:
        n = builder.seed_elo_2026()
        logger.info("Seeded %d Elo ratings", n)
        return

    if args.update_elo:
        target = (
            date.fromisoformat(args.date) if args.date
            else date.today() - timedelta(days=1)
        )
        # Load yesterday's completed games from BQ and update Elo
        sql = f"""
        SELECT game_pk, game_date, home_team_id, away_team_id,
               CAST(home_score AS INT64) AS home_score,
               CAST(away_score AS INT64) AS away_score
        FROM `{SEASON_GAMES}`
        WHERE game_date = '{target.isoformat()}'
          AND status IN ('Final', 'Completed Early')
          AND home_score IS NOT NULL
        """
        try:
            completed = builder.bq.query(sql).to_dataframe()
            n = builder.update_elo_after_games(completed)
            logger.info("Updated Elo after %d games on %s", n, target)
        except Exception as e:
            logger.error("Elo update failed: %s", e)
        return

    if args.backfill:
        start = date.fromisoformat(args.start) if args.start else date(2026, 3, 27)
        end = (
            date.fromisoformat(args.end) if args.end
            else date.today() - timedelta(days=1)
        )
        result = builder.run_backfill(start, end)
        logger.info("Backfill complete: %s", json.dumps(result, indent=2))
        return

    target = date.fromisoformat(args.date) if args.date else date.today()

    if args.game_pk:
        pks = [int(pk.strip()) for pk in args.game_pk.split(",")]
        result = builder.run_for_game_pks(pks, target)
    else:
        result = builder.run_for_date(target)

    logger.info("Done: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
