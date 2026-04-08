#!/usr/bin/env python3
"""
V8 Feature Builder — "Accuracy-First" Feature Engineering

Goal: Break the ~54.6% accuracy ceiling of V3 by adding high-signal features
that the previous models completely lacked:

1. ELO RATINGS
   - Team strength tracked game-by-game (probabilistic, self-correcting)
   - Home advantage baked into ELO update (~70 Elo point bonus for home team)
   - Season-start regression to mean (revert 1/3 toward 1500 each off-season)
   - Why: Elo is the gold standard for tournament/game prediction globally.
     A 100-point Elo gap → ~64% win probability, validated across domains.

2. PYTHAGOREAN WIN PERCENTAGE
   - Expected win% based on run differential: RS^1.83 / (RS^1.83 + RA^1.83)
   - Measures "true" team quality, not luck
   - Luck factor = actual_win_pct - pythagorean_win_pct → regression predictor
   - Why: Teams with positive luck are expected to regress; negative luck = underrated.
     Research shows Pythagorean predicts future wins better than actual W/L.

3. RUN DIFFERENTIAL FEATURES
   - Rolling 10-game, 30-game run differential (avg runs/game)
   - Rolling scoring (RS/game and RA/game)
   - Run scoring momentum (recent vs longer-term)
   - Why: Run differential is more stable and predictive than win/loss records.
     A team scoring +2 runs/game consistently will win long-term.

4. STREAK & MOMENTUM FEATURES
   - Current win/loss streak magnitude and direction
   - Recent form vs historical form (momentum)
   - Why: Winning streaks correlate with 52-53% win probability next game
     (small but consistent signal from sports psychology literature).

5. HEAD-TO-HEAD RECORDS
   - Season H2H win percentage
   - 3-year H2H win percentage
   - Why: Some matchups are historically lopsided due to pitching style vs lineup,
     manager tendencies, and ballpark preferences for specific batters.

6. PITCHER QUALITY (IMPROVED)
   - Rolling 10-game team ERA (from runs allowed data)
   - Rolling 30-game team ERA
   - ERA differential
   - Why: Existing pitcher_quality feature is a flat 0.5 for ALL games 
     (completely non-informative). Real rolling ERA captures actual pitching quality.

7. CONSISTENCY FEATURES
   - Win% standard deviation (volatile vs consistent teams)
   - Run differential variance
   - Why: Consistent teams are more predictable; volatile teams have higher uncertainty.

8. GAME CONTEXT FEATURES
   - Season game number (1-162)
   - Games remaining in season
   - Season stage (early/mid/late)
   - Why: Teams play differently under playoff pressure vs early season.

Usage:
    from build_v8_features import V8FeatureBuilder
    builder = V8FeatureBuilder()
    train_df, val_df = builder.build()
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "training"

# Game scores CSV from the backend repo.
# Resolution priority:
#   1. GAMES_CSV_PATH environment variable (set in CI / GCP if needed)
#   2. Standard relative path from this repo structure (local dev)
#   3. Sibling directory fallback (monorepo layout)
_GAMES_CSV_ENV = os.environ.get("GAMES_CSV_PATH")
_GAMES_CSV_REL = BASE_DIR.parent / "hanks_tank_backend" / "data" / "games" / "games_2015_2024.csv"
_GAMES_CSV_GCS_FALLBACK_ENV = os.environ.get("GAMES_CSV_GCS")  # gs://bucket/path if needed

if _GAMES_CSV_ENV:
    GAMES_CSV = Path(_GAMES_CSV_ENV)
elif _GAMES_CSV_REL.exists():
    GAMES_CSV = _GAMES_CSV_REL
else:
    # Allow import to succeed; the actual path will be validated at runtime in _load_data()
    GAMES_CSV = _GAMES_CSV_REL
    logger.warning(
        "games_2015_2024.csv not found at expected path %s. "
        "Set GAMES_CSV_PATH env variable or ensure hanks_tank_backend is a sibling directory.",
        GAMES_CSV,
    )

# MLB team ID to division mapping (2021+ alignment; includes expansion from 2022 Athletics move)
# AL East: 110, 111, 139, 141, 147
# AL Central: 114, 116, 118, 142, 145
# AL West: 108, 117, 133, 137, 140
# NL East: 120, 121, 143, 144, 146
# NL Central: 112, 113, 134, 138, 158
# NL West: 109, 115, 119, 135, 136
DIVISION_MAP: Dict[int, str] = {
    # AL East
    110: "AL_East", 111: "AL_East", 139: "AL_East", 141: "AL_East", 147: "AL_East",
    # AL Central
    114: "AL_Central", 116: "AL_Central", 118: "AL_Central", 142: "AL_Central", 145: "AL_Central",
    # AL West
    108: "AL_West", 117: "AL_West", 133: "AL_West", 137: "AL_West", 140: "AL_West",
    # NL East
    120: "NL_East", 121: "NL_East", 143: "NL_East", 144: "NL_East", 146: "NL_East",
    # NL Central
    112: "NL_Central", 113: "NL_Central", 134: "NL_Central", 138: "NL_Central", 158: "NL_Central",
    # NL West
    109: "NL_West", 115: "NL_West", 119: "NL_West", 135: "NL_West", 136: "NL_West",
}

# ELO constants
ELO_START = 1500.0        # Initial rating for all teams
ELO_K = 20.0              # K-factor (learning rate)
ELO_HOME_BONUS = 70.0     # Home field advantage in Elo points (~54-55% base win probability)
ELO_SEASON_REGRESSION = 0.33   # Regress 1/3 toward mean at season start


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected win probability for rating_a vs rating_b."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def pythagorean_win_pct(runs_scored: float, runs_allowed: float, exponent: float = 1.83) -> float:
    """
    Pythagorean win percentage (Bill James formula).
    Uses 1.83 exponent (Pythagenpat formula, empirically optimal for MLB).
    """
    if runs_scored + runs_allowed <= 0:
        return 0.5
    rs_exp = runs_scored ** exponent
    ra_exp = runs_allowed ** exponent
    return rs_exp / (rs_exp + ra_exp) if (rs_exp + ra_exp) > 0 else 0.5


class V8FeatureBuilder:
    """
    Builds V8 enhanced features by computing game-over-game statistics
    from raw game outcomes and scores.

    All features are computed STRICTLY BEFORE each game's outcome
    (no lookahead leakage).
    """

    def __init__(self):
        self.all_games = None     # All games 2015-2025 with outcomes
        self.games_with_scores = None   # 2015-2024 with scores
        self.features_df = None  # Final feature matrix for all games

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load all available game data."""
        logger.info("Loading game data...")

        # All outcomes 2015-2025 (for Elo, streaks, win pcts)
        self.all_games = pd.read_parquet(DATA_DIR / "all_games_2015_2025.parquet")
        self.all_games["game_date"] = pd.to_datetime(self.all_games["game_date"])
        self.all_games = self.all_games.sort_values("game_date").reset_index(drop=True)

        # Scores 2015-2024 (for run differential, Pythagorean features)
        self.games_with_scores = pd.read_csv(GAMES_CSV)
        self.games_with_scores["game_date"] = pd.to_datetime(self.games_with_scores["game_date"])
        # Keep only regular season (status_code == 'F', game_type == 'R')
        self.games_with_scores = self.games_with_scores[
            (self.games_with_scores["status_code"] == "F") &
            (self.games_with_scores["game_type"] == "R")
        ].dropna(subset=["home_score", "away_score"])

        logger.info(f"Loaded {len(self.all_games)} total game outcomes (2015-2025)")
        logger.info(f"Loaded {len(self.games_with_scores)} scored games (2015-2024)")

    # ------------------------------------------------------------------
    # Elo Rating computation
    # ------------------------------------------------------------------

    def _compute_elo_ratings(self) -> pd.DataFrame:
        """
        Compute pre-game Elo ratings for all teams in all games.

        Algorithm:
        1. All teams start at ELO_START = 1500
        2. Before each game: home team gets +ELO_HOME_BONUS advantage
        3. After each game: ratings updated using expected vs actual outcome
        4. At each new season: ratings regress 1/3 toward 1500

        Returns DataFrame with [game_pk, home_elo, away_elo, elo_differential, elo_home_win_prob]
        """
        logger.info("Computing Elo ratings for 2015-2025...")

        games = self.all_games.copy()

        elo: Dict[int, float] = {}
        records = []
        current_season = None

        for _, row in games.iterrows():
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
            game_season = int(row["year"])
            home_won = int(row["home_won"])

            # Initialize new teams
            if home_id not in elo:
                elo[home_id] = ELO_START
            if away_id not in elo:
                elo[away_id] = ELO_START

            # Season regression to mean (revert at season start)
            if current_season is not None and game_season != current_season:
                for team_id in list(elo.keys()):
                    elo[team_id] = elo[team_id] + ELO_SEASON_REGRESSION * (ELO_START - elo[team_id])
            current_season = game_season

            # Pre-game ratings (BEFORE this game)
            home_pre = elo[home_id]
            away_pre = elo[away_id]

            # Expected win probability (home gets home field bonus)
            home_expected = elo_expected(home_pre + ELO_HOME_BONUS, away_pre)
            away_expected = 1.0 - home_expected

            # Elo differential raw (before home bonus)
            elo_diff = home_pre - away_pre

            records.append({
                "game_pk": int(row["game_pk"]),
                "home_elo": round(home_pre, 2),
                "away_elo": round(away_pre, 2),
                "elo_differential": round(elo_diff, 2),
                "elo_home_win_prob": round(home_expected, 4),
                "elo_win_prob_differential": round(home_expected - 0.5, 4),
            })

            # Update ratings after game
            home_actual = float(home_won)
            away_actual = 1.0 - home_actual

            elo[home_id] += ELO_K * (home_actual - home_expected)
            elo[away_id] += ELO_K * (away_actual - away_expected)

        elo_df = pd.DataFrame(records)
        logger.info(f"Computed Elo for {len(elo_df)} games. Elo range: [{elo_df['home_elo'].min():.0f}, {elo_df['home_elo'].max():.0f}]")
        return elo_df

    # ------------------------------------------------------------------
    # Run differential & Pythagorean features
    # ------------------------------------------------------------------

    def _compute_run_features(self) -> pd.DataFrame:
        """
        Compute run-based features from scored games (2015-2024):
        - Rolling 10-game & 30-game avg run differential
        - Rolling 10-game & 30-game avg runs scored and allowed
        - Season-to-date run differential
        - Season-to-date Pythagorean win percentage
        - Luck factor (actual_win_pct - pythagorean_win_pct)
        - Rolling ERA proxy (runs_allowed / games)

        For 2025 games (no score data), values are set to each team's
        end-of-2024 season values, decayed toward league average.

        Returns DataFrame with [game_pk, home_*/away_* run features]
        """
        logger.info("Computing run differential and Pythagorean features...")

        scored = self.games_with_scores.copy()
        scored = scored.sort_values("game_date")

        # We need team-level stats per game, computed before the game
        # Build a long-format dataset: one row per team per game
        home_view = scored[[
            "game_pk", "game_date", "year", "home_team_id", "away_team_id",
            "home_score", "away_score", "winning_team_id"
        ]].copy()
        home_view.columns = [
            "game_pk", "game_date", "year", "team_id", "opp_id",
            "runs_scored", "runs_allowed", "winning_team_id"
        ]
        home_view["is_home"] = 1
        home_view["won"] = (home_view["winning_team_id"] == home_view["team_id"]).astype(int)

        away_view = scored[[
            "game_pk", "game_date", "year", "away_team_id", "home_team_id",
            "away_score", "home_score", "winning_team_id"
        ]].copy()
        away_view.columns = [
            "game_pk", "game_date", "year", "team_id", "opp_id",
            "runs_scored", "runs_allowed", "winning_team_id"
        ]
        away_view["is_home"] = 0
        away_view["won"] = (away_view["winning_team_id"] == away_view["team_id"]).astype(int)

        team_games = pd.concat([home_view, away_view], ignore_index=True)
        team_games = team_games.sort_values(["team_id", "game_date"]).reset_index(drop=True)

        # Dictionary to hold pre-game rolling stats per team
        # Structure: {team_id: list of game records}
        team_history: Dict[int, list] = {}
        team_season_history: Dict[Tuple[int, int], list] = {}

        result_records = []

        # Process each unique game in chronological order
        all_game_pks = scored.sort_values("game_date")["game_pk"].unique()

        for game_pk in all_game_pks:
            game_row = scored[scored["game_pk"] == game_pk].iloc[0]
            home_id = int(game_row["home_team_id"])
            away_id = int(game_row["away_team_id"])
            game_date = game_row["game_date"]
            game_year = int(game_row["year"])

            # Compute pre-game stats for home team
            home_stats = self._team_rolling_stats(
                team_history.get(home_id, []),
                team_season_history.get((home_id, game_year), [])
            )
            away_stats = self._team_rolling_stats(
                team_history.get(away_id, []),
                team_season_history.get((away_id, game_year), [])
            )

            rec = {"game_pk": int(game_pk)}
            for k, v in home_stats.items():
                rec[f"home_{k}"] = v
            for k, v in away_stats.items():
                rec[f"away_{k}"] = v

            # Differential features
            rec["run_diff_differential"] = (
                (home_stats.get("run_diff_10g") or 0) -
                (away_stats.get("run_diff_10g") or 0)
            )
            rec["pythag_differential"] = (
                (home_stats.get("pythag_season") or 0.5) -
                (away_stats.get("pythag_season") or 0.5)
            )
            rec["luck_differential"] = (
                (home_stats.get("luck_factor") or 0) -
                (away_stats.get("luck_factor") or 0)
            )
            rec["era_proxy_differential"] = (
                (away_stats.get("era_proxy_10g") or 4.0) -
                (home_stats.get("era_proxy_10g") or 4.0)
            )  # positive = home team's pitching has been better

            result_records.append(rec)

            # Update team history AFTER the game
            home_result = {
                "game_pk": game_pk,
                "game_date": game_date,
                "year": game_year,
                "runs_scored": float(game_row["home_score"]),
                "runs_allowed": float(game_row["away_score"]),
                "won": int(game_row["home_score"] > game_row["away_score"]),
            }
            away_result = {
                "game_pk": game_pk,
                "game_date": game_date,
                "year": game_year,
                "runs_scored": float(game_row["away_score"]),
                "runs_allowed": float(game_row["home_score"]),
                "won": int(game_row["away_score"] > game_row["home_score"]),
            }

            if home_id not in team_history:
                team_history[home_id] = []
            if away_id not in team_history:
                team_history[away_id] = []
            if (home_id, game_year) not in team_season_history:
                team_season_history[(home_id, game_year)] = []
            if (away_id, game_year) not in team_season_history:
                team_season_history[(away_id, game_year)] = []

            team_history[home_id].append(home_result)
            team_history[away_id].append(away_result)
            team_season_history[(home_id, game_year)].append(home_result)
            team_season_history[(away_id, game_year)].append(away_result)

        run_df = pd.DataFrame(result_records)
        logger.info(f"Computed run features for {len(run_df)} scored games")

        # For 2025 games (no scores), carry forward end-of-2024 stats
        # per team with league-average decay
        run_df = self._add_2025_run_features(run_df, team_history, team_season_history)

        return run_df

    def _team_rolling_stats(
        self,
        all_history: list,
        season_history: list,
    ) -> dict:
        """Compute rolling stats from team history."""
        stats = {}

        if len(all_history) == 0:
            # No history = league average defaults
            return {
                "run_diff_10g": 0.0,
                "run_diff_30g": 0.0,
                "runs_scored_10g": 4.5,
                "runs_allowed_10g": 4.5,
                "runs_scored_30g": 4.5,
                "runs_allowed_30g": 4.5,
                "pythag_season": 0.5,
                "pythag_last30": 0.5,
                "luck_factor": 0.0,
                "era_proxy_10g": 4.5,   # runs allowed per game (proxy for ERA)
                "era_proxy_30g": 4.5,
                "win_pct_season": 0.5,
                "scoring_momentum": 0.0,
                "consistency_score": 0.0,
            }

        # Rolling 10 games
        last10 = all_history[-10:] if len(all_history) >= 10 else all_history
        rs_10 = np.mean([g["runs_scored"] for g in last10])
        ra_10 = np.mean([g["runs_allowed"] for g in last10])
        stats["run_diff_10g"] = round(rs_10 - ra_10, 3)
        stats["runs_scored_10g"] = round(rs_10, 3)
        stats["runs_allowed_10g"] = round(ra_10, 3)
        stats["era_proxy_10g"] = round(ra_10, 3)

        # Rolling 30 games
        last30 = all_history[-30:] if len(all_history) >= 30 else all_history
        rs_30 = np.mean([g["runs_scored"] for g in last30])
        ra_30 = np.mean([g["runs_allowed"] for g in last30])
        stats["run_diff_30g"] = round(rs_30 - ra_30, 3)
        stats["runs_scored_30g"] = round(rs_30, 3)
        stats["runs_allowed_30g"] = round(ra_30, 3)
        stats["era_proxy_30g"] = round(ra_30, 3)

        # Season stats
        if season_history:
            season_rs = sum(g["runs_scored"] for g in season_history)
            season_ra = sum(g["runs_allowed"] for g in season_history)
            n_games = len(season_history)
            season_wins = sum(g["won"] for g in season_history)
            actual_win_pct = season_wins / n_games if n_games > 0 else 0.5
            stats["win_pct_season"] = round(actual_win_pct, 4)
            pythag = pythagorean_win_pct(season_rs, season_ra)
            stats["pythag_season"] = round(pythag, 4)
            stats["luck_factor"] = round(actual_win_pct - pythag, 4)
        else:
            stats["win_pct_season"] = 0.5
            stats["pythag_season"] = 0.5
            stats["luck_factor"] = 0.0

        # Pythagorean from last 30
        last30_rs = sum(g["runs_scored"] for g in last30)
        last30_ra = sum(g["runs_allowed"] for g in last30)
        stats["pythag_last30"] = round(pythagorean_win_pct(last30_rs, last30_ra), 4)

        # Scoring momentum: recent scoring rate vs season average
        season_rs_per_g = (
            sum(g["runs_scored"] for g in season_history) / len(season_history)
            if season_history else 4.5
        )
        stats["scoring_momentum"] = round(rs_10 - season_rs_per_g, 3)

        # Consistency: std dev of run differential over last 20 games
        last20 = all_history[-20:] if len(all_history) >= 20 else all_history
        if len(last20) >= 3:
            run_diffs = [g["runs_scored"] - g["runs_allowed"] for g in last20]
            stats["consistency_score"] = round(-np.std(run_diffs), 3)  # negative = more volatile
        else:
            stats["consistency_score"] = 0.0

        return stats

    def _add_2025_run_features(
        self,
        run_df: pd.DataFrame,
        team_history: dict,
        team_season_history: dict,
    ) -> pd.DataFrame:
        """
        For 2025 games (no score data), use each team's end-of-2024 stats
        decayed slightly toward league average (40% decay).
        """
        games_2025 = self.all_games[self.all_games["year"] == 2025].copy()
        if len(games_2025) == 0:
            return run_df

        logger.info(f"Adding 2025 run feature estimates for {len(games_2025)} games...")

        # End-of-2024 stats per team
        end_2024_stats: Dict[int, dict] = {}
        league_avg_rd = 0.0
        league_avg_rs = 4.5
        league_avg_ra = 4.5

        all_team_ids = list({
            int(r["home_team_id"]) for _, r in games_2025.iterrows()
        } | {
            int(r["away_team_id"]) for _, r in games_2025.iterrows()
        })

        for team_id in all_team_ids:
            th = team_history.get(team_id, [])
            season_th = team_season_history.get((team_id, 2024), [])
            stats = self._team_rolling_stats(th, season_th)
            # Decay toward league average (partial regression to mean)
            decay = 0.4
            end_2024_stats[team_id] = {
                "run_diff_10g": stats["run_diff_10g"] * (1 - decay) + league_avg_rd * decay,
                "run_diff_30g": stats["run_diff_30g"] * (1 - decay) + league_avg_rd * decay,
                "runs_scored_10g": stats["runs_scored_10g"] * (1 - decay) + league_avg_rs * decay,
                "runs_allowed_10g": stats["runs_allowed_10g"] * (1 - decay) + league_avg_ra * decay,
                "runs_scored_30g": stats["runs_scored_30g"] * (1 - decay) + league_avg_rs * decay,
                "runs_allowed_30g": stats["runs_allowed_30g"] * (1 - decay) + league_avg_ra * decay,
                "pythag_season": stats["pythag_season"] * (1 - decay) + 0.5 * decay,
                "pythag_last30": stats["pythag_last30"] * (1 - decay) + 0.5 * decay,
                "luck_factor": stats["luck_factor"] * (1 - decay),
                "era_proxy_10g": stats["era_proxy_10g"] * (1 - decay) + league_avg_ra * decay,
                "era_proxy_30g": stats["era_proxy_30g"] * (1 - decay) + league_avg_ra * decay,
                "win_pct_season": 0.5,   # Reset for new season
                "scoring_momentum": 0.0,  # Reset
                "consistency_score": 0.0,  # Reset
            }

        records_2025 = []
        for _, row in games_2025.iterrows():
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
            home_stats = end_2024_stats.get(home_id, {})
            away_stats = end_2024_stats.get(away_id, {})

            if not home_stats:
                home_stats = self._team_rolling_stats([], [])
            if not away_stats:
                away_stats = self._team_rolling_stats([], [])

            rec = {"game_pk": int(row["game_pk"])}
            for k, v in home_stats.items():
                rec[f"home_{k}"] = v
            for k, v in away_stats.items():
                rec[f"away_{k}"] = v
            rec["run_diff_differential"] = (
                home_stats.get("run_diff_10g", 0) - away_stats.get("run_diff_10g", 0)
            )
            rec["pythag_differential"] = (
                home_stats.get("pythag_season", 0.5) - away_stats.get("pythag_season", 0.5)
            )
            rec["luck_differential"] = (
                home_stats.get("luck_factor", 0) - away_stats.get("luck_factor", 0)
            )
            rec["era_proxy_differential"] = (
                away_stats.get("era_proxy_10g", 4.5) - home_stats.get("era_proxy_10g", 4.5)
            )
            records_2025.append(rec)

        df_2025 = pd.DataFrame(records_2025)
        run_df = pd.concat([run_df, df_2025], ignore_index=True)
        return run_df

    # ------------------------------------------------------------------
    # Streak features
    # ------------------------------------------------------------------

    def _compute_streak_features(self) -> pd.DataFrame:
        """
        Compute pre-game streak and recent form features.

        Features:
        - current_streak: positive = win streak length, negative = loss streak
        - max_win_streak_season: longest win streak this season
        - games_since_last_win: for teams on losing streaks
        - recent_win_pct_7g: win pct over last 7 games

        Returns DataFrame with [game_pk, home_streak, away_streak, ...]
        """
        logger.info("Computing streak features...")

        games = self.all_games.copy()
        games = games.sort_values("game_date").reset_index(drop=True)

        team_history: Dict[int, list] = {}
        team_season_wins: Dict[Tuple[int, int], int] = {}
        team_season_streak: Dict[Tuple[int, int], int] = {}

        records = []

        for _, row in games.iterrows():
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
            game_year = int(row["year"])
            home_won = int(row["home_won"])

            home_streak_stats = self._team_streak_stats(
                team_history.get(home_id, []),
                team_season_streak.get((home_id, game_year), 0),
                team_season_wins.get((home_id, game_year), 0)
            )
            away_streak_stats = self._team_streak_stats(
                team_history.get(away_id, []),
                team_season_streak.get((away_id, game_year), 0),
                team_season_wins.get((away_id, game_year), 0)
            )

            rec = {"game_pk": int(row["game_pk"])}
            for k, v in home_streak_stats.items():
                rec[f"home_{k}"] = v
            for k, v in away_streak_stats.items():
                rec[f"away_{k}"] = v
            rec["streak_differential"] = (
                home_streak_stats["current_streak"] - away_streak_stats["current_streak"]
            )

            records.append(rec)

            # Update history AFTER game
            if home_id not in team_history:
                team_history[home_id] = []
            if away_id not in team_history:
                team_history[away_id] = []

            team_history[home_id].append({"won": home_won, "year": game_year})
            team_history[away_id].append({"won": 1 - home_won, "year": game_year})

            # Update season streaks
            prev_home_streak = team_season_streak.get((home_id, game_year), 0)
            prev_away_streak = team_season_streak.get((away_id, game_year), 0)

            if home_won == 1:
                team_season_streak[(home_id, game_year)] = max(prev_home_streak, 0) + 1
                team_season_streak[(away_id, game_year)] = min(prev_away_streak, 0) - 1
            else:
                team_season_streak[(home_id, game_year)] = min(prev_home_streak, 0) - 1
                team_season_streak[(away_id, game_year)] = max(prev_away_streak, 0) + 1

            team_season_wins[(home_id, game_year)] = (
                team_season_wins.get((home_id, game_year), 0) + home_won
            )
            team_season_wins[(away_id, game_year)] = (
                team_season_wins.get((away_id, game_year), 0) + (1 - home_won)
            )

        streak_df = pd.DataFrame(records)
        logger.info(f"Computed streak features for {len(streak_df)} games")
        return streak_df

    def _team_streak_stats(
        self,
        all_history: list,
        current_season_streak: int = 0,
        season_wins: int = 0,
    ) -> dict:
        """Compute streak stats from team history."""
        if len(all_history) == 0:
            return {
                "current_streak": 0,
                "streak_magnitude": 0,
                "streak_direction": 0,
                "win_pct_7g": 0.5,
                "win_pct_14g": 0.5,
                "on_winning_streak": 0,
                "on_losing_streak": 0,
            }

        # Current streak
        current_streak = current_season_streak

        # Last 7 and 14 game win pcts
        last7 = all_history[-7:] if len(all_history) >= 7 else all_history
        last14 = all_history[-14:] if len(all_history) >= 14 else all_history
        win_pct_7g = np.mean([g["won"] for g in last7])
        win_pct_14g = np.mean([g["won"] for g in last14])

        return {
            "current_streak": float(current_streak),
            "streak_magnitude": float(abs(current_streak)),
            "streak_direction": float(np.sign(current_streak)),  # +1 = winning, -1 = losing
            "win_pct_7g": round(float(win_pct_7g), 4),
            "win_pct_14g": round(float(win_pct_14g), 4),
            "on_winning_streak": int(current_streak >= 3),  # 3+ game win streak
            "on_losing_streak": int(current_streak <= -3),  # 3+ game losing streak
        }

    # ------------------------------------------------------------------
    # Head-to-Head features
    # ------------------------------------------------------------------

    def _compute_h2h_features(self) -> pd.DataFrame:
        """
        Compute season and 3-year head-to-head records.

        Features:
        - h2h_win_pct_season: home team win % vs this opponent this season
        - h2h_games_season: sample size
        - h2h_win_pct_3yr: home team win % vs this opponent last 3 seasons
        - h2h_games_3yr: sample size

        Returns DataFrame with [game_pk, h2h_*]
        """
        logger.info("Computing head-to-head features...")

        games = self.all_games.copy()
        games = games.sort_values("game_date").reset_index(drop=True)

        # H2H history: {(home_id, away_id, year): [won]}
        h2h_season: Dict[Tuple[int, int, int], list] = {}
        h2h_all: Dict[Tuple[int, int], list] = {}

        records = []

        for _, row in games.iterrows():
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
            game_year = int(row["year"])
            home_won = int(row["home_won"])

            # Season H2H (this season, home vs away)
            season_key = (home_id, away_id, game_year)
            season_h2h = h2h_season.get(season_key, [])
            h2h_games_season = len(season_h2h)
            h2h_win_pct_season = np.mean(season_h2h) if season_h2h else 0.5

            # Multi-year H2H (all available history, home vs away)
            all_key = (home_id, away_id)
            all_h2h = h2h_all.get(all_key, [])
            # Use last 3 years of games (max 57 games = 3 × ~19 per year)
            recent_h2h = all_h2h[-57:] if len(all_h2h) > 57 else all_h2h
            h2h_games_3yr = len(recent_h2h)
            h2h_win_pct_3yr = np.mean(recent_h2h) if recent_h2h else 0.5

            records.append({
                "game_pk": int(row["game_pk"]),
                "h2h_win_pct_season": round(h2h_win_pct_season, 4),
                "h2h_games_season": h2h_games_season,
                "h2h_win_pct_3yr": round(h2h_win_pct_3yr, 4),
                "h2h_games_3yr": h2h_games_3yr,
                "h2h_advantage_season": round(h2h_win_pct_season - 0.5, 4),
                "h2h_advantage_3yr": round(h2h_win_pct_3yr - 0.5, 4),
            })

            # Update history AFTER game
            if season_key not in h2h_season:
                h2h_season[season_key] = []
            h2h_season[season_key].append(home_won)

            if all_key not in h2h_all:
                h2h_all[all_key] = []
            h2h_all[all_key].append(home_won)

        h2h_df = pd.DataFrame(records)
        logger.info(f"Computed H2H features for {len(h2h_df)} games")
        return h2h_df

    # ------------------------------------------------------------------
    # Game context features
    # ------------------------------------------------------------------

    def _compute_context_features(self) -> pd.DataFrame:
        """
        Game context features:
        - is_divisional: home and away teams in same division
        - game_number_season: sequential game number within season
        - season_pct_complete: 0-1 fraction of season complete
        - season_stage: 0=early(games 1-40), 1=mid(41-120), 2=late(121+)

        Returns DataFrame with [game_pk, context_*]
        """
        logger.info("Computing game context features...")

        games = self.all_games.copy()
        games = games.sort_values("game_date").reset_index(drop=True)

        # Count game number per team per season
        team_game_count: Dict[Tuple[int, int], int] = {}

        records = []

        for _, row in games.iterrows():
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
            game_year = int(row["year"])

            home_key = (home_id, game_year)
            away_key = (away_id, game_year)

            home_game_n = team_game_count.get(home_key, 0)
            away_game_n = team_game_count.get(away_key, 0)

            # Game number in season proxy (avg of home and away)
            game_n = (home_game_n + away_game_n) / 2.0
            season_pct = game_n / 162.0
            season_pct = min(season_pct, 1.0)

            # Season stage
            if game_n < 40:
                season_stage = 0  # Early
            elif game_n < 120:
                season_stage = 1  # Mid
            else:
                season_stage = 2  # Late (playoff implications)

            # Divisional matchup
            home_div = DIVISION_MAP.get(home_id, "Unknown")
            away_div = DIVISION_MAP.get(away_id, "Unknown")
            is_divisional = int(home_div == away_div and home_div != "Unknown")

            # League matchup (interleague)
            is_interleague = int(
                home_div != "Unknown" and away_div != "Unknown" and
                home_div[0] != away_div[0]
            )

            records.append({
                "game_pk": int(row["game_pk"]),
                "home_games_played_season": home_game_n,
                "away_games_played_season": away_game_n,
                "season_pct_complete": round(season_pct, 4),
                "season_stage": season_stage,
                "is_divisional": is_divisional,
                "is_interleague": is_interleague,
                "season_stage_late": int(season_stage == 2),
                "season_stage_early": int(season_stage == 0),
            })

            # Update game counts AFTER game
            team_game_count[home_key] = home_game_n + 1
            team_game_count[away_key] = away_game_n + 1

        context_df = pd.DataFrame(records)
        logger.info(f"Computed context features for {len(context_df)} games")
        return context_df

    # ------------------------------------------------------------------
    # Main build method
    # ------------------------------------------------------------------

    def build(
        self,
        save: bool = True,
        train_output: Optional[Path] = None,
        val_output: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build V8 feature matrix for all games 2015-2025.

        Returns (train_df, val_df) where:
        - train_df: 2015-2024 games with V3 + V8 features
        - val_df: 2025 games with V3 + V8 features
        """
        self._load_data()

        # Compute each feature group
        elo_df = self._compute_elo_ratings()
        run_df = self._compute_run_features()
        streak_df = self._compute_streak_features()
        h2h_df = self._compute_h2h_features()
        context_df = self._compute_context_features()

        # Load base V3 features
        logger.info("Loading V3 base features...")
        train_v3 = pd.read_parquet(DATA_DIR / "train_v3_2015_2024.parquet")
        val_v3 = pd.read_parquet(DATA_DIR / "val_v3_2025.parquet")

        base_df = pd.concat([train_v3, val_v3], ignore_index=True)
        logger.info(f"Base V3 features: {base_df.shape}")

        # Merge all feature groups on game_pk
        feature_tables = [elo_df, run_df, streak_df, h2h_df, context_df]
        feature_names = ["elo", "run", "streak", "h2h", "context"]

        merged = base_df.copy()
        for table, name in zip(feature_tables, feature_names):
            before = len(merged)
            merged = merged.merge(table, on="game_pk", how="left")
            after = len(merged)
            joined = merged[table.columns[1]].notna().sum()
            logger.info(f"Merged {name}: {before}→{after} rows, {joined} non-null")

        # Fill any remaining NaNs with reasonable defaults
        merged = self._fill_missing_values(merged)

        # Split back into train and val
        train_df = merged[merged["year"] <= 2024].reset_index(drop=True)
        val_df = merged[merged["year"] == 2025].reset_index(drop=True)

        logger.info(f"V8 TRAIN: {train_df.shape} ({train_df['year'].min()}-{train_df['year'].max()})")
        logger.info(f"V8 VAL: {val_df.shape} ({val_df['year'].unique()})")

        total_v8_features = len([c for c in merged.columns if c not in
                                  set(train_v3.columns) | {"game_pk"}])
        logger.info(f"New V8 features added: {total_v8_features}")
        logger.info(f"Total V8 features: {len(merged.columns)}")

        if save:
            out_train = train_output or (DATA_DIR / "train_v8_2015_2024.parquet")
            out_val = val_output or (DATA_DIR / "val_v8_2025.parquet")
            train_df.to_parquet(out_train, index=False)
            val_df.to_parquet(out_val, index=False)
            logger.info(f"Saved: {out_train} ({len(train_df)} rows)")
            logger.info(f"Saved: {out_val} ({len(val_df)} rows)")

        return train_df, val_df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values with sensible defaults."""
        fill_rules = {
            # Elo features
            "home_elo": 1500.0, "away_elo": 1500.0,
            "elo_differential": 0.0, "elo_home_win_prob": 0.535,
            "elo_win_prob_differential": 0.035,
            # Run features (neutral = league average)
            "home_run_diff_10g": 0.0, "away_run_diff_10g": 0.0,
            "home_run_diff_30g": 0.0, "away_run_diff_30g": 0.0,
            "home_runs_scored_10g": 4.5, "away_runs_scored_10g": 4.5,
            "home_runs_allowed_10g": 4.5, "away_runs_allowed_10g": 4.5,
            "home_runs_scored_30g": 4.5, "away_runs_scored_30g": 4.5,
            "home_runs_allowed_30g": 4.5, "away_runs_allowed_30g": 4.5,
            "home_pythag_season": 0.5, "away_pythag_season": 0.5,
            "home_pythag_last30": 0.5, "away_pythag_last30": 0.5,
            "home_luck_factor": 0.0, "away_luck_factor": 0.0,
            "home_era_proxy_10g": 4.5, "away_era_proxy_10g": 4.5,
            "home_era_proxy_30g": 4.5, "away_era_proxy_30g": 4.5,
            "home_win_pct_season": 0.5, "away_win_pct_season": 0.5,
            "home_scoring_momentum": 0.0, "away_scoring_momentum": 0.0,
            "home_consistency_score": 0.0, "away_consistency_score": 0.0,
            "run_diff_differential": 0.0, "pythag_differential": 0.0,
            "luck_differential": 0.0, "era_proxy_differential": 0.0,
            # Streak features
            "home_current_streak": 0, "away_current_streak": 0,
            "home_streak_magnitude": 0, "away_streak_magnitude": 0,
            "home_streak_direction": 0, "away_streak_direction": 0,
            "home_win_pct_7g": 0.5, "away_win_pct_7g": 0.5,
            "home_win_pct_14g": 0.5, "away_win_pct_14g": 0.5,
            "home_on_winning_streak": 0, "away_on_winning_streak": 0,
            "home_on_losing_streak": 0, "away_on_losing_streak": 0,
            "streak_differential": 0.0,
            # H2H features
            "h2h_win_pct_season": 0.5, "h2h_win_pct_3yr": 0.5,
            "h2h_games_season": 0, "h2h_games_3yr": 0,
            "h2h_advantage_season": 0.0, "h2h_advantage_3yr": 0.0,
            # Context features
            "home_games_played_season": 0, "away_games_played_season": 0,
            "season_pct_complete": 0.0, "season_stage": 0,
            "is_divisional": 0, "is_interleague": 0,
            "season_stage_late": 0, "season_stage_early": 1,
        }

        for col, val in fill_rules.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        # Fill any remaining NaN numeric with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                med = df[col].median()
                df[col] = df[col].fillna(med if not np.isnan(med) else 0)

        return df


if __name__ == "__main__":
    builder = V8FeatureBuilder()
    train_df, val_df = builder.build()
    print(f"\nV8 Feature Matrix built successfully.")
    print(f"Training:   {train_df.shape}")
    print(f"Validation: {val_df.shape}")
    print(f"\nNew V8 features (sample):")
    v3_cols = set(pd.read_parquet(DATA_DIR / "train_v3_2015_2024.parquet").columns)
    new_cols = [c for c in train_df.columns if c not in v3_cols]
    for c in new_cols:
        vals = train_df[c].describe()
        print(f"  {c:45s} mean={vals['mean']:.3f} std={vals['std']:.3f}")
