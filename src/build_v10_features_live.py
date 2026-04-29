#!/usr/bin/env python3
"""
V10 Live Feature Builder — BigQuery-Native Implementation

Extends V8 features with the core V10 groups that drove a
+6.71% accuracy gain on 2026 live data (61.48% vs 54.77% for V8):

  1. SP Quality (18 features)  — game-level starting pitcher Statcast percentile
     ranks (xERA, K%, BB%, Whiff%, FBV). Source: Baseball Savant via GCS/local.
  2. Park Factors (5 features) — venue-based run-scoring factors from 2015–2024
     historical data. Embedded as a static dict (39 venues, rarely changes).
  3. Rest / Travel (7 features) — days rest since last game per team, consecutive
     away games on road trip. Computed from BQ game history.

Also computes extra rolling stats that V8 doesn't produce but V10 needs:
  - win_pct_3g / 10g / 30g (V8 only has 7g + season)
  - run_diff_3g / 7g (V8 only has 10g + 30g)
  - runs_scored / runs_allowed (10g, 30g)
  - Team quality features from MLB Stats API (ERA, WHIP, K9, BB9, OBP, SLG, OPS)
  - Calendar features (day_of_week, month, season phase)

Data sources:
  - mlb_2026_season.game_v8_features    (Elo, Pythagorean, most rolling stats)
  - mlb_historical_data.games_historical  (rolling stat windows)
  - mlb_2026_season.games                 (2026 results so far)
  - MLB Stats API /teams/stats            (team season ERA/WHIP/OBP/SLG)
  - MLB Stats API /schedule               (probable pitchers, venue, series context)
  - statcast_sp_pct_{year}.parquet        (SP Statcast percentile ranks, local or GCS)

Output: mlb_2026_season.game_v10_features  (one row per game_pk, expanded V10 feature set)

Usage:
    python build_v10_features_live.py                      # today
    python build_v10_features_live.py --date 2026-05-01
    python build_v10_features_live.py --game-pk 825100,825101
    python build_v10_features_live.py --backfill --start 2026-03-27
    python build_v10_features_live.py --dry-run
"""

import argparse
import logging
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
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

HIST_GAMES     = f"{PROJECT}.{HIST_DATASET}.games_historical"
SEASON_GAMES   = f"{PROJECT}.{SEASON_DATASET}.games"
V8_TABLE       = f"{PROJECT}.{SEASON_DATASET}.game_v8_features"
V10_TABLE      = f"{PROJECT}.{SEASON_DATASET}.game_v10_features"
MATCHUP_TABLE  = f"{PROJECT}.{SEASON_DATASET}.matchup_features"
ELO_TABLE      = f"{PROJECT}.{SEASON_DATASET}.team_elo_ratings"
MLB_API        = "https://statsapi.mlb.com/api/v1"

# ---------------------------------------------------------------------------
# SP quality defaults (50 = league average percentile rank)
# ---------------------------------------------------------------------------
SP_DEFAULT = {"xera": 50.0, "k_pct": 50.0, "bb_pct": 50.0, "whiff": 50.0, "fbv": 50.0}
SP_COMPOSITE_WEIGHTS = {"xera": 0.35, "k_pct": 0.25, "bb_pct": 0.20, "whiff": 0.15, "fbv": 0.05}
SP_METRICS = ["xera", "k_pct", "bb_pct", "whiff", "fbv"]

# ---------------------------------------------------------------------------
# Park factors — static dict {venue_id: (ratio, 100_scale)}
# Derived from 2015–2024 run-scoring data (39 historical venues).
# 1.0 = league neutral; 1.273 = Coors Field (+27% run environment).
# Missing venues default to (1.0, 100.0).
# ---------------------------------------------------------------------------
PARK_FACTORS: Dict[int, tuple] = {
    19:   (1.2734, 127.3),  # Coors Field
    13:   (1.1868, 118.7),  # Globe Life Park in Arlington (old TEX)
    3:    (1.1236, 112.4),  # Fenway Park
    2602: (1.0710, 107.1),  # Great American Ball Park
    15:   (1.0708, 107.1),  # Chase Field
    2:    (1.0504, 105.0),  # Oriole Park at Camden Yards
    3312: (1.0412, 104.1),  # Target Field
    1:    (1.0368, 103.7),  # Angel Stadium
    7:    (1.0324, 103.2),  # Kauffman Stadium
    4705: (1.0205, 102.1),  # Truist Park (SunTrust Park)
    3309: (1.0176, 101.8),  # Nationals Park
    5325: (1.0132, 101.3),  # Globe Life Field (new TEX, 2020+)
    14:   (1.0106, 101.1),  # Rogers Centre
    2681: (1.0085, 100.9),  # Citizens Bank Park
    3313: (1.0047, 100.5),  # Yankee Stadium
    4:    (0.9961,  99.6),  # Guaranteed Rate Field / U.S. Cellular Field
    2394: (0.9938,  99.4),  # Comerica Park
    32:   (0.9925,  99.2),  # American Family Field (Miller Park)
    31:   (0.9874,  98.7),  # PNC Park
    5:    (0.9698,  97.0),  # Progressive Field
    17:   (0.9638,  96.4),  # Wrigley Field
    2392: (0.9585,  95.8),  # Minute Maid Park
    10:   (0.9543,  95.4),  # Oakland Coliseum
    16:   (0.9499,  95.0),  # Turner Field
    2889: (0.9325,  93.2),  # Busch Stadium
    2395: (0.9309,  93.1),  # Oracle Park / AT&T Park
    2680: (0.9296,  93.0),  # Petco Park
    22:   (0.9283,  92.8),  # Dodger Stadium
    4169: (0.9111,  91.1),  # loanDepot park / Marlins Park
    3289: (0.9096,  91.0),  # Citi Field
    12:   (0.9072,  90.7),  # Tropicana Field
    680:  (0.9053,  90.5),  # T-Mobile Park / Safeco Field
}

# ---------------------------------------------------------------------------
# Repo-relative path to Statcast SP parquet files (local fallback)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SP_DATA_DIR = _REPO_ROOT / "data" / "v9" / "raw"
_V10_SP_DIR  = _REPO_ROOT / "data" / "v10" / "raw"

# ---------------------------------------------------------------------------
# BigQuery schema for game_v10_features table
# ---------------------------------------------------------------------------
def _float(name):  return bigquery.SchemaField(name, "FLOAT")
def _int(name):    return bigquery.SchemaField(name, "INTEGER")
def _ts(name):     return bigquery.SchemaField(name, "TIMESTAMP")
def _date(name):   return bigquery.SchemaField(name, "DATE")

V10_FEATURES_SCHEMA = [
    _int("game_pk"), _date("game_date"),
    _int("home_team_id"), _int("away_team_id"),
    # --- Elo (from V8) ---
    _float("home_elo"), _float("away_elo"), _float("elo_differential"),
    _float("elo_home_win_prob"), _float("elo_win_prob_differential"),
    # --- Pythagorean (from V8) ---
    _float("home_pythag_season"), _float("away_pythag_season"),
    _float("home_pythag_last30"), _float("away_pythag_last30"),
    _float("pythag_differential"),
    _float("home_luck_factor"), _float("away_luck_factor"), _float("luck_differential"),
    # --- Rolling form ---
    _float("home_win_pct_3g"), _float("away_win_pct_3g"),
    _float("home_win_pct_7g"), _float("away_win_pct_7g"),
    _float("home_win_pct_10g"), _float("away_win_pct_10g"),
    _float("home_win_pct_30g"), _float("away_win_pct_30g"),
    _float("home_win_pct_season"), _float("away_win_pct_season"),
    _float("win_pct_diff"),
    _float("home_run_diff_3g"), _float("away_run_diff_3g"),
    _float("home_run_diff_7g"), _float("away_run_diff_7g"),
    _float("home_run_diff_10g"), _float("away_run_diff_10g"),
    _float("home_run_diff_30g"), _float("away_run_diff_30g"),
    _float("run_diff_differential"),
    _float("home_runs_scored_10g"), _float("away_runs_scored_10g"),
    _float("home_runs_scored_30g"), _float("away_runs_scored_30g"),
    _float("home_runs_allowed_10g"), _float("away_runs_allowed_10g"),
    _float("home_runs_allowed_30g"), _float("away_runs_allowed_30g"),
    _float("home_scoring_momentum"), _float("away_scoring_momentum"),
    _float("home_era_proxy_10g"), _float("away_era_proxy_10g"),
    _float("home_era_proxy_30g"), _float("away_era_proxy_30g"),
    _float("era_proxy_differential"),
    # --- Streaks (from V8) ---
    _int("home_current_streak"), _int("away_current_streak"),
    _int("home_streak_direction"), _int("away_streak_direction"),
    _int("streak_differential"),
    # --- H2H (from V8) ---
    _float("home_h2h_win_pct_season"), _int("home_h2h_games_season"),
    _float("home_h2h_win_pct_3yr"),
    # --- Matchup / lineup quality (new V10 upgrade) ---
    _int("lineup_confirmed"),
    _float("home_lineup_woba_vs_hand"), _float("away_lineup_woba_vs_hand"),
    _float("lineup_woba_differential"),
    _float("home_lineup_k_pct_vs_hand"), _float("away_lineup_k_pct_vs_hand"),
    _float("lineup_k_pct_differential"),
    _float("home_top3_woba_vs_hand"), _float("away_top3_woba_vs_hand"),
    _float("home_middle4_woba_vs_hand"), _float("away_middle4_woba_vs_hand"),
    _float("home_bottom2_woba_vs_hand"), _float("away_bottom2_woba_vs_hand"),
    _float("home_pct_same_hand"), _float("away_pct_same_hand"),
    _float("home_h2h_woba"), _float("away_h2h_woba"),
    _float("h2h_woba_differential"),
    _float("matchup_advantage_home"),
    # --- Team quality (MLB API) ---
    _float("home_fg_era"), _float("away_fg_era"),
    _float("home_fg_whip"), _float("away_fg_whip"),
    _float("home_fg_k9"), _float("away_fg_k9"),
    _float("home_fg_bb9"), _float("away_fg_bb9"),
    _float("home_fg_xfip"), _float("away_fg_xfip"),   # proxy via ERA
    _float("home_fg_k_pct"), _float("away_fg_k_pct"),
    _float("home_fg_bb_pct"), _float("away_fg_bb_pct"),
    _float("home_fg_whiff_pct"), _float("away_fg_whiff_pct"),
    _float("home_fg_fbv_pct"), _float("away_fg_fbv_pct"),
    _float("fg_era_differential"), _float("fg_xfip_differential"), _float("fg_whip_differential"),
    _float("home_fg_ops"), _float("away_fg_ops"),
    _float("home_fg_obp"), _float("away_fg_obp"),
    _float("home_fg_slg"), _float("away_fg_slg"),
    _float("home_fg_woba"), _float("away_fg_woba"),
    _float("home_fg_ev_pct"), _float("away_fg_ev_pct"),
    _float("home_fg_hh_pct"), _float("away_fg_hh_pct"),
    _float("home_fg_brl_pct"), _float("away_fg_brl_pct"),
    _float("fg_ops_differential"), _float("fg_woba_differential"), _float("fg_obp_differential"),
    # --- Park factors ---
    _float("home_park_factor"), _float("home_park_factor_100"),
    _int("park_factor_known"), _int("is_hitter_park"), _int("is_pitcher_park"),
    # --- Calendar ---
    _int("day_of_week"), _int("month"), _int("is_weekend"),
    _int("month_3"), _int("month_4"), _int("month_5"), _int("month_6"),
    _int("month_7"), _int("month_8"), _int("month_9"), _int("month_10"),
    _int("season_game_number"), _float("season_pct_complete"),
    _int("is_late_season"), _int("is_early_season"),
    # --- SP Quality (V10 new) ---
    _float("home_sp_xera"), _float("away_sp_xera"), _float("sp_xera_diff"),
    _float("home_sp_k_pct"), _float("away_sp_k_pct"), _float("sp_k_pct_diff"),
    _float("home_sp_bb_pct"), _float("away_sp_bb_pct"), _float("sp_bb_pct_diff"),
    _float("home_sp_whiff"), _float("away_sp_whiff"), _float("sp_whiff_diff"),
    _float("home_sp_fbv"), _float("away_sp_fbv"), _float("sp_fbv_diff"),
    _float("sp_quality_composite_diff"),
    _int("home_sp_known"), _int("away_sp_known"),
    # --- Rest / Travel (V10 new) ---
    _float("home_days_rest"), _float("away_days_rest"), _float("rest_differential"),
    _float("away_road_trip_length"),
    _int("home_rested"), _int("away_tired"), _int("long_road_trip"),
    # --- Series Context (V10 new) ---
    _int("series_game_number"), _int("games_in_series"), _int("is_series_opener"),
    # --- Metadata ---
    _ts("computed_at"),
]

# Canonical V10 feature list for live/train parity.
V10_MODEL_FEATURES = [
    "home_elo", "away_elo", "elo_differential", "elo_home_win_prob", "elo_win_prob_differential",
    "home_pythag_season", "away_pythag_season", "home_pythag_last30", "away_pythag_last30",
    "pythag_differential", "home_luck_factor", "away_luck_factor", "luck_differential",
    "home_win_pct_3g", "away_win_pct_3g", "home_win_pct_7g", "away_win_pct_7g",
    "home_win_pct_10g", "away_win_pct_10g", "home_win_pct_30g", "away_win_pct_30g",
    "home_win_pct_season", "away_win_pct_season", "win_pct_diff",
    "home_run_diff_3g", "away_run_diff_3g", "home_run_diff_7g", "away_run_diff_7g",
    "home_run_diff_10g", "away_run_diff_10g", "home_run_diff_30g", "away_run_diff_30g",
    "run_diff_differential",
    "home_runs_scored_10g", "away_runs_scored_10g", "home_runs_scored_30g", "away_runs_scored_30g",
    "home_runs_allowed_10g", "away_runs_allowed_10g", "home_runs_allowed_30g", "away_runs_allowed_30g",
    "home_scoring_momentum", "away_scoring_momentum",
    "home_era_proxy_10g", "away_era_proxy_10g", "home_era_proxy_30g", "away_era_proxy_30g",
    "era_proxy_differential",
    "home_current_streak", "away_current_streak", "home_streak_direction", "away_streak_direction",
    "streak_differential",
    "home_h2h_win_pct_season", "home_h2h_games_season", "home_h2h_win_pct_3yr",
    "lineup_confirmed",
    "home_lineup_woba_vs_hand", "away_lineup_woba_vs_hand", "lineup_woba_differential",
    "home_lineup_k_pct_vs_hand", "away_lineup_k_pct_vs_hand", "lineup_k_pct_differential",
    "home_top3_woba_vs_hand", "away_top3_woba_vs_hand",
    "home_middle4_woba_vs_hand", "away_middle4_woba_vs_hand",
    "home_bottom2_woba_vs_hand", "away_bottom2_woba_vs_hand",
    "home_pct_same_hand", "away_pct_same_hand",
    "home_h2h_woba", "away_h2h_woba", "h2h_woba_differential",
    "matchup_advantage_home",
    "home_fg_era", "away_fg_era", "home_fg_whip", "away_fg_whip",
    "home_fg_k9", "away_fg_k9", "home_fg_bb9", "away_fg_bb9",
    "home_fg_xfip", "away_fg_xfip",
    "home_fg_k_pct", "away_fg_k_pct", "home_fg_bb_pct", "away_fg_bb_pct",
    "home_fg_whiff_pct", "away_fg_whiff_pct", "home_fg_fbv_pct", "away_fg_fbv_pct",
    "fg_era_differential", "fg_xfip_differential", "fg_whip_differential",
    "home_fg_ops", "away_fg_ops", "home_fg_obp", "away_fg_obp",
    "home_fg_slg", "away_fg_slg", "home_fg_woba", "away_fg_woba",
    "home_fg_ev_pct", "away_fg_ev_pct", "home_fg_hh_pct", "away_fg_hh_pct",
    "home_fg_brl_pct", "away_fg_brl_pct",
    "fg_ops_differential", "fg_woba_differential", "fg_obp_differential",
    "home_park_factor", "home_park_factor_100",
    "day_of_week", "month", "is_weekend",
    "month_3", "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10",
    "season_game_number", "season_pct_complete", "is_late_season", "is_early_season",
    "home_sp_xera", "away_sp_xera", "sp_xera_diff",
    "home_sp_k_pct", "away_sp_k_pct", "sp_k_pct_diff",
    "home_sp_bb_pct", "away_sp_bb_pct", "sp_bb_pct_diff",
    "home_sp_whiff", "away_sp_whiff", "sp_whiff_diff",
    "home_sp_fbv", "away_sp_fbv", "sp_fbv_diff",
    "sp_quality_composite_diff", "home_sp_known", "away_sp_known",
    "home_days_rest", "away_days_rest", "rest_differential",
    "away_road_trip_length", "home_rested", "away_tired", "long_road_trip",
    "series_game_number", "games_in_series", "is_series_opener",
]

SEASON_GAMES_TOTAL = 162


class V10LiveFeatureBuilder:
    """
    Computes the live V10 feature set for upcoming games.

    Combines features from multiple sources:
      - BQ game_v8_features: Elo, Pythagorean, most rolling stats, streaks, H2H
      - BQ game history: extra rolling windows not in V8 (3g, 10g, 30g, runs scored)
      - MLB Stats API: team season ERA/WHIP/OBP/SLG, probable pitchers, series info
      - Local parquet: SP Statcast percentile ranks (xERA, K%, BB%, Whiff%, FBV)
      - Static dict: park factors by venue_id
    """

    def __init__(self, dry_run: bool = False):
        self.bq = bigquery.Client(project=PROJECT)
        self.dry_run = dry_run
        self._sp_lookup: Dict = {}     # {(player_id, season): stats} — loaded once
        self._team_quality: Dict = {}  # {(team_id, season): stats} — loaded once per call
        self._ensure_table()

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------
    def _ensure_table(self) -> None:
        tbl = bigquery.Table(V10_TABLE, schema=V10_FEATURES_SCHEMA)
        tbl.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="game_date"
        )
        tbl.clustering_fields = ["game_pk"]
        try:
            existing = self.bq.get_table(V10_TABLE)
            existing_names = {field.name for field in existing.schema}
            missing_fields = [
                field for field in V10_FEATURES_SCHEMA if field.name not in existing_names
            ]
            if missing_fields and not self.dry_run:
                existing.schema = [*existing.schema, *missing_fields]
                self.bq.update_table(existing, ["schema"])
                logger.info(
                    "Extended %s schema with %d new columns",
                    V10_TABLE,
                    len(missing_fields),
                )
        except Exception:
            if not self.dry_run:
                self.bq.create_table(tbl, exists_ok=True)
                logger.info("Created table %s", V10_TABLE)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def run_for_date(self, target_date: date, include_final: bool = False) -> dict:
        games = self._fetch_schedule(target_date, include_final=include_final)
        if not games:
            logger.info("No upcoming games on %s", target_date)
            return {"games_processed": 0, "date": target_date.isoformat()}
        return self._build_and_write(games, target_date)

    def run_for_game_pks(
        self,
        game_pks: List[int],
        target_date: date,
        include_final: bool = False,
    ) -> dict:
        games = self._fetch_schedule(target_date, game_pks, include_final=include_final)
        if not games:
            return {"games_processed": 0, "date": target_date.isoformat()}
        return self._build_and_write(games, target_date)

    def run_backfill(self, start: date, end: date) -> dict:
        import time
        total = 0
        errors = []
        current = start
        while current <= end:
            try:
                games = self._fetch_schedule(current, include_final=True)
                if not games:
                    logger.info("No games on %s", current)
                    current += timedelta(days=1)
                    continue
                r = self._build_and_write(games, current)
                total += r.get("games_processed", 0)
            except Exception as e:
                errors.append({"date": current.isoformat(), "error": str(e)})
                logger.warning("V10 backfill error on %s: %s", current, e)
            current += timedelta(days=1)
            time.sleep(0.3)
        return {
            "dates_processed": (end - start).days + 1,
            "games_processed": total,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Schedule / probable pitchers fetch
    # ------------------------------------------------------------------
    def _fetch_schedule(
        self,
        target_date: date,
        game_pks: Optional[List[int]] = None,
        include_final: bool = False,
    ) -> List[dict]:
        url = f"{MLB_API}/schedule"
        params = {
            "date": target_date.isoformat(),
            "sportId": 1,
            "hydrate": "team,probablePitcher,venue,seriesStatus,seriesInformation",
            "gameType": "R,F,D,L,W",
        }
        try:
            resp = requests.get(url, params=params,
                                headers={"User-Agent": "HanksTank/2.0"},
                                timeout=30, verify=False)
            resp.raise_for_status()
        except Exception as e:
            logger.error("MLB API schedule fetch failed: %s", e)
            return []

        data = resp.json()
        games = []
        for day in data.get("dates", []):
            for g in day.get("games", []):
                pk = g["gamePk"]
                if game_pks and pk not in game_pks:
                    continue
                if not include_final and g.get("status", {}).get("abstractGameState") == "Final":
                    continue
                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})

                series_status = g.get("seriesStatus", {})
                series_info   = g.get("seriesInformation", g.get("seriesSummary", {}))
                game_number = (
                    series_status.get("gameNumber")
                    or series_info.get("seriesGameNumber")
                    or series_info.get("gameNumber")
                    or 1
                )
                games_in_series = (
                    series_status.get("seriesLength")
                    or series_info.get("totalGames")
                    or series_info.get("seriesLength")
                    or 3
                )

                games.append({
                    "game_pk": pk,
                    "game_date": day["date"],
                    "home_team_id": home.get("team", {}).get("id"),
                    "home_team_name": home.get("team", {}).get("name", ""),
                    "away_team_id": away.get("team", {}).get("id"),
                    "away_team_name": away.get("team", {}).get("name", ""),
                    "venue_id": g.get("venue", {}).get("id"),
                    "venue_name": g.get("venue", {}).get("name", ""),
                    "home_sp_id": home.get("probablePitcher", {}).get("id"),
                    "away_sp_id": away.get("probablePitcher", {}).get("id"),
                    "series_game_number": int(game_number) if game_number else 1,
                    "games_in_series": int(games_in_series) if games_in_series else 3,
                })
        return games

    # ------------------------------------------------------------------
    # Load V8 features from BQ
    # ------------------------------------------------------------------
    def _load_v8_features(self, game_pks: List[int]) -> pd.DataFrame:
        pk_list = ", ".join(str(pk) for pk in game_pks)
        sql = f"""
            SELECT *
            FROM `{V8_TABLE}`
            WHERE game_pk IN ({pk_list})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) = 1
        """
        try:
            return self.bq.query(sql).to_dataframe()
        except Exception as e:
            logger.warning("V8 features unavailable: %s — using neutral defaults", e)
            return pd.DataFrame()

    def _load_matchup_features(self, game_pks: List[int]) -> pd.DataFrame:
        pk_list = ", ".join(str(pk) for pk in game_pks)
        sql = f"""
            SELECT *
            FROM `{MATCHUP_TABLE}`
            WHERE game_pk IN ({pk_list})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) = 1
        """
        try:
            return self.bq.query(sql).to_dataframe()
        except Exception as e:
            logger.warning("Matchup features unavailable: %s — using neutral defaults", e)
            return pd.DataFrame()

    def _load_elo_direct(self) -> Dict[int, float]:
        """
        Fallback: read current Elo ratings directly from team_elo_ratings table.
        Used when game_v8_features is empty (e.g., first run of the day before
        V8 features have been built).  Returns {team_id: elo_rating}.
        """
        try:
            rows = self.bq.query(f"""
                SELECT team_id, elo_rating
                FROM `{ELO_TABLE}`
                WHERE season = EXTRACT(YEAR FROM CURRENT_DATE())
            """).result()
            return {int(r.team_id): float(r.elo_rating) for r in rows}
        except Exception as e:
            logger.warning("Could not read Elo from %s: %s", ELO_TABLE, e)
            return {}

    # ------------------------------------------------------------------
    # BQ game history for extra rolling stats
    # ------------------------------------------------------------------
    def _load_game_history(self, target_date: date) -> pd.DataFrame:
        """
        Load completed game results from ~2025-01-01 to target_date.
        Used to compute rolling windows (3g, 10g, 30g) not in V8.
        """
        cutoff = (target_date - timedelta(days=365)).isoformat()
        sql = f"""
            SELECT
              game_pk,
              game_date,
              CAST(home_team_id AS INT64) AS home_team_id,
              CAST(away_team_id AS INT64) AS away_team_id,
              CAST(home_score AS INT64) AS home_score,
              CAST(away_score AS INT64) AS away_score
            FROM `{HIST_GAMES}`
            WHERE game_date >= '{cutoff}'
              AND game_date < '{target_date.isoformat()}'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
            UNION ALL
            SELECT
              game_pk,
              game_date,
              CAST(home_team_id AS INT64) AS home_team_id,
              CAST(away_team_id AS INT64) AS away_team_id,
              CAST(home_score AS INT64) AS home_score,
              CAST(away_score AS INT64) AS away_score
            FROM `{SEASON_GAMES}`
            WHERE game_date < '{target_date.isoformat()}'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
            ORDER BY game_date
        """
        try:
            df = self.bq.query(sql).to_dataframe()
            df["game_date"] = pd.to_datetime(df["game_date"])
            return df
        except Exception as e:
            logger.warning("Game history unavailable: %s", e)
            return pd.DataFrame()

    def _compute_team_rolling(self, history: pd.DataFrame) -> Dict[int, dict]:
        """
        Compute per-team rolling statistics from historical game results.
        Returns {team_id: {win_pct_3g, win_pct_10g, win_pct_30g, win_pct_season,
                           run_diff_3g, run_diff_7g, run_diff_10g, run_diff_30g,
                           runs_scored_10g, runs_scored_30g, runs_allowed_10g, runs_allowed_30g,
                           scoring_momentum, era_proxy_10g, era_proxy_30g,
                           last_game_date, consecutive_away, road_trip_len}}
        """
        if history.empty:
            return {}

        # Build a team-centric game list (each game appears twice: once per team)
        records = []
        for _, row in history.iterrows():
            htid = int(row["home_team_id"])
            atid = int(row["away_team_id"])
            gdate = row["game_date"]
            hs, as_ = int(row["home_score"]), int(row["away_score"])
            records.append({
                "team_id": htid, "game_date": gdate,
                "rs": hs, "ra": as_, "won": int(hs > as_), "is_home": 1,
            })
            records.append({
                "team_id": atid, "game_date": gdate,
                "rs": as_, "ra": hs, "won": int(as_ > hs), "is_home": 0,
            })

        team_df = (
            pd.DataFrame(records)
            .sort_values(["team_id", "game_date"])
            .reset_index(drop=True)
        )

        stats: Dict[int, dict] = {}
        for team_id, grp in team_df.groupby("team_id"):
            g = grp.sort_values("game_date").reset_index(drop=True)
            n = len(g)

            def win_pct(last_n):
                if n == 0 or last_n <= 0:
                    return 0.5
                sub = g["won"].iloc[max(0, n - last_n):]
                return float(sub.mean()) if len(sub) > 0 else 0.5

            def run_diff_mean(last_n):
                if n == 0:
                    return 0.0
                sub = (g["rs"] - g["ra"]).iloc[max(0, n - last_n):]
                return float(sub.mean()) if len(sub) > 0 else 0.0

            def runs_mean(col, last_n):
                if n == 0:
                    return 0.0
                sub = g[col].iloc[max(0, n - last_n):]
                return float(sub.mean()) if len(sub) > 0 else 0.0

            # Scoring momentum: slope of RS over last 10 games
            def scoring_momentum(last_n=10):
                if n < 3:
                    return 0.0
                sub = g["rs"].iloc[max(0, n - last_n):].values
                if len(sub) < 2:
                    return 0.0
                x = np.arange(len(sub))
                slope = np.polyfit(x, sub, 1)[0]
                return float(np.clip(slope, -3.0, 3.0))

            # ERA proxy: runs allowed per game (lower = better pitching)
            era_10 = runs_mean("ra", 10)
            era_30 = runs_mean("ra", 30)

            # Consecutive away games at end of history
            consec_away = 0
            for i in range(len(g) - 1, -1, -1):
                if g["is_home"].iloc[i] == 0:
                    consec_away += 1
                else:
                    break

            last_date = g["game_date"].iloc[-1] if n > 0 else None

            stats[int(team_id)] = {
                "win_pct_season": win_pct(n),     # all games
                "win_pct_30g":    win_pct(30),
                "win_pct_10g":    win_pct(10),
                "win_pct_7g":     win_pct(7),
                "win_pct_3g":     win_pct(3),
                "run_diff_30g":   run_diff_mean(30),
                "run_diff_10g":   run_diff_mean(10),
                "run_diff_7g":    run_diff_mean(7),
                "run_diff_3g":    run_diff_mean(3),
                "runs_scored_30g":  runs_mean("rs", 30),
                "runs_scored_10g":  runs_mean("rs", 10),
                "runs_allowed_30g": era_30,
                "runs_allowed_10g": era_10,
                "scoring_momentum": scoring_momentum(),
                "era_proxy_10g":  era_10,
                "era_proxy_30g":  era_30,
                "last_game_date":  last_date,
                "road_trip_len":   consec_away,
                "games_played":    n,
            }
        return stats

    # ------------------------------------------------------------------
    # Team quality from MLB Stats API
    # ------------------------------------------------------------------
    def _fetch_team_quality(self, season: int) -> Dict[int, dict]:
        """
        Fetch team season pitching and batting stats from MLB Stats API.
        Returns {team_id: {fg_era, fg_whip, fg_k9, fg_bb9, fg_k_pct, fg_bb_pct,
                            fg_obp, fg_slg, fg_ops, fg_woba (≈ obp), ...}}
        Statcast-only features (xFIP, whiff%, barrel%, EV%, HH%) default to 50.0
        (league-average percentile rank) since they're unavailable from the MLB API.
        """
        if self._team_quality:
            return self._team_quality

        quality: Dict[int, dict] = {}

        # ---- Pitching stats ----
        try:
            resp = requests.get(
                f"{MLB_API}/teams/stats",
                params={"season": season, "sportId": 1, "stats": "season",
                        "group": "pitching", "gameType": "R"},
                headers={"User-Agent": "HanksTank/2.0"},
                timeout=20, verify=False,
            )
            resp.raise_for_status()
            for stat_block in resp.json().get("stats", []):
                for split in stat_block.get("splits", []):
                    tid = split["team"]["id"]
                    s = split.get("stat", {})
                    if tid not in quality:
                        quality[tid] = {}

                    era = _safe_float(s.get("era"))
                    whip = _safe_float(s.get("whip"))
                    ks = _safe_float(s.get("strikeOuts", 0))
                    bbs = _safe_float(s.get("baseOnBalls", 0))
                    ip_str = s.get("inningsPitched", "0.0")
                    try:
                        # "123.1" → 123 + 1/3 = 123.33 innings
                        parts = str(ip_str).split(".")
                        ip = float(parts[0]) + float(parts[1]) / 3 if len(parts) > 1 else float(parts[0])
                    except Exception:
                        ip = 0.0

                    bf = _safe_float(s.get("battersFaced", 0)) or max(ip * 4.3, 1)
                    k9  = (ks / ip * 9)  if ip > 0 else 8.0
                    bb9 = (bbs / ip * 9) if ip > 0 else 3.0
                    k_pct  = (ks  / bf)  if bf > 0 else 0.22
                    bb_pct = (bbs / bf)  if bf > 0 else 0.08

                    quality[tid].update({
                        "fg_era":    era  if era  is not None else 4.20,
                        "fg_whip":   whip if whip is not None else 1.30,
                        "fg_k9":     k9,
                        "fg_bb9":    bb9,
                        "fg_k_pct":  k_pct,
                        "fg_bb_pct": bb_pct,
                        # xFIP proxy: use ERA (Statcast xFIP not in MLB API)
                        "fg_xfip":   era if era is not None else 4.20,
                        # Statcast-only: default to league average
                        "fg_whiff_pct": 50.0,
                        "fg_fbv_pct":   50.0,
                    })
        except Exception as e:
            logger.warning("MLB API pitching stats unavailable: %s — using defaults", e)

        # ---- Batting stats ----
        try:
            resp = requests.get(
                f"{MLB_API}/teams/stats",
                params={"season": season, "sportId": 1, "stats": "season",
                        "group": "hitting", "gameType": "R"},
                headers={"User-Agent": "HanksTank/2.0"},
                timeout=20, verify=False,
            )
            resp.raise_for_status()
            for stat_block in resp.json().get("stats", []):
                for split in stat_block.get("splits", []):
                    tid = split["team"]["id"]
                    s = split.get("stat", {})
                    if tid not in quality:
                        quality[tid] = {}
                    obp = _safe_float(s.get("obp"))
                    slg = _safe_float(s.get("slg"))
                    ops = _safe_float(s.get("ops"))
                    quality[tid].update({
                        "fg_obp":    obp if obp is not None else 0.315,
                        "fg_slg":    slg if slg is not None else 0.410,
                        "fg_ops":    ops if ops is not None else 0.725,
                        "fg_woba":   obp if obp is not None else 0.315,  # OBP ≈ wOBA proxy
                        # Statcast batting — not in MLB API
                        "fg_ev_pct":  50.0,
                        "fg_hh_pct":  50.0,
                        "fg_brl_pct": 50.0,
                    })
        except Exception as e:
            logger.warning("MLB API batting stats unavailable: %s — using defaults", e)

        self._team_quality = quality
        logger.info("Team quality loaded for %d teams (season %s)", len(quality), season)
        return quality

    def _team_stats_for(self, team_id: int, quality: Dict) -> dict:
        """Return quality stats for a team, with league-average defaults."""
        return quality.get(team_id, {
            "fg_era": 4.20, "fg_whip": 1.30, "fg_k9": 8.0, "fg_bb9": 3.0,
            "fg_xfip": 4.20, "fg_k_pct": 0.22, "fg_bb_pct": 0.08,
            "fg_whiff_pct": 50.0, "fg_fbv_pct": 50.0,
            "fg_obp": 0.315, "fg_slg": 0.410, "fg_ops": 0.725,
            "fg_woba": 0.315, "fg_ev_pct": 50.0, "fg_hh_pct": 50.0, "fg_brl_pct": 50.0,
        })

    # ------------------------------------------------------------------
    # SP quality from local Statcast parquet / GCS
    # ------------------------------------------------------------------
    def _load_sp_lookup(self) -> Dict:
        """
        Build {(player_id, season): {xera, k_pct, bb_pct, whiff, fbv}} from
        statcast_sp_pct_YEAR.parquet files (Baseball Savant percentile ranks, 0–100).
        Tries local paths first, then GCS bucket fallback.
        """
        if self._sp_lookup:
            return self._sp_lookup

        lookup: Dict = {}
        years = list(range(2015, 2027))

        for year in years:
            df = None
            # Try local paths
            for d in [_SP_DATA_DIR, _V10_SP_DIR]:
                fp = d / f"statcast_sp_pct_{year}.parquet"
                if fp.exists():
                    try:
                        df = pd.read_parquet(fp)
                        break
                    except Exception as e:
                        logger.debug("Error reading %s: %s", fp, e)

            # GCS fallback
            if df is None:
                try:
                    from google.cloud import storage as gcs
                    client = gcs.Client(project=PROJECT)
                    bucket = client.bucket("hanks_tank_data")
                    blob = bucket.blob(f"data/statcast_sp_pct_{year}.parquet")
                    import io
                    df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
                except Exception:
                    pass  # silently skip years unavailable in GCS

            if df is None:
                continue

            season_col = "season" if "season" in df.columns else "year"
            for _, row in df.iterrows():
                pid = row.get("player_id")
                s   = row.get(season_col, year)
                if pd.isna(pid) or pd.isna(s):
                    continue
                lookup[(int(pid), int(s))] = {
                    "xera":  float(row["xera"])          if not pd.isna(row.get("xera"))          else np.nan,
                    "k_pct": float(row["k_percent"])     if not pd.isna(row.get("k_percent"))     else np.nan,
                    "bb_pct":float(row["bb_percent"])    if not pd.isna(row.get("bb_percent"))    else np.nan,
                    "whiff": float(row["whiff_percent"]) if not pd.isna(row.get("whiff_percent")) else np.nan,
                    "fbv":   float(row["fb_velocity"])   if not pd.isna(row.get("fb_velocity"))   else np.nan,
                }

        logger.info("SP quality lookup: %d pitcher-season entries", len(lookup))
        self._sp_lookup = lookup
        return lookup

    def _get_sp_stats(
        self, player_id: Optional[int], season: int, sp_lookup: Dict
    ) -> tuple:
        """
        Return (stats_dict, is_known).
        Tries current year, then prior year as fallback.
        Missing metrics are imputed with 50.0 (league average).
        """
        if player_id is None:
            return {m: SP_DEFAULT[m] for m in SP_METRICS}, False
        for yr in [season, season - 1]:
            entry = sp_lookup.get((player_id, yr))
            if entry is not None:
                result = {}
                any_real = False
                for m in SP_METRICS:
                    v = entry.get(m, np.nan)
                    if not np.isnan(v):
                        result[m] = v
                        any_real = True
                    else:
                        result[m] = SP_DEFAULT[m]
                return result, any_real
        return {m: SP_DEFAULT[m] for m in SP_METRICS}, False

    # ------------------------------------------------------------------
    # Park factors
    # ------------------------------------------------------------------
    def _park_factors_for(self, venue_id: Optional[int]) -> dict:
        if venue_id is None:
            return {"ratio": 1.0, "ratio_100": 100.0, "known": 0}
        entry = PARK_FACTORS.get(int(venue_id))
        if entry:
            return {"ratio": entry[0], "ratio_100": entry[1], "known": 1}
        return {"ratio": 1.0, "ratio_100": 100.0, "known": 0}

    # ------------------------------------------------------------------
    # Core assembly
    # ------------------------------------------------------------------
    def _build_and_write(self, games: List[dict], target_date: date) -> dict:
        game_pks = [g["game_pk"] for g in games]
        season = target_date.year

        # Load all data sources
        v8_df        = self._load_v8_features(game_pks)
        matchup_df   = self._load_matchup_features(game_pks)
        # If V8 features are absent (e.g., V10 ran before V8 today), fall back
        # to Elo ratings table so predictions have real team-strength context.
        elo_direct   = self._load_elo_direct() if v8_df.empty else {}
        history      = self._load_game_history(target_date)
        team_rolling = self._compute_team_rolling(history)
        team_quality = self._fetch_team_quality(season)
        sp_lookup    = self._load_sp_lookup()

        # Season game number: how many games has each team played so far?
        team_game_count: Dict[int, int] = {}
        if not history.empty:
            for _, row in history.iterrows():
                h, a = int(row["home_team_id"]), int(row["away_team_id"])
                team_game_count[h] = team_game_count.get(h, 0) + 1
                team_game_count[a] = team_game_count.get(a, 0) + 1

        rows = []
        for game in games:
            row = self._assemble_row(
                game, v8_df, matchup_df, team_rolling, team_quality, sp_lookup,
                team_game_count, target_date, season, elo_direct
            )
            rows.append(row)

        if rows and not self.dry_run:
            self._write_to_bq(rows, target_date)

        return {"games_processed": len(rows), "date": target_date.isoformat()}

    def _assemble_row(
        self,
        game: dict,
        v8_df: pd.DataFrame,
        matchup_df: pd.DataFrame,
        team_rolling: Dict,
        team_quality: Dict,
        sp_lookup: Dict,
        team_game_count: Dict,
        target_date: date,
        season: int,
        elo_direct: Dict[int, float] = None,
    ) -> dict:
        pk   = game["game_pk"]
        htid = game["home_team_id"]
        atid = game["away_team_id"]
        gdate = game["game_date"]

        # --- V8 features (Elo, Pythagorean, streaks, H2H) ---
        v8_row = None
        if not v8_df.empty and "game_pk" in v8_df.columns:
            sub = v8_df[v8_df["game_pk"] == pk]
            if not sub.empty:
                v8_row = sub.iloc[0]

        def v8(col, default=0.0):
            if v8_row is not None and col in v8_row.index:
                val = v8_row[col]
                if pd.notna(val):
                    return float(val)
            return default

        def v8i(col, default=0):
            if v8_row is not None and col in v8_row.index:
                val = v8_row[col]
                if pd.notna(val):
                    return int(val)
            return default

        matchup_row = None
        if not matchup_df.empty and "game_pk" in matchup_df.columns:
            sub = matchup_df[matchup_df["game_pk"] == pk]
            if not sub.empty:
                matchup_row = sub.iloc[0]

        def matchup(col, default=0.0):
            if matchup_row is not None and col in matchup_row.index:
                val = matchup_row[col]
                if pd.notna(val):
                    return float(val)
            return default

        def matchupi(col, default=0):
            if matchup_row is not None and col in matchup_row.index:
                val = matchup_row[col]
                if pd.notna(val):
                    return int(bool(val))
            return default

        # --- Rolling stats from game history ---
        hr = team_rolling.get(htid, {})
        ar = team_rolling.get(atid, {})

        # --- Team quality ---
        hq = self._team_stats_for(htid, team_quality)
        aq = self._team_stats_for(atid, team_quality)

        # --- SP quality ---
        home_sp_id = game.get("home_sp_id")
        away_sp_id = game.get("away_sp_id")
        home_sp, home_known = self._get_sp_stats(home_sp_id, season, sp_lookup)
        away_sp, away_known = self._get_sp_stats(away_sp_id, season, sp_lookup)

        # --- Park factors ---
        venue_id = game.get("venue_id")
        pf = self._park_factors_for(venue_id)

        # --- Rest / travel ---
        target_dt = pd.Timestamp(target_date)
        home_last = hr.get("last_game_date")
        away_last = ar.get("last_game_date")
        home_rest = float(min((target_dt - home_last).days, 7)) if home_last is not None else 3.0
        away_rest = float(min((target_dt - away_last).days, 7)) if away_last is not None else 3.0
        away_road = float(ar.get("road_trip_len", 0))

        # --- Calendar ---
        gd = pd.Timestamp(gdate)
        day_of_week = int(gd.dayofweek)
        month = int(gd.month)

        # --- Season context ---
        home_gp = team_game_count.get(htid, 0)
        away_gp = team_game_count.get(atid, 0)
        avg_gp  = (home_gp + away_gp) / 2 if (home_gp + away_gp) > 0 else 5
        season_pct = float(np.clip(avg_gp / SEASON_GAMES_TOTAL, 0.0, 1.0))

        home_lineup_woba = matchup("home_lineup_woba_vs_hand", 0.320)
        away_lineup_woba = matchup("away_lineup_woba_vs_hand", 0.320)
        home_lineup_k_pct = matchup("home_lineup_k_pct_vs_hand", 0.220)
        away_lineup_k_pct = matchup("away_lineup_k_pct_vs_hand", 0.220)
        home_top3_woba = matchup("home_top3_woba_vs_hand", home_lineup_woba)
        away_top3_woba = matchup("away_top3_woba_vs_hand", away_lineup_woba)
        home_middle4_woba = matchup("home_middle4_woba_vs_hand", home_lineup_woba)
        away_middle4_woba = matchup("away_middle4_woba_vs_hand", away_lineup_woba)
        home_bottom2_woba = matchup("home_bottom2_woba_vs_hand", home_lineup_woba)
        away_bottom2_woba = matchup("away_bottom2_woba_vs_hand", away_lineup_woba)
        home_pct_same_hand = matchup("home_pct_same_hand", 0.50)
        away_pct_same_hand = matchup("away_pct_same_hand", 0.50)
        home_h2h_woba = matchup("home_h2h_woba", home_lineup_woba)
        away_h2h_woba = matchup("away_h2h_woba", away_lineup_woba)
        matchup_advantage_home = matchup(
            "matchup_advantage_home",
            home_lineup_woba - away_lineup_woba,
        )

        # Series context
        series_game_number = int(game.get("series_game_number", 1))
        games_in_series    = int(game.get("games_in_series", 3))

        # Elo fallback: use team_elo_ratings table when V8 features are absent
        import math as _math
        _elo_home = elo_direct.get(htid, 1500.0) if elo_direct else 1500.0
        _elo_away = elo_direct.get(atid, 1500.0) if elo_direct else 1500.0
        _elo_diff = _elo_home - _elo_away
        _elo_prob = 1.0 / (1.0 + _math.pow(10, -_elo_diff / 400.0))

        # --- Assemble ---
        row = {
            "game_pk": pk,
            "game_date": gdate,
            "home_team_id": htid,
            "away_team_id": atid,
            # Elo from V8 features (preferred), or team_elo_ratings table
            # (when V8 features not yet built for today), or neutral 1500.
            "home_elo":               v8("home_elo", _elo_home),
            "away_elo":               v8("away_elo", _elo_away),
            "elo_differential":       v8("elo_differential", _elo_diff),
            "elo_home_win_prob":      v8("elo_home_win_prob", _elo_prob),
            "elo_win_prob_differential": v8("elo_win_prob_differential", _elo_prob - 0.5),
            # Pythagorean (from V8)
            "home_pythag_season":  v8("home_pythag_season", 0.5),
            "away_pythag_season":  v8("away_pythag_season", 0.5),
            "home_pythag_last30":  v8("home_pythag_last30", 0.5),
            "away_pythag_last30":  v8("away_pythag_last30", 0.5),
            "pythag_differential": v8("pythag_differential", 0.0),
            "home_luck_factor":    v8("home_luck_factor", 0.0),
            "away_luck_factor":    v8("away_luck_factor", 0.0),
            "luck_differential":   v8("luck_differential", 0.0),
            # Rolling form (prefer game history, fall back to V8)
            "home_win_pct_3g":  hr.get("win_pct_3g",    v8("home_win_pct_7g", 0.5)),
            "away_win_pct_3g":  ar.get("win_pct_3g",    v8("away_win_pct_7g", 0.5)),
            "home_win_pct_7g":  hr.get("win_pct_7g",    v8("home_win_pct_7g", 0.5)),
            "away_win_pct_7g":  ar.get("win_pct_7g",    v8("away_win_pct_7g", 0.5)),
            "home_win_pct_10g": hr.get("win_pct_10g",   v8("home_win_pct_season", 0.5)),
            "away_win_pct_10g": ar.get("win_pct_10g",   v8("away_win_pct_season", 0.5)),
            "home_win_pct_30g": hr.get("win_pct_30g",   v8("home_win_pct_season", 0.5)),
            "away_win_pct_30g": ar.get("win_pct_30g",   v8("away_win_pct_season", 0.5)),
            "home_win_pct_season": hr.get("win_pct_season", v8("home_win_pct_season", 0.5)),
            "away_win_pct_season": ar.get("win_pct_season", v8("away_win_pct_season", 0.5)),
            "win_pct_diff":        hr.get("win_pct_season", 0.5) - ar.get("win_pct_season", 0.5),
            # Run differential
            "home_run_diff_3g":  hr.get("run_diff_3g",  0.0),
            "away_run_diff_3g":  ar.get("run_diff_3g",  0.0),
            "home_run_diff_7g":  hr.get("run_diff_7g",  0.0),
            "away_run_diff_7g":  ar.get("run_diff_7g",  0.0),
            "home_run_diff_10g": hr.get("run_diff_10g", v8("home_run_diff_10g", 0.0)),
            "away_run_diff_10g": ar.get("run_diff_10g", v8("away_run_diff_10g", 0.0)),
            "home_run_diff_30g": hr.get("run_diff_30g", v8("home_run_diff_30g", 0.0)),
            "away_run_diff_30g": ar.get("run_diff_30g", v8("away_run_diff_30g", 0.0)),
            "run_diff_differential": hr.get("run_diff_10g", 0.0) - ar.get("run_diff_10g", 0.0),
            # Runs scored / allowed
            "home_runs_scored_10g":  hr.get("runs_scored_10g",  4.5),
            "away_runs_scored_10g":  ar.get("runs_scored_10g",  4.5),
            "home_runs_scored_30g":  hr.get("runs_scored_30g",  4.5),
            "away_runs_scored_30g":  ar.get("runs_scored_30g",  4.5),
            "home_runs_allowed_10g": hr.get("runs_allowed_10g", 4.5),
            "away_runs_allowed_10g": ar.get("runs_allowed_10g", 4.5),
            "home_runs_allowed_30g": hr.get("runs_allowed_30g", 4.5),
            "away_runs_allowed_30g": ar.get("runs_allowed_30g", 4.5),
            "home_scoring_momentum": hr.get("scoring_momentum", v8("home_scoring_momentum", 0.0)),
            "away_scoring_momentum": ar.get("scoring_momentum", v8("away_scoring_momentum", 0.0)),
            "home_era_proxy_10g":  hr.get("era_proxy_10g", v8("home_era_proxy_10g", 4.5)),
            "away_era_proxy_10g":  ar.get("era_proxy_10g", v8("away_era_proxy_10g", 4.5)),
            "home_era_proxy_30g":  hr.get("era_proxy_30g", v8("home_era_proxy_30g", 4.5)),
            "away_era_proxy_30g":  ar.get("era_proxy_30g", v8("away_era_proxy_30g", 4.5)),
            "era_proxy_differential": ar.get("era_proxy_10g", 4.5) - hr.get("era_proxy_10g", 4.5),
            # Streaks (from V8 — most accurate since it tracks intra-season)
            "home_current_streak":  v8i("home_current_streak", 0),
            "away_current_streak":  v8i("away_current_streak", 0),
            "home_streak_direction": v8i("home_streak_direction", 0),
            "away_streak_direction": v8i("away_streak_direction", 0),
            "streak_differential":  v8i("streak_differential", 0),
            # H2H (from V8)
            "home_h2h_win_pct_season": v8("h2h_win_pct_season", 0.5),
            "home_h2h_games_season":   v8i("h2h_games_3yr", 0),  # best available from V8
            "home_h2h_win_pct_3yr":    v8("h2h_win_pct_3yr", 0.5),
            # Matchup / lineup quality
            "lineup_confirmed": matchupi("lineup_confirmed", 0),
            "home_lineup_woba_vs_hand": home_lineup_woba,
            "away_lineup_woba_vs_hand": away_lineup_woba,
            "lineup_woba_differential": home_lineup_woba - away_lineup_woba,
            "home_lineup_k_pct_vs_hand": home_lineup_k_pct,
            "away_lineup_k_pct_vs_hand": away_lineup_k_pct,
            "lineup_k_pct_differential": away_lineup_k_pct - home_lineup_k_pct,
            "home_top3_woba_vs_hand": home_top3_woba,
            "away_top3_woba_vs_hand": away_top3_woba,
            "home_middle4_woba_vs_hand": home_middle4_woba,
            "away_middle4_woba_vs_hand": away_middle4_woba,
            "home_bottom2_woba_vs_hand": home_bottom2_woba,
            "away_bottom2_woba_vs_hand": away_bottom2_woba,
            "home_pct_same_hand": home_pct_same_hand,
            "away_pct_same_hand": away_pct_same_hand,
            "home_h2h_woba": home_h2h_woba,
            "away_h2h_woba": away_h2h_woba,
            "h2h_woba_differential": home_h2h_woba - away_h2h_woba,
            "matchup_advantage_home": matchup_advantage_home,
            # Team quality (from MLB API)
            "home_fg_era":      hq.get("fg_era", 4.20),
            "away_fg_era":      aq.get("fg_era", 4.20),
            "home_fg_whip":     hq.get("fg_whip", 1.30),
            "away_fg_whip":     aq.get("fg_whip", 1.30),
            "home_fg_k9":       hq.get("fg_k9", 8.0),
            "away_fg_k9":       aq.get("fg_k9", 8.0),
            "home_fg_bb9":      hq.get("fg_bb9", 3.0),
            "away_fg_bb9":      aq.get("fg_bb9", 3.0),
            "home_fg_xfip":     hq.get("fg_xfip", 4.20),
            "away_fg_xfip":     aq.get("fg_xfip", 4.20),
            "home_fg_k_pct":    hq.get("fg_k_pct", 0.22),
            "away_fg_k_pct":    aq.get("fg_k_pct", 0.22),
            "home_fg_bb_pct":   hq.get("fg_bb_pct", 0.08),
            "away_fg_bb_pct":   aq.get("fg_bb_pct", 0.08),
            "home_fg_whiff_pct": hq.get("fg_whiff_pct", 50.0),
            "away_fg_whiff_pct": aq.get("fg_whiff_pct", 50.0),
            "home_fg_fbv_pct":  hq.get("fg_fbv_pct", 50.0),
            "away_fg_fbv_pct":  aq.get("fg_fbv_pct", 50.0),
            "fg_era_differential":  aq.get("fg_era", 4.20) - hq.get("fg_era", 4.20),
            "fg_xfip_differential": aq.get("fg_xfip", 4.20) - hq.get("fg_xfip", 4.20),
            "fg_whip_differential": aq.get("fg_whip", 1.30) - hq.get("fg_whip", 1.30),
            "home_fg_ops":      hq.get("fg_ops", 0.725),
            "away_fg_ops":      aq.get("fg_ops", 0.725),
            "home_fg_obp":      hq.get("fg_obp", 0.315),
            "away_fg_obp":      aq.get("fg_obp", 0.315),
            "home_fg_slg":      hq.get("fg_slg", 0.410),
            "away_fg_slg":      aq.get("fg_slg", 0.410),
            "home_fg_woba":     hq.get("fg_woba", 0.315),
            "away_fg_woba":     aq.get("fg_woba", 0.315),
            "home_fg_ev_pct":   hq.get("fg_ev_pct", 50.0),
            "away_fg_ev_pct":   aq.get("fg_ev_pct", 50.0),
            "home_fg_hh_pct":   hq.get("fg_hh_pct", 50.0),
            "away_fg_hh_pct":   aq.get("fg_hh_pct", 50.0),
            "home_fg_brl_pct":  hq.get("fg_brl_pct", 50.0),
            "away_fg_brl_pct":  aq.get("fg_brl_pct", 50.0),
            "fg_ops_differential":  hq.get("fg_ops", 0.725) - aq.get("fg_ops", 0.725),
            "fg_woba_differential": hq.get("fg_woba", 0.315) - aq.get("fg_woba", 0.315),
            "fg_obp_differential":  hq.get("fg_obp", 0.315) - aq.get("fg_obp", 0.315),
            # Park factors
            "home_park_factor":     pf["ratio"],
            "home_park_factor_100": pf["ratio_100"],
            "park_factor_known":    pf["known"],
            "is_hitter_park":       int(pf["ratio"] > 1.05),
            "is_pitcher_park":      int(pf["ratio"] < 0.95),
            # Calendar
            "day_of_week": day_of_week,
            "month":       month,
            "is_weekend":  int(day_of_week >= 4),
            "month_3":  int(month == 3),  "month_4":  int(month == 4),
            "month_5":  int(month == 5),  "month_6":  int(month == 6),
            "month_7":  int(month == 7),  "month_8":  int(month == 8),
            "month_9":  int(month == 9),  "month_10": int(month == 10),
            # Season context
            "season_game_number": int(avg_gp),
            "season_pct_complete": season_pct,
            "is_late_season":   int(season_pct >= 0.75),
            "is_early_season":  int(season_pct <= 0.20),
            # SP quality (V10 NEW)
            "home_sp_xera":  home_sp["xera"],
            "away_sp_xera":  away_sp["xera"],
            "sp_xera_diff":  home_sp["xera"] - away_sp["xera"],
            "home_sp_k_pct": home_sp["k_pct"],
            "away_sp_k_pct": away_sp["k_pct"],
            "sp_k_pct_diff": home_sp["k_pct"] - away_sp["k_pct"],
            "home_sp_bb_pct": home_sp["bb_pct"],
            "away_sp_bb_pct": away_sp["bb_pct"],
            "sp_bb_pct_diff": home_sp["bb_pct"] - away_sp["bb_pct"],
            "home_sp_whiff": home_sp["whiff"],
            "away_sp_whiff": away_sp["whiff"],
            "sp_whiff_diff": home_sp["whiff"] - away_sp["whiff"],
            "home_sp_fbv":   home_sp["fbv"],
            "away_sp_fbv":   away_sp["fbv"],
            "sp_fbv_diff":   home_sp["fbv"] - away_sp["fbv"],
            "sp_quality_composite_diff": sum(
                SP_COMPOSITE_WEIGHTS[m] * (home_sp[m] - away_sp[m]) for m in SP_METRICS
            ),
            "home_sp_known": int(home_known),
            "away_sp_known": int(away_known),
            # Rest / travel (V10 NEW)
            "home_days_rest":     home_rest,
            "away_days_rest":     away_rest,
            "rest_differential":  home_rest - away_rest,
            "away_road_trip_length": away_road,
            "home_rested":    int(home_rest >= 2),
            "away_tired":     int(away_rest <= 1),
            "long_road_trip": int(away_road >= 5),
            # Series context (V10 NEW)
            "series_game_number": series_game_number,
            "games_in_series":    games_in_series,
            "is_series_opener":   int(series_game_number == 1),
            # Metadata
            "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        logger.info(
            "V10 features: %s (%s) vs %s (%s) | SP: %s/%.0f vs %s/%.0f | "
            "Park: %.3f | Rest: %.0f/%.0f",
            game["home_team_name"], htid,
            game["away_team_name"], atid,
            "known" if home_known else "default", home_sp["xera"],
            "known" if away_known else "default", away_sp["xera"],
            pf["ratio"],
            home_rest, away_rest,
        )
        return row

    # ------------------------------------------------------------------
    # BQ write
    # ------------------------------------------------------------------
    def _write_to_bq(self, rows: List[dict], target_date: date) -> None:
        import io
        import json as _json

        pk_list = ", ".join(str(r["game_pk"]) for r in rows)
        try:
            self.bq.query(
                f"DELETE FROM `{V10_TABLE}` "
                f"WHERE game_pk IN ({pk_list}) "
                f"AND DATE(computed_at) = '{target_date.isoformat()}'"
            ).result()
        except Exception as exc:
            if "streaming buffer" in str(exc).lower() or "Not found" in str(exc):
                pass
            else:
                logger.warning("DELETE from V10 table failed (non-fatal): %s", exc)

        ndjson = io.BytesIO(
            "\n".join(_json.dumps(r, default=str) for r in rows).encode()
        )
        job_cfg = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema=V10_FEATURES_SCHEMA,
        )
        job = self.bq.load_table_from_file(ndjson, V10_TABLE, job_config=job_cfg)
        job.result()
        if job.errors:
            logger.error("V10 BQ write errors: %s", job.errors)
        else:
            logger.info("Wrote %d V10 feature rows to BigQuery", len(rows))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build V10 live features for upcoming games")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--game-pk", help="Comma-separated game PKs")
    parser.add_argument("--backfill", action="store_true", help="Backfill from --start to --end")
    parser.add_argument("--start", help="Backfill start date (YYYY-MM-DD)")
    parser.add_argument("--end",   help="Backfill end date (YYYY-MM-DD, default: today)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-final", action="store_true",
                        help="Include completed/final games (for historical backfills)")
    args = parser.parse_args()

    builder = V10LiveFeatureBuilder(dry_run=args.dry_run)

    if args.backfill:
        start = date.fromisoformat(args.start or "2026-03-27")
        end   = date.fromisoformat(args.end or date.today().isoformat())
        result = builder.run_backfill(start, end)
    elif args.game_pk:
        pks = [int(pk.strip()) for pk in args.game_pk.split(",")]
        target = date.fromisoformat(args.date) if args.date else date.today()
        result = builder.run_for_game_pks(pks, target, include_final=args.include_final)
    else:
        target = date.fromisoformat(args.date) if args.date else date.today()
        result = builder.run_for_date(target, include_final=args.include_final)

    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()
