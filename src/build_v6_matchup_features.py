#!/usr/bin/env python3
"""
V6 Matchup Feature Builder

Extends V5 matchup features with:
  1. Pitcher arsenal / stuff quality (pitch mix, velocity, spin, K-BB%, xwOBA)
     - Sourced from mlb_historical_data.pitcher_game_stats (rolling 30-day avg)
  2. Batter venue history (per-batter wOBA at this specific park)
     - Sourced from mlb_historical_data.player_venue_splits

New V6 columns added on top of existing matchup_features schema:

  Pitcher arsenal (home and away starter):
    home_starter_fastball_pct      away_starter_fastball_pct
    home_starter_breaking_pct      away_starter_breaking_pct
    home_starter_offspeed_pct      away_starter_offspeed_pct
    home_starter_mean_velo         away_starter_mean_velo
    home_starter_velo_norm         away_starter_velo_norm  (velo - 93.0) / 3.0
    home_starter_velo_trend        away_starter_velo_trend (recent 2 vs prev 3 starts)
    home_starter_k_bb_pct          away_starter_k_bb_pct
    home_starter_xwoba_allowed     away_starter_xwoba_allowed
    starter_arsenal_advantage      composite quality diff (home minus away)

  Venue history (lineup at this specific park):
    home_lineup_venue_woba         home batters' wOBA at this park (career)
    away_lineup_venue_woba         away batters' wOBA at this park (career)
    venue_woba_differential        home_lineup_venue_woba - away_lineup_venue_woba
    home_venue_advantage           home batters' wOBA at park minus their season wOBA
    away_venue_disadvantage        away batters' wOBA at park minus their season wOBA

Output table: mlb_2026_season.matchup_v6_features
  (all V5 matchup_features columns + the 17 new V6 columns above,
   plus common JOIN keys: game_pk, game_date, home_team_id, away_team_id)

Usage:
    python build_v6_matchup_features.py                    # today's games
    python build_v6_matchup_features.py --date 2026-04-03
    python build_v6_matchup_features.py --game-pk 824786,824621
    python build_v6_matchup_features.py --dry-run
"""

import argparse
import logging
import warnings
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import urllib3
from google.cloud import bigquery

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
HIST_DS = "mlb_historical_data"
SEASON_DS = "mlb_2026_season"
MLB_API = "https://statsapi.mlb.com/api/v1"

MATCHUP_V5_TABLE = f"{PROJECT}.{SEASON_DS}.matchup_features"
MATCHUP_V6_TABLE = f"{PROJECT}.{SEASON_DS}.matchup_v6_features"
LINEUPS_TABLE = f"{PROJECT}.{SEASON_DS}.lineups"
PITCHER_STATS_TABLE = f"{PROJECT}.{HIST_DS}.pitcher_game_stats"
VENUE_SPLITS_TABLE = f"{PROJECT}.{HIST_DS}.player_venue_splits"
SEASON_SPLITS_TABLE = f"{PROJECT}.{HIST_DS}.player_season_splits"
GAMES_HIST_TABLE = f"{PROJECT}.{HIST_DS}.games_historical"

# Rolling window for pitcher arsenal features
PITCHER_LOOKBACK_DAYS = 30
# Minimum pitch count per start for arsenal features to be reliable
MIN_PA_VENUE = 15
# League-average fastball velocity for normalisation
LEAGUE_AVG_VELO = 93.0
VELO_STD = 3.0

# New V6 schema fields (appended to V5 schema)
V6_EXTRA_SCHEMA = [
    bigquery.SchemaField("home_starter_fastball_pct",   "FLOAT"),
    bigquery.SchemaField("away_starter_fastball_pct",   "FLOAT"),
    bigquery.SchemaField("home_starter_breaking_pct",   "FLOAT"),
    bigquery.SchemaField("away_starter_breaking_pct",   "FLOAT"),
    bigquery.SchemaField("home_starter_offspeed_pct",   "FLOAT"),
    bigquery.SchemaField("away_starter_offspeed_pct",   "FLOAT"),
    bigquery.SchemaField("home_starter_mean_velo",      "FLOAT"),
    bigquery.SchemaField("away_starter_mean_velo",      "FLOAT"),
    bigquery.SchemaField("home_starter_velo_norm",      "FLOAT"),
    bigquery.SchemaField("away_starter_velo_norm",      "FLOAT"),
    bigquery.SchemaField("home_starter_velo_trend",     "FLOAT"),
    bigquery.SchemaField("away_starter_velo_trend",     "FLOAT"),
    bigquery.SchemaField("home_starter_k_bb_pct",       "FLOAT"),
    bigquery.SchemaField("away_starter_k_bb_pct",       "FLOAT"),
    bigquery.SchemaField("home_starter_xwoba_allowed",  "FLOAT"),
    bigquery.SchemaField("away_starter_xwoba_allowed",  "FLOAT"),
    bigquery.SchemaField("starter_arsenal_advantage",   "FLOAT"),
    bigquery.SchemaField("home_lineup_venue_woba",      "FLOAT"),
    bigquery.SchemaField("away_lineup_venue_woba",      "FLOAT"),
    bigquery.SchemaField("venue_woba_differential",     "FLOAT"),
    bigquery.SchemaField("home_venue_advantage",        "FLOAT"),
    bigquery.SchemaField("away_venue_disadvantage",     "FLOAT"),
]

# Build full V6 schema (copy V5 schema + add V6 extras)
# We'll fetch V5 schema dynamically in _ensure_table
V6_BASE_SCHEMA_FIELDS = [
    bigquery.SchemaField("game_pk",        "INTEGER"),
    bigquery.SchemaField("game_date",      "DATE"),
    bigquery.SchemaField("home_team_id",   "INTEGER"),
    bigquery.SchemaField("away_team_id",   "INTEGER"),
    bigquery.SchemaField("venue_id",       "INTEGER"),
    bigquery.SchemaField("venue_name",     "STRING"),
    # V5 matchup fields
    bigquery.SchemaField("home_lineup_woba_vs_hand",  "FLOAT"),
    bigquery.SchemaField("home_lineup_k_pct_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_lineup_bb_pct_vs_hand","FLOAT"),
    bigquery.SchemaField("home_lineup_iso_vs_hand",   "FLOAT"),
    bigquery.SchemaField("home_top3_woba_vs_hand",    "FLOAT"),
    bigquery.SchemaField("home_middle4_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_bottom2_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_pct_same_hand",        "FLOAT"),
    bigquery.SchemaField("home_h2h_woba",             "FLOAT"),
    bigquery.SchemaField("home_h2h_pa_total",         "INTEGER"),
    bigquery.SchemaField("home_h2h_k_pct",            "FLOAT"),
    bigquery.SchemaField("home_h2h_hr_rate",          "FLOAT"),
    bigquery.SchemaField("away_lineup_woba_vs_hand",  "FLOAT"),
    bigquery.SchemaField("away_lineup_k_pct_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_lineup_bb_pct_vs_hand","FLOAT"),
    bigquery.SchemaField("away_lineup_iso_vs_hand",   "FLOAT"),
    bigquery.SchemaField("away_top3_woba_vs_hand",    "FLOAT"),
    bigquery.SchemaField("away_middle4_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_bottom2_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_pct_same_hand",        "FLOAT"),
    bigquery.SchemaField("away_h2h_woba",             "FLOAT"),
    bigquery.SchemaField("away_h2h_pa_total",         "INTEGER"),
    bigquery.SchemaField("away_h2h_k_pct",            "FLOAT"),
    bigquery.SchemaField("away_h2h_hr_rate",          "FLOAT"),
    bigquery.SchemaField("home_starter_id",           "INTEGER"),
    bigquery.SchemaField("home_starter_hand",         "STRING"),
    bigquery.SchemaField("home_starter_woba_allowed", "FLOAT"),
    bigquery.SchemaField("home_starter_k_pct",        "FLOAT"),
    bigquery.SchemaField("home_starter_whip",         "FLOAT"),
    bigquery.SchemaField("away_starter_id",           "INTEGER"),
    bigquery.SchemaField("away_starter_hand",         "STRING"),
    bigquery.SchemaField("away_starter_woba_allowed", "FLOAT"),
    bigquery.SchemaField("away_starter_k_pct",        "FLOAT"),
    bigquery.SchemaField("away_starter_whip",         "FLOAT"),
    bigquery.SchemaField("matchup_advantage_home",    "FLOAT"),
    bigquery.SchemaField("lineup_confirmed",          "BOOLEAN"),
    bigquery.SchemaField("computed_at",               "TIMESTAMP"),
] + V6_EXTRA_SCHEMA


class V6MatchupBuilder:
    """Compute V6 matchup features (V5 + pitcher arsenal + venue history)."""

    def __init__(self, dry_run: bool = False):
        self.bq = bigquery.Client(project=PROJECT)
        self.dry_run = dry_run
        self._ensure_table()

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------
    def _ensure_table(self) -> None:
        table_ref = bigquery.Table(MATCHUP_V6_TABLE, schema=V6_BASE_SCHEMA_FIELDS)
        table_ref.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="game_date"
        )
        table_ref.clustering_fields = ["game_pk"]
        try:
            self.bq.get_table(MATCHUP_V6_TABLE)
        except Exception:
            if not self.dry_run:
                self.bq.create_table(table_ref, exists_ok=True)
                logger.info("Created V6 matchup table %s", MATCHUP_V6_TABLE)

    # ------------------------------------------------------------------
    # Load V5 matchup features (base)
    # ------------------------------------------------------------------
    def load_v5_matchup(self, game_pks: list[int]) -> pd.DataFrame:
        pk_list = ", ".join(str(pk) for pk in game_pks)
        sql = f"""
            SELECT *
            FROM `{MATCHUP_V5_TABLE}`
            WHERE game_pk IN ({pk_list})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) = 1
        """
        try:
            return self.bq.query(sql).to_dataframe()
        except Exception as e:
            logger.warning("Could not load V5 matchup features: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Get venue_id for games
    # ------------------------------------------------------------------
    def get_venue_ids(self, target_date: date, game_pks: list[int]) -> dict[int, tuple[int, str]]:
        """Return {game_pk: (venue_id, venue_name)} from MLB API."""
        headers = {"User-Agent": "HanksTank/2.0"}
        resp = requests.get(
            f"{MLB_API}/schedule",
            params={"date": target_date.isoformat(), "sportId": 1,
                    "hydrate": "venue", "gameType": "R,F,D,L,W"},
            headers=headers, timeout=30, verify=False,
        )
        resp.raise_for_status()
        data = resp.json()
        venue_map: dict[int, tuple[int, str]] = {}
        for day in data.get("dates", []):
            for g in day.get("games", []):
                pk = g["gamePk"]
                if game_pks and pk not in game_pks:
                    continue
                v = g.get("venue", {})
                venue_map[pk] = (v.get("id", 0), v.get("name", ""))
        return venue_map

    # ------------------------------------------------------------------
    # Load pitcher arsenal features (rolling 30-day lookback)
    # ------------------------------------------------------------------
    def load_pitcher_arsenal(
        self, pitcher_ids: list[int], as_of_date: date
    ) -> pd.DataFrame:
        """
        For each pitcher, compute rolling 30-day weighted average arsenal stats
        from their recent starts (up to as_of_date).
        Uses pitcher_game_stats (must be pre-built via build_pitcher_game_stats.py).
        Falls back gracefully if table doesn't exist.
        """
        if not pitcher_ids:
            return pd.DataFrame()

        try:
            self.bq.get_table(PITCHER_STATS_TABLE)
        except Exception:
            logger.warning("pitcher_game_stats table not found — skipping pitcher arsenal features")
            return pd.DataFrame()

        pitcher_list = ", ".join(str(p) for p in pitcher_ids)
        cutoff = (as_of_date - pd.Timedelta(days=PITCHER_LOOKBACK_DAYS)).isoformat()

        sql = f"""
            WITH recent_starts AS (
                SELECT *
                FROM `{PITCHER_STATS_TABLE}`
                WHERE pitcher IN ({pitcher_list})
                  AND game_date < '{as_of_date.isoformat()}'
                  AND game_date >= '{cutoff}'
                  AND total_pitches >= 25
            ),
            season_starts AS (
                -- Broader window for velocity trend baseline
                SELECT *
                FROM `{PITCHER_STATS_TABLE}`
                WHERE pitcher IN ({pitcher_list})
                  AND game_date < '{as_of_date.isoformat()}'
                  AND EXTRACT(YEAR FROM game_date) >= EXTRACT(YEAR FROM DATE '{as_of_date.isoformat()}') - 1
                  AND total_pitches >= 25
            ),
            -- Recent 2 starts for velocity trend
            recent_ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY pitcher ORDER BY game_date DESC) AS rn
                FROM recent_starts
            ),
            velo_trend AS (
                SELECT
                    pitcher,
                    AVG(CASE WHEN rn <= 2 THEN mean_fastball_velo END) AS velo_recent,
                    AVG(CASE WHEN rn > 2 THEN mean_fastball_velo END)  AS velo_baseline
                FROM recent_ranked
                GROUP BY pitcher
            ),
            -- 30-day rolling aggregate
            rolling AS (
                SELECT
                    pitcher,
                    ANY_VALUE(p_throws)                            AS p_throws,
                    ROUND(AVG(fastball_pct), 4)                    AS fastball_pct,
                    ROUND(AVG(breaking_pct), 4)                    AS breaking_pct,
                    ROUND(AVG(offspeed_pct), 4)                    AS offspeed_pct,
                    ROUND(AVG(cutter_pct),   4)                    AS cutter_pct,
                    ROUND(AVG(mean_release_speed), 2)              AS mean_release_speed,
                    ROUND(AVG(mean_fastball_velo), 2)             AS mean_fastball_velo,
                    ROUND(AVG(mean_spin_rate), 1)                  AS mean_spin_rate,
                    ROUND(AVG(k_bb_pct), 4)                        AS k_bb_pct,
                    ROUND(AVG(xwoba_allowed), 4)                  AS xwoba_allowed,
                    ROUND(AVG(k_pct), 4)                           AS k_pct,
                    ROUND(AVG(bb_pct), 4)                          AS bb_pct,
                    COUNT(1)                                        AS starts_in_window
                FROM recent_starts
                GROUP BY pitcher
            )
            SELECT
                r.*,
                ROUND(v.velo_recent - v.velo_baseline, 2) AS velo_trend
            FROM rolling r
            LEFT JOIN velo_trend v USING (pitcher)
        """
        try:
            df = self.bq.query(sql).to_dataframe()
            logger.info("Loaded arsenal features for %d pitchers", len(df))
            return df
        except Exception as e:
            logger.warning("Pitcher arsenal query failed: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Load venue history for lineups
    # ------------------------------------------------------------------
    def load_venue_history(
        self, player_ids: list[int], venue_id: int
    ) -> pd.DataFrame:
        """Load each player's wOBA at the given venue from player_venue_splits."""
        if not player_ids or not venue_id:
            return pd.DataFrame()

        try:
            self.bq.get_table(VENUE_SPLITS_TABLE)
            self.bq.get_table(SEASON_SPLITS_TABLE)
        except Exception:
            logger.warning("Venue splits tables not found — skipping venue features")
            return pd.DataFrame()

        player_list = ", ".join(str(p) for p in player_ids)
        sql = f"""
            SELECT
                pvs.player_id,
                pvs.woba          AS venue_woba,
                pvs.xwoba         AS venue_xwoba,
                pvs.pa_total      AS venue_pa,
                pss.woba          AS season_woba,
                pss.xwoba         AS season_xwoba
            FROM `{VENUE_SPLITS_TABLE}` pvs
            LEFT JOIN (
                SELECT player_id,
                    AVG(woba)  AS woba,
                    AVG(xwoba) AS xwoba
                FROM `{SEASON_SPLITS_TABLE}`
                WHERE player_id IN ({player_list})
                  AND game_year >= 2021
                GROUP BY player_id
            ) pss USING (player_id)
            WHERE pvs.player_id IN ({player_list})
              AND pvs.venue_id = {venue_id}
              AND pvs.pa_total >= {MIN_PA_VENUE}
        """
        try:
            df = self.bq.query(sql).to_dataframe()
            logger.info("Loaded venue history for %d players at venue %d", len(df), venue_id)
            return df
        except Exception as e:
            logger.warning("Venue history query failed: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Arsenal feature extraction helpers
    # ------------------------------------------------------------------
    def _get_arsenal_features(
        self, arsenal_df: pd.DataFrame, pitcher_id: int | None
    ) -> dict:
        """Extract arsenal features for one pitcher. Returns neutral values if missing."""
        neutral = {
            "fastball_pct": 0.35,
            "breaking_pct": 0.30,
            "offspeed_pct": 0.20,
            "mean_fastball_velo": LEAGUE_AVG_VELO,
            "velo_norm": 0.0,
            "velo_trend": 0.0,
            "k_bb_pct": 0.10,
            "xwoba_allowed": 0.320,
            "starts_in_window": 0,
        }
        if pitcher_id is None or arsenal_df.empty:
            return neutral

        row = arsenal_df[arsenal_df["pitcher"] == pitcher_id]
        if row.empty:
            return neutral

        r = row.iloc[0]
        velo = float(r["mean_fastball_velo"]) if pd.notna(r.get("mean_fastball_velo")) else LEAGUE_AVG_VELO
        return {
            "fastball_pct":   float(r["fastball_pct"])  if pd.notna(r.get("fastball_pct"))  else neutral["fastball_pct"],
            "breaking_pct":   float(r["breaking_pct"])  if pd.notna(r.get("breaking_pct"))  else neutral["breaking_pct"],
            "offspeed_pct":   float(r["offspeed_pct"])  if pd.notna(r.get("offspeed_pct"))  else neutral["offspeed_pct"],
            "mean_fastball_velo": velo,
            "velo_norm":      (velo - LEAGUE_AVG_VELO) / VELO_STD,
            "velo_trend":     float(r["velo_trend"])    if pd.notna(r.get("velo_trend"))     else 0.0,
            "k_bb_pct":       float(r["k_bb_pct"])      if pd.notna(r.get("k_bb_pct"))       else neutral["k_bb_pct"],
            "xwoba_allowed":  float(r["xwoba_allowed"]) if pd.notna(r.get("xwoba_allowed"))  else neutral["xwoba_allowed"],
            "starts_in_window": int(r.get("starts_in_window", 0)),
        }

    # ------------------------------------------------------------------
    # Venue feature extraction helpers
    # ------------------------------------------------------------------
    def _lineup_venue_woba(
        self, venue_df: pd.DataFrame, overall_df: pd.DataFrame,
        lineup_player_ids: list[int]
    ) -> tuple[float | None, float | None]:
        """
        Calculate average lineup wOBA at venue and venue_advantage
        for a set of player_ids. Only uses players with enough PA.
        Returns (venue_woba, venue_advantage) where advantage = venue_woba - season_woba.
        """
        if venue_df.empty or not lineup_player_ids:
            return None, None

        relevant = venue_df[venue_df["player_id"].isin(lineup_player_ids)]
        if relevant.empty:
            return None, None

        # Weighted average by PA (more PA = more reliable)
        woba_vals = relevant["venue_woba"].dropna()
        pa_vals = relevant.loc[woba_vals.index, "venue_pa"]
        if woba_vals.empty:
            return None, None

        lineup_venue_woba = float(np.average(woba_vals, weights=pa_vals))

        # Venue advantage: compare to season average
        has_season = relevant[relevant["season_woba"].notna()]
        if not has_season.empty:
            season_vals = has_season["season_woba"]
            pa_s = has_season["venue_pa"]
            season_woba = float(np.average(season_vals, weights=pa_s))
            venue_advantage = lineup_venue_woba - season_woba
        else:
            venue_advantage = None

        return round(lineup_venue_woba, 4), (round(venue_advantage, 4) if venue_advantage is not None else None)

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------
    def run_for_date(self, target_date: date) -> dict:
        games = self._fetch_game_pks(target_date)
        if not games:
            return {"games": 0}
        return self._run(games, target_date)

    def run_for_game_pks(self, game_pks: list[int], game_date: date) -> dict:
        games = self._fetch_game_pks(game_date, game_pks)
        return self._run(games, game_date)

    def _fetch_game_pks(
        self, target_date: date, game_pks: list[int] | None = None
    ) -> list[int]:
        """Return game PKs from MLB API for the date."""
        resp = requests.get(
            f"{MLB_API}/schedule",
            params={"date": target_date.isoformat(), "sportId": 1, "gameType": "R,F,D,L,W"},
            headers={"User-Agent": "HanksTank/2.0"}, timeout=30, verify=False,
        )
        resp.raise_for_status()
        pks = []
        for day in resp.json().get("dates", []):
            for g in day.get("games", []):
                pk = g["gamePk"]
                if not game_pks or pk in game_pks:
                    state = g.get("status", {}).get("abstractGameState", "")
                    if state != "Final":
                        pks.append(pk)
        return pks

    def _run(self, game_pks: list[int], target_date: date) -> dict:
        # 1. Load V5 matchup features as base
        v5_df = self.load_v5_matchup(game_pks)
        if v5_df.empty:
            logger.warning("No V5 matchup features found for %d games — V6 will have nulls for V5 fields",
                           len(game_pks))

        # 2. Get venue IDs
        venue_map = {}
        try:
            venue_map = self.get_venue_ids(target_date, game_pks)
        except Exception as e:
            logger.warning("Could not fetch venue IDs: %s", e)

        # 3. Get lineup player IDs and starter pitcher IDs from lineups table
        lineups_df = self._load_lineups(game_pks)

        # 3b. Fallback: load starter IDs from game_predictions for games without lineups
        predictions_starters = self._load_prediction_starters(game_pks)

        # 4. Collect all pitcher IDs for arsenal lookup (lineups + prediction starters)
        pitcher_ids = []
        if not lineups_df.empty:
            pitchers = lineups_df[lineups_df["is_probable_pitcher"] == True]["player_id"].dropna().astype(int).tolist()
            pitcher_ids.extend(pitchers)
        # Add IDs from game_predictions for games not yet in lineups
        for pk, (h_id, a_id) in predictions_starters.items():
            if h_id:
                pitcher_ids.append(h_id)
            if a_id:
                pitcher_ids.append(a_id)
        pitcher_ids = list(set(pitcher_ids))

        arsenal_df = self.load_pitcher_arsenal(pitcher_ids, target_date)

        rows = []
        for game_pk in game_pks:
            venue_id, venue_name = venue_map.get(game_pk, (None, None))

            # Get V5 base row
            v5_row = None
            if not v5_df.empty and "game_pk" in v5_df.columns:
                subset = v5_df[v5_df["game_pk"] == game_pk]
                if not subset.empty:
                    v5_row = subset.iloc[0]

            # Get lineup player IDs split by team
            home_batters, away_batters = [], []
            home_starter_id, away_starter_id = None, None
            if not lineups_df.empty:
                game_lineups = lineups_df[lineups_df["game_pk"] == game_pk]
                home_rows = game_lineups[game_lineups["team_type"] == "home"]
                away_rows = game_lineups[game_lineups["team_type"] == "away"]
                home_batters = home_rows[home_rows["batting_order"].notna()]["player_id"].dropna().astype(int).tolist()
                away_batters = away_rows[away_rows["batting_order"].notna()]["player_id"].dropna().astype(int).tolist()
                h_pit = home_rows[home_rows["is_probable_pitcher"] == True]["player_id"].dropna()
                a_pit = away_rows[away_rows["is_probable_pitcher"] == True]["player_id"].dropna()
                home_starter_id = int(h_pit.iloc[0]) if not h_pit.empty else None
                away_starter_id = int(a_pit.iloc[0]) if not a_pit.empty else None

            # Fallback to game_predictions probable starters when lineups aren't posted yet
            if home_starter_id is None or away_starter_id is None:
                pred_h, pred_a = predictions_starters.get(game_pk, (None, None))
                if home_starter_id is None:
                    home_starter_id = pred_h
                if away_starter_id is None:
                    away_starter_id = pred_a

            # 5. Pitcher arsenal features
            home_arsenal = self._get_arsenal_features(arsenal_df, home_starter_id)
            away_arsenal = self._get_arsenal_features(arsenal_df, away_starter_id)

            # Arsenal quality composite: (higher xwoba_allowed = worse, lower k_bb = worse)
            # Positive starter_arsenal_advantage = home pitcher has better stuff
            def _arsenal_quality(a: dict) -> float:
                return (a["k_bb_pct"] * 2.0) - a["xwoba_allowed"] + (a["velo_norm"] * 0.05)

            home_q = _arsenal_quality(home_arsenal)
            away_q = _arsenal_quality(away_arsenal)
            arsenal_advantage = round(home_q - away_q, 4)

            # 6. Venue history features
            home_venue_woba, home_venue_adv = None, None
            away_venue_woba, away_venue_disadv = None, None
            venue_woba_diff = None

            all_players = list(set(home_batters + away_batters))
            if venue_id and all_players:
                venue_df = self.load_venue_history(all_players, venue_id)
                if not venue_df.empty:
                    home_venue_woba, home_venue_adv = self._lineup_venue_woba(venue_df, venue_df, home_batters)
                    away_venue_woba, away_venue_disadv = self._lineup_venue_woba(venue_df, venue_df, away_batters)
                    if home_venue_woba is not None and away_venue_woba is not None:
                        venue_woba_diff = round(home_venue_woba - away_venue_woba, 4)

            # 7. Build output row
            row = {"game_pk": game_pk, "game_date": target_date.isoformat()}

            # Copy all V5 fields
            if v5_row is not None:
                for col in v5_row.index:
                    val = v5_row[col]
                    row[col] = None if (isinstance(val, float) and np.isnan(val)) else val
            else:
                # V5 base fields will be null — flag for debugging
                row["lineup_confirmed"] = False
                row["matchup_advantage_home"] = 0.0

            # Venue fields
            row["venue_id"] = venue_id
            row["venue_name"] = venue_name

            # V6 pitcher arsenal
            row["home_starter_fastball_pct"]  = round(home_arsenal["fastball_pct"],  4)
            row["away_starter_fastball_pct"]  = round(away_arsenal["fastball_pct"],  4)
            row["home_starter_breaking_pct"]  = round(home_arsenal["breaking_pct"],  4)
            row["away_starter_breaking_pct"]  = round(away_arsenal["breaking_pct"],  4)
            row["home_starter_offspeed_pct"]  = round(home_arsenal["offspeed_pct"],  4)
            row["away_starter_offspeed_pct"]  = round(away_arsenal["offspeed_pct"],  4)
            row["home_starter_mean_velo"]     = round(home_arsenal["mean_fastball_velo"], 2)
            row["away_starter_mean_velo"]     = round(away_arsenal["mean_fastball_velo"], 2)
            row["home_starter_velo_norm"]     = round(home_arsenal["velo_norm"],      4)
            row["away_starter_velo_norm"]     = round(away_arsenal["velo_norm"],      4)
            row["home_starter_velo_trend"]    = round(home_arsenal["velo_trend"],     4)
            row["away_starter_velo_trend"]    = round(away_arsenal["velo_trend"],     4)
            row["home_starter_k_bb_pct"]      = round(home_arsenal["k_bb_pct"],       4)
            row["away_starter_k_bb_pct"]      = round(away_arsenal["k_bb_pct"],       4)
            row["home_starter_xwoba_allowed"] = round(home_arsenal["xwoba_allowed"],  4)
            row["away_starter_xwoba_allowed"] = round(away_arsenal["xwoba_allowed"],  4)
            row["starter_arsenal_advantage"]  = arsenal_advantage

            # V6 venue history
            row["home_lineup_venue_woba"]     = home_venue_woba
            row["away_lineup_venue_woba"]     = away_venue_woba
            row["venue_woba_differential"]    = venue_woba_diff
            row["home_venue_advantage"]       = home_venue_adv
            row["away_venue_disadvantage"]    = away_venue_disadv

            # Ensure computed_at is set
            row["computed_at"] = datetime.now(tz=timezone.utc).isoformat()

            rows.append(row)
            logger.info(
                "V6 [%d] arsenal home=%.3f/%.2fmph away=%.3f/%.2fmph | venue diff=%s",
                game_pk,
                home_arsenal["k_bb_pct"], home_arsenal["mean_fastball_velo"],
                away_arsenal["k_bb_pct"], away_arsenal["mean_fastball_velo"],
                f"{venue_woba_diff:+.3f}" if venue_woba_diff is not None else "N/A",
            )

        if rows and not self.dry_run:
            # Sanitize numpy scalar types → Python native (BQ JSON serialization requires this)
            def _sanitize(v):
                from datetime import date as _date, datetime as _datetime
                if isinstance(v, (_date, _datetime)):
                    return v.isoformat()
                if isinstance(v, (np.integer,)):
                    return int(v)
                if isinstance(v, (np.floating,)):
                    return None if np.isnan(v) else float(v)
                if isinstance(v, (np.bool_,)):
                    return bool(v)
                return v

            rows = [{k: _sanitize(v) for k, v in r.items()} for r in rows]

            pk_list = ", ".join(str(r["game_pk"]) for r in rows)
            try:
                self.bq.query(
                    f"DELETE FROM `{MATCHUP_V6_TABLE}` "
                    f"WHERE game_pk IN ({pk_list}) "
                    f"AND game_date = '{target_date.isoformat()}'"
                ).result()
            except Exception as del_err:
                logger.warning("V6 delete (streaming buffer): %s", del_err)

            errors = self.bq.insert_rows_json(MATCHUP_V6_TABLE, rows)
            if errors:
                logger.error("V6 BQ insert errors: %s", errors)
            else:
                logger.info("Wrote %d V6 matchup rows to BigQuery", len(rows))
        elif rows and self.dry_run:
            logger.info("[DRY RUN] Would write %d V6 matchup rows", len(rows))

        return {"games": len(rows), "date": target_date.isoformat()}

    def _load_lineups(self, game_pks: list[int]) -> pd.DataFrame:
        if not game_pks:
            return pd.DataFrame()
        pk_list = ", ".join(str(pk) for pk in game_pks)
        sql = f"""
            SELECT game_pk, team_type, player_id, player_name,
                   batting_order, bat_side, pitch_hand,
                   is_starter, is_probable_pitcher, lineup_confirmed
            FROM `{LINEUPS_TABLE}`
            WHERE game_pk IN ({pk_list}) AND is_starter = TRUE
            ORDER BY game_pk, team_type, batting_order NULLS LAST
        """
        try:
            return self.bq.query(sql).to_dataframe()
        except Exception as e:
            logger.warning("Could not load lineups: %s", e)
            return pd.DataFrame()

    def _load_prediction_starters(self, game_pks: list[int]) -> dict[int, tuple]:
        """Fallback: load probable starter IDs from game_predictions when lineups aren't posted yet.
        Returns {game_pk: (home_starter_id, away_starter_id)}.
        """
        if not game_pks:
            return {}
        pk_list = ", ".join(str(pk) for pk in game_pks)
        sql = f"""
            SELECT game_pk, home_starter_id, away_starter_id
            FROM `{PROJECT}.mlb_2026_season.game_predictions`
            WHERE game_pk IN ({pk_list})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY predicted_at DESC) = 1
        """
        try:
            result = {}
            for row in self.bq.query(sql).result():
                h = int(row["home_starter_id"]) if row["home_starter_id"] else None
                a = int(row["away_starter_id"]) if row["away_starter_id"] else None
                result[row["game_pk"]] = (h, a)
            return result
        except Exception as e:
            logger.warning("Could not load prediction starters: %s", e)
            return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build V6 matchup features")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--game-pk", help="Comma-separated game PKs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    builder = V6MatchupBuilder(dry_run=args.dry_run)
    target = date.fromisoformat(args.date) if args.date else date.today()

    if args.game_pk:
        pks = [int(pk.strip()) for pk in args.game_pk.split(",")]
        result = builder.run_for_game_pks(pks, target)
    else:
        result = builder.run_for_date(target)

    logger.info("V6 matchup build complete: %s", result)


if __name__ == "__main__":
    main()
