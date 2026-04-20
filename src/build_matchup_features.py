#!/usr/bin/env python3
"""
Matchup Feature Builder

Computes pitcher/batter matchup features for each scheduled game using:
  1. Confirmed lineups from mlb_2026_season.lineups
  2. Historical Statcast data (2015–2025) for H2H and platoon splits
  3. 2026 Statcast data for in-season recency

Features computed per game:

  Team-level aggregates (home & away):
    - lineup_woba_vs_hand       avg wOBA vs pitcher handedness (L/R)
    - lineup_k_pct_vs_hand      avg K% vs pitcher handedness
    - lineup_bb_pct_vs_hand     avg BB% vs pitcher handedness
    - lineup_iso_vs_hand        avg ISO vs pitcher handedness

  Starter pitcher features:
    - starter_woba_allowed_vs_hand  wOBA allowed vs batter handedness split
    - starter_k_pct_vs_hand         K% vs batter handedness
    - starter_whip_season           current season WHIP proxy

  Head-to-head (lineup vs specific starter):
    - h2h_pa_total              total PA in H2H history (lineup aggregate)
    - h2h_woba                  wOBA in those PAs
    - h2h_k_pct                 K% in H2H
    - h2h_hr_rate               HR rate in H2H

  Batting order effects:
    - top3_woba_vs_hand         positions 1-3 aggregate wOBA vs hand
    - middle4_woba_vs_hand      positions 4-7
    - bottom2_woba_vs_hand      positions 8-9

  Handedness matchup summary:
    - pct_lineup_same_hand      % of batters facing same-hand pitcher (harder matchup)
    - matchup_advantage_home    composite matchup score differential (home minus away)

Output table: mlb_2026_season.matchup_features

Usage:
    python build_matchup_features.py                    # today's games
    python build_matchup_features.py --date 2026-04-02
    python build_matchup_features.py --game-pk 745612,745613
    python build_matchup_features.py --dry-run
"""

import argparse
import logging
import warnings
from datetime import date, datetime, timezone, timedelta
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
HIST_DATASET = "mlb_historical_data"
SEASON_DATASET = "mlb_2026_season"
MLB_API = "https://statsapi.mlb.com/api/v1"

MATCHUP_TABLE = f"{PROJECT}.{SEASON_DATASET}.matchup_features"
LINEUPS_TABLE = f"{PROJECT}.{SEASON_DATASET}.lineups"
HIST_STATCAST = f"{PROJECT}.{HIST_DATASET}.statcast_pitches"
SEASON_STATCAST = f"{PROJECT}.{SEASON_DATASET}.statcast_pitches"

# Minimum PA threshold for reliable split statistics
MIN_PA_SPLITS = 20
MIN_PA_H2H = 5
PROJECTED_LINEUP_LOOKBACK_DAYS = 14

MATCHUP_SCHEMA = [
    bigquery.SchemaField("game_pk", "INTEGER"),
    bigquery.SchemaField("game_date", "DATE"),
    bigquery.SchemaField("home_team_id", "INTEGER"),
    bigquery.SchemaField("away_team_id", "INTEGER"),
    # Home batting lineup vs away starter
    bigquery.SchemaField("home_lineup_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_lineup_k_pct_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_lineup_bb_pct_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_lineup_iso_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_top3_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_middle4_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_bottom2_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_pct_same_hand", "FLOAT"),
    bigquery.SchemaField("home_h2h_woba", "FLOAT"),
    bigquery.SchemaField("home_h2h_pa_total", "INTEGER"),
    bigquery.SchemaField("home_h2h_k_pct", "FLOAT"),
    bigquery.SchemaField("home_h2h_hr_rate", "FLOAT"),
    # Away batting lineup vs home starter
    bigquery.SchemaField("away_lineup_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_lineup_k_pct_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_lineup_bb_pct_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_lineup_iso_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_top3_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_middle4_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_bottom2_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_pct_same_hand", "FLOAT"),
    bigquery.SchemaField("away_h2h_woba", "FLOAT"),
    bigquery.SchemaField("away_h2h_pa_total", "INTEGER"),
    bigquery.SchemaField("away_h2h_k_pct", "FLOAT"),
    bigquery.SchemaField("away_h2h_hr_rate", "FLOAT"),
    # Starter pitcher splits
    bigquery.SchemaField("home_starter_id", "INTEGER"),
    bigquery.SchemaField("home_starter_hand", "STRING"),
    bigquery.SchemaField("home_starter_woba_allowed", "FLOAT"),
    bigquery.SchemaField("home_starter_k_pct", "FLOAT"),
    bigquery.SchemaField("home_starter_whip", "FLOAT"),
    bigquery.SchemaField("away_starter_id", "INTEGER"),
    bigquery.SchemaField("away_starter_hand", "STRING"),
    bigquery.SchemaField("away_starter_woba_allowed", "FLOAT"),
    bigquery.SchemaField("away_starter_k_pct", "FLOAT"),
    bigquery.SchemaField("away_starter_whip", "FLOAT"),
    # Composite differential (positive = home advantage)
    bigquery.SchemaField("matchup_advantage_home", "FLOAT"),
    bigquery.SchemaField("lineup_confirmed", "BOOLEAN"),
    bigquery.SchemaField("computed_at", "TIMESTAMP"),
]


class MatchupFeatureBuilder:
    def __init__(self, dry_run: bool = False):
        self.bq = bigquery.Client(project=PROJECT)
        self.dry_run = dry_run
        self._ensure_table()

    # -----------------------------------------------------------------------
    # Table setup
    # -----------------------------------------------------------------------
    def _ensure_table(self) -> None:
        table_ref = bigquery.Table(MATCHUP_TABLE, schema=MATCHUP_SCHEMA)
        table_ref.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="game_date",
        )
        table_ref.clustering_fields = ["game_pk"]
        try:
            self.bq.get_table(MATCHUP_TABLE)
        except Exception:
            if not self.dry_run:
                self.bq.create_table(table_ref, exists_ok=True)
                logger.info("Created table %s", MATCHUP_TABLE)

    # -----------------------------------------------------------------------
    # Load lineups
    # -----------------------------------------------------------------------
    def load_lineups(self, game_pks: list[int]) -> pd.DataFrame:
        pk_list = ", ".join(str(pk) for pk in game_pks)
        sql = f"""
            SELECT
                game_pk, game_date, team_id, team_type,
                player_id, player_name, batting_order, position,
                bat_side, pitch_hand, is_starter, is_probable_pitcher,
                lineup_confirmed
            FROM `{LINEUPS_TABLE}`
            WHERE game_pk IN ({pk_list})
              AND is_starter = TRUE
            ORDER BY game_pk, team_type, batting_order NULLS LAST
        """
        return self.bq.query(sql).to_dataframe()

    def load_recent_confirmed_lineups(
        self, team_ids: list[int], target_date: date
    ) -> pd.DataFrame:
        if not team_ids:
            return pd.DataFrame()

        start_date = target_date - timedelta(days=PROJECTED_LINEUP_LOOKBACK_DAYS)
        team_list = ", ".join(str(team_id) for team_id in sorted(set(team_ids)))
        sql = f"""
            SELECT
                game_pk, game_date, team_id, team_type,
                player_id, player_name, batting_order, position,
                bat_side, pitch_hand, is_starter, is_probable_pitcher,
                lineup_confirmed
            FROM `{LINEUPS_TABLE}`
            WHERE team_id IN ({team_list})
              AND game_date >= DATE('{start_date.isoformat()}')
              AND game_date < DATE('{target_date.isoformat()}')
              AND lineup_confirmed = TRUE
            ORDER BY game_date DESC, team_id, batting_order NULLS LAST
        """
        return self.bq.query(sql).to_dataframe()

    # -----------------------------------------------------------------------
    # Load statcast splits — H2H and platoon
    # -----------------------------------------------------------------------
    def load_statcast_splits(
        self, pitcher_ids: list[int], batter_ids: list[int]
    ) -> pd.DataFrame:
        """
        Load statcast pitch-by-pitch data filtered to relevant pitcher/batter combos.
        Combines historical (2015–2025) and 2026 data.
        Uses wOBA weights per event type (approximation).
        """
        if not pitcher_ids or not batter_ids:
            return pd.DataFrame()

        pitcher_list = ", ".join(str(p) for p in pitcher_ids)
        batter_list = ", ".join(str(b) for b in batter_ids)

        # wOBA weights (2020 approximate constants)
        woba_sql = """
            WITH pitches AS (
                SELECT
                    pitcher, batter, p_throws, stand,
                    events, game_date,
                    CASE events
                        WHEN 'walk' THEN 0.69
                        WHEN 'hit_by_pitch' THEN 0.72
                        WHEN 'single' THEN 0.88
                        WHEN 'double' THEN 1.247
                        WHEN 'triple' THEN 1.578
                        WHEN 'home_run' THEN 2.031
                        ELSE 0.0
                    END AS woba_value,
                    CASE WHEN events IN (
                        'strikeout','strikeout_double_play',
                        'field_out','force_out','grounded_into_double_play',
                        'double_play','triple_play',
                        'field_error','fielders_choice','fielders_choice_out',
                        'walk','hit_by_pitch','single','double','triple','home_run',
                        'sac_fly','sac_bunt'
                    ) THEN 1 ELSE 0 END AS is_pa,
                    CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END AS is_k,
                    CASE WHEN events = 'walk' THEN 1 ELSE 0 END AS is_bb,
                    CASE WHEN events = 'home_run' THEN 1 ELSE 0 END AS is_hr,
                    CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END AS is_hit,
                    CASE WHEN events IN ('double','triple','home_run') THEN 1 ELSE 0 END AS is_xbh
                FROM `{hist_table}`
                WHERE pitcher IN ({pitchers})
                  AND batter IN ({batters})
                  AND events IS NOT NULL

                UNION ALL

                SELECT
                    pitcher, batter, p_throws, stand,
                    events, game_date,
                    CASE events
                        WHEN 'walk' THEN 0.69
                        WHEN 'hit_by_pitch' THEN 0.72
                        WHEN 'single' THEN 0.88
                        WHEN 'double' THEN 1.247
                        WHEN 'triple' THEN 1.578
                        WHEN 'home_run' THEN 2.031
                        ELSE 0.0
                    END AS woba_value,
                    CASE WHEN events IN (
                        'strikeout','strikeout_double_play',
                        'field_out','force_out','grounded_into_double_play',
                        'double_play','triple_play',
                        'field_error','fielders_choice','fielders_choice_out',
                        'walk','hit_by_pitch','single','double','triple','home_run',
                        'sac_fly','sac_bunt'
                    ) THEN 1 ELSE 0 END AS is_pa,
                    CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END AS is_k,
                    CASE WHEN events = 'walk' THEN 1 ELSE 0 END AS is_bb,
                    CASE WHEN events = 'home_run' THEN 1 ELSE 0 END AS is_hr,
                    CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END AS is_hit,
                    CASE WHEN events IN ('double','triple','home_run') THEN 1 ELSE 0 END AS is_xbh
                FROM `{season_table}`
                WHERE pitcher IN ({pitchers})
                  AND batter IN ({batters})
                  AND events IS NOT NULL
            )
            SELECT
                pitcher, batter, p_throws, stand,
                SUM(is_pa) AS pa,
                SUM(is_k) AS k,
                SUM(is_bb) AS bb,
                SUM(is_hr) AS hr,
                SUM(is_hit) AS hits,
                SUM(woba_value) AS woba_numerator
            FROM pitches
            WHERE is_pa = 1
            GROUP BY pitcher, batter, p_throws, stand
        """.format(
            hist_table=HIST_STATCAST,
            season_table=SEASON_STATCAST,
            pitchers=pitcher_list,
            batters=batter_list,
        )

        try:
            df = self.bq.query(woba_sql).to_dataframe()
            logger.info("Loaded %d statcast split rows", len(df))
            return df
        except Exception as e:
            logger.error("Statcast split query failed: %s", e)
            return pd.DataFrame()

    def load_batter_career_splits(self, batter_ids: list[int]) -> pd.DataFrame:
        """
        Load each batter's career platoon splits (vs LHP/RHP) across ALL pitchers.
        Used for lineup_woba_vs_hand — NOT filtered to specific pitchers (unlike H2H).
        Combines 2015-2025 historical + 2026 data.
        """
        if not batter_ids:
            return pd.DataFrame()

        batter_list = ", ".join(str(b) for b in batter_ids)
        sql = f"""
            WITH pitches AS (
                SELECT
                    batter, p_throws,
                    CASE events
                        WHEN 'walk' THEN 0.69 WHEN 'hit_by_pitch' THEN 0.72
                        WHEN 'single' THEN 0.88 WHEN 'double' THEN 1.247
                        WHEN 'triple' THEN 1.578 WHEN 'home_run' THEN 2.031
                        ELSE 0.0
                    END AS woba_value,
                    CASE WHEN events IN (
                        'strikeout','strikeout_double_play','field_out','force_out',
                        'grounded_into_double_play','double_play','triple_play',
                        'field_error','fielders_choice','fielders_choice_out',
                        'walk','hit_by_pitch','single','double','triple','home_run',
                        'sac_fly','sac_bunt'
                    ) THEN 1 ELSE 0 END AS is_pa,
                    CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END AS is_k,
                    CASE WHEN events = 'walk' THEN 1 ELSE 0 END AS is_bb,
                    CASE WHEN events = 'home_run' THEN 1 ELSE 0 END AS is_hr,
                    CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END AS is_hit
                FROM `{HIST_STATCAST}`
                WHERE batter IN ({batter_list})
                  AND events IS NOT NULL

                UNION ALL

                SELECT
                    batter, p_throws,
                    CASE events
                        WHEN 'walk' THEN 0.69 WHEN 'hit_by_pitch' THEN 0.72
                        WHEN 'single' THEN 0.88 WHEN 'double' THEN 1.247
                        WHEN 'triple' THEN 1.578 WHEN 'home_run' THEN 2.031
                        ELSE 0.0
                    END AS woba_value,
                    CASE WHEN events IN (
                        'strikeout','strikeout_double_play','field_out','force_out',
                        'grounded_into_double_play','double_play','triple_play',
                        'field_error','fielders_choice','fielders_choice_out',
                        'walk','hit_by_pitch','single','double','triple','home_run',
                        'sac_fly','sac_bunt'
                    ) THEN 1 ELSE 0 END AS is_pa,
                    CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END AS is_k,
                    CASE WHEN events = 'walk' THEN 1 ELSE 0 END AS is_bb,
                    CASE WHEN events = 'home_run' THEN 1 ELSE 0 END AS is_hr,
                    CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END AS is_hit
                FROM `{SEASON_STATCAST}`
                WHERE batter IN ({batter_list})
                  AND events IS NOT NULL
            )
            SELECT
                batter, p_throws,
                SUM(is_pa) AS pa,
                SUM(is_k) AS k,
                SUM(is_bb) AS bb,
                SUM(is_hr) AS hr,
                SUM(is_hit) AS hits,
                SUM(woba_value) AS woba_numerator
            FROM pitches
            WHERE is_pa = 1
            GROUP BY batter, p_throws
        """
        try:
            df = self.bq.query(sql).to_dataframe()
            logger.info("Loaded %d batter career platoon rows", len(df))
            return df
        except Exception as e:
            logger.error("Batter career split query failed: %s", e)
            return pd.DataFrame()

    def load_pitcher_overall_splits(self, pitcher_ids: list[int]) -> pd.DataFrame:
        """Load pitcher career/recent splits vs LHB and RHB (not batter-specific)."""
        if not pitcher_ids:
            return pd.DataFrame()

        pitcher_list = ", ".join(str(p) for p in pitcher_ids)
        sql = f"""
            WITH pitches AS (
                SELECT pitcher, p_throws, stand, events, game_date,
                    CASE events
                        WHEN 'walk' THEN 0.69 WHEN 'hit_by_pitch' THEN 0.72
                        WHEN 'single' THEN 0.88 WHEN 'double' THEN 1.247
                        WHEN 'triple' THEN 1.578 WHEN 'home_run' THEN 2.031
                        ELSE 0.0
                    END AS woba_value,
                    CASE WHEN events IN (
                        'strikeout','strikeout_double_play','field_out','force_out',
                        'grounded_into_double_play','double_play','field_error',
                        'fielders_choice','fielders_choice_out','walk','hit_by_pitch',
                        'single','double','triple','home_run','sac_fly','sac_bunt'
                    ) THEN 1 ELSE 0 END AS is_pa,
                    CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END AS is_k,
                    CASE WHEN events = 'walk' THEN 1 ELSE 0 END AS is_bb,
                    CASE WHEN events = 'home_run' THEN 1 ELSE 0 END AS is_hr
                FROM `{HIST_STATCAST}`
                WHERE pitcher IN ({pitcher_list})
                  AND events IS NOT NULL
                  AND EXTRACT(YEAR FROM game_date) >= 2022

                UNION ALL

                SELECT pitcher, p_throws, stand, events, game_date,
                    CASE events
                        WHEN 'walk' THEN 0.69 WHEN 'hit_by_pitch' THEN 0.72
                        WHEN 'single' THEN 0.88 WHEN 'double' THEN 1.247
                        WHEN 'triple' THEN 1.578 WHEN 'home_run' THEN 2.031
                        ELSE 0.0
                    END AS woba_value,
                    CASE WHEN events IN (
                        'strikeout','strikeout_double_play','field_out','force_out',
                        'grounded_into_double_play','double_play','field_error',
                        'fielders_choice','fielders_choice_out','walk','hit_by_pitch',
                        'single','double','triple','home_run','sac_fly','sac_bunt'
                    ) THEN 1 ELSE 0 END AS is_pa,
                    CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END AS is_k,
                    CASE WHEN events = 'walk' THEN 1 ELSE 0 END AS is_bb,
                    CASE WHEN events = 'home_run' THEN 1 ELSE 0 END AS is_hr
                FROM `{SEASON_STATCAST}`
                WHERE pitcher IN ({pitcher_list})
                  AND events IS NOT NULL
            )
            SELECT
                pitcher, p_throws, stand,
                SUM(is_pa) AS pa,
                SUM(is_k) AS k,
                SUM(is_bb) AS bb,
                SUM(is_hr) AS hr,
                SUM(woba_value) AS woba_numerator,
                COUNT(DISTINCT DATE_TRUNC(game_date, MONTH)) AS months_pitched
            FROM pitches
            WHERE is_pa = 1
            GROUP BY pitcher, p_throws, stand
        """
        try:
            df = self.bq.query(sql).to_dataframe()
            logger.info("Loaded %d pitcher overall split rows", len(df))
            return df
        except Exception as e:
            logger.error("Pitcher splits query failed: %s", e)
            return pd.DataFrame()

    # -----------------------------------------------------------------------
    # Today's schedule (for game metadata if not passed in)
    # -----------------------------------------------------------------------
    def fetch_game_metadata(self, target_date: date) -> list[dict]:
        url = f"{MLB_API}/schedule"
        params = {
            "date": target_date.isoformat(),
            "sportId": 1,
            "hydrate": "team,probablePitcher",
            "gameType": "R,F,D,L,W",
        }
        resp = requests.get(url, params=params, headers={"User-Agent": "HanksTank/2.0"},
                            timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()
        games = []
        for day in data.get("dates", []):
            for g in day.get("games", []):
                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})
                games.append({
                    "game_pk": g["gamePk"],
                    "game_date": day["date"],
                    "home_team_id": home.get("team", {}).get("id"),
                    "away_team_id": away.get("team", {}).get("id"),
                    "home_probable_pitcher_id": home.get("probablePitcher", {}).get("id"),
                    "home_probable_pitcher": home.get("probablePitcher", {}).get("fullName"),
                    "away_probable_pitcher_id": away.get("probablePitcher", {}).get("id"),
                    "away_probable_pitcher": away.get("probablePitcher", {}).get("fullName"),
                })
        return games

    # -----------------------------------------------------------------------
    # Feature computation helpers
    # -----------------------------------------------------------------------
    def _safe_woba(self, pa: float, woba_num: float) -> Optional[float]:
        if pa >= MIN_PA_SPLITS:
            return round(woba_num / pa, 3)
        return None

    def _compute_pitcher_features(
        self, pitcher_id: int, pitcher_splits_df: pd.DataFrame
    ) -> dict:
        """Compute pitcher-level split features."""
        feats = {
            "starter_id": pitcher_id,
            "starter_hand": None,
            "starter_woba_allowed": None,
            "starter_k_pct": None,
            "starter_whip": None,
        }
        if pitcher_splits_df.empty or pitcher_id is None:
            return feats

        p_data = pitcher_splits_df[pitcher_splits_df["pitcher"] == pitcher_id]
        if p_data.empty:
            return feats

        hand = p_data["p_throws"].iloc[0] if not p_data.empty else None
        feats["starter_hand"] = str(hand) if hand else None

        total_pa = p_data["pa"].sum()
        total_woba_num = p_data["woba_value"].sum() if "woba_value" in p_data.columns else p_data["woba_numerator"].sum()
        total_k = p_data["k"].sum()

        if total_pa >= MIN_PA_SPLITS:
            feats["starter_woba_allowed"] = round(total_woba_num / total_pa, 3)
            feats["starter_k_pct"] = round(total_k / total_pa, 3)

        return feats

    def _compute_lineup_features(
        self,
        lineup: pd.DataFrame,
        pitcher_id: int,
        pitcher_hand: Optional[str],
        statcast_df: pd.DataFrame,
        pitcher_splits_df: pd.DataFrame,
        platoon_df: pd.DataFrame,
    ) -> dict:
        """
        Compute full lineup-level matchup features.
        lineup: DataFrame of batters for one team (batting_order 1–9)
        statcast_df: H2H data filtered to specific pitchers vs these batters
        platoon_df: career batter splits vs all LHP/RHP (not pitcher-filtered)
        """
        # Only batting lineup (exclude pitchers from batting stats if NL-style)
        batters = lineup[
            (lineup["batting_order"].notna()) & (lineup["batting_order"] > 0)
        ].sort_values("batting_order")

        if batters.empty:
            return self._empty_lineup_features()

        batter_ids = batters["player_id"].tolist()

        # --- Platoon splits (batter career wOBA vs LHP/RHP across all pitchers) ---
        # Uses platoon_df (no pitcher filter) so ≥MIN_PA_SPLITS is realistically achievable
        woba_by_batter = {}
        k_pct_by_batter = {}
        bb_pct_by_batter = {}
        iso_by_batter = {}

        for _, row in batters.iterrows():
            bid = row["player_id"]

            if not platoon_df.empty and pitcher_hand:
                # Career performance vs same pitcher handedness across ALL pitchers
                mask = (
                    (platoon_df["batter"] == bid)
                    & (platoon_df["p_throws"] == pitcher_hand)
                )
                sub = platoon_df[mask]
                if sub["pa"].sum() >= MIN_PA_SPLITS:
                    pa = sub["pa"].sum()
                    woba_by_batter[bid] = sub["woba_numerator"].sum() / pa
                    k_pct_by_batter[bid] = sub["k"].sum() / pa
                    bb_pct_by_batter[bid] = sub["bb"].sum() / pa
                    hits = sub["hits"].sum()
                    hr = sub["hr"].sum()
                    slg_approx = (hits + hr) / pa if pa > 0 else 0
                    iso_by_batter[bid] = max(0, slg_approx - (hits / pa if pa > 0 else 0))

        # Aggregate wOBA across batting order positions
        def _position_group_woba(positions: list[int]) -> Optional[float]:
            vals = []
            for _, row in batters[batters["batting_order"].isin(positions)].iterrows():
                bid = row["player_id"]
                if bid in woba_by_batter:
                    vals.append(woba_by_batter[bid])
            return round(np.mean(vals), 3) if vals else None

        top3 = _position_group_woba([1, 2, 3])
        middle4 = _position_group_woba([4, 5, 6, 7])
        bottom2 = _position_group_woba([8, 9])

        lineup_woba = (
            round(np.mean(list(woba_by_batter.values())), 3)
            if woba_by_batter else None
        )
        lineup_k = (
            round(np.mean(list(k_pct_by_batter.values())), 3)
            if k_pct_by_batter else None
        )
        lineup_bb = (
            round(np.mean(list(bb_pct_by_batter.values())), 3)
            if bb_pct_by_batter else None
        )
        lineup_iso = (
            round(np.mean(list(iso_by_batter.values())), 3)
            if iso_by_batter else None
        )

        # --- % of lineup facing same-hand pitcher (harder matchup) ---
        pct_same_hand = None
        if pitcher_hand:
            same_hand_count = 0
            total_count = len(batters)
            for _, row in batters.iterrows():
                bs = str(row.get("bat_side", "R"))
                if bs == pitcher_hand:
                    same_hand_count += 1
                elif bs == "S":
                    # Switch hitter always on opposite side — slight advantage to the batter
                    pass
            if total_count > 0:
                pct_same_hand = round(same_hand_count / total_count, 3)

        # --- H2H (head-to-head) stats for this lineup vs specific pitcher ---
        h2h_pa = 0
        h2h_woba_num = 0.0
        h2h_k = 0
        h2h_hr = 0

        if not statcast_df.empty and pitcher_id:
            for bid in batter_ids:
                mask = (
                    (statcast_df["pitcher"] == pitcher_id)
                    & (statcast_df["batter"] == bid)
                )
                sub = statcast_df[mask]
                h2h_pa += int(sub["pa"].sum())
                h2h_woba_num += float(sub["woba_numerator"].sum())
                h2h_k += int(sub["k"].sum())
                h2h_hr += int(sub["hr"].sum())

        # Bayesian shrinkage: blend H2H wOBA toward league average based on PA count.
        # With H2H_SHRINK_PA=50, at 56 PA the raw value gets ~53% weight (47% prior).
        # This prevents tiny lineup-vs-pitcher samples from dominating the composite.
        LEAGUE_AVG_WOBA = 0.320
        H2H_SHRINK_PA = 50
        if h2h_pa >= MIN_PA_H2H:
            shrink_w = h2h_pa / (h2h_pa + H2H_SHRINK_PA)
            h2h_woba = round(
                (h2h_woba_num / h2h_pa) * shrink_w + LEAGUE_AVG_WOBA * (1 - shrink_w), 3
            )
            h2h_k_pct = round(h2h_k / h2h_pa, 3)
            h2h_hr_rate = round(h2h_hr / h2h_pa, 3)
        else:
            h2h_woba = None
            h2h_k_pct = None
            h2h_hr_rate = None

        return {
            "lineup_woba_vs_hand": lineup_woba,
            "lineup_k_pct_vs_hand": lineup_k,
            "lineup_bb_pct_vs_hand": lineup_bb,
            "lineup_iso_vs_hand": lineup_iso,
            "top3_woba_vs_hand": top3,
            "middle4_woba_vs_hand": middle4,
            "bottom2_woba_vs_hand": bottom2,
            "pct_same_hand": pct_same_hand,
            "h2h_woba": h2h_woba,
            "h2h_pa_total": h2h_pa,
            "h2h_k_pct": h2h_k_pct,
            "h2h_hr_rate": h2h_hr_rate,
        }

    def _empty_lineup_features(self) -> dict:
        return {
            "lineup_woba_vs_hand": None,
            "lineup_k_pct_vs_hand": None,
            "lineup_bb_pct_vs_hand": None,
            "lineup_iso_vs_hand": None,
            "top3_woba_vs_hand": None,
            "middle4_woba_vs_hand": None,
            "bottom2_woba_vs_hand": None,
            "pct_same_hand": None,
            "h2h_woba": None,
            "h2h_pa_total": 0,
            "h2h_k_pct": None,
            "h2h_hr_rate": None,
        }

    def _project_team_lineup_rows(
        self,
        game: dict,
        team_type: str,
        target_date: date,
        current_team_rows: pd.DataFrame,
        recent_team_rows: pd.DataFrame,
    ) -> list[dict]:
        existing_batters = current_team_rows[
            current_team_rows["batting_order"].notna()
        ] if not current_team_rows.empty else pd.DataFrame()
        if len(existing_batters) >= 8:
            return []

        recent_batters = recent_team_rows[
            recent_team_rows["batting_order"].notna()
            & (recent_team_rows["position"] != "P")
        ].copy()
        if recent_batters.empty:
            return []

        if not current_team_rows.empty:
            current_candidates = set(
                current_team_rows[
                    current_team_rows["position"] != "P"
                ]["player_id"].dropna().astype(int).tolist()
            )
            filtered = recent_batters[recent_batters["player_id"].isin(current_candidates)]
            if filtered["player_id"].nunique() >= 6:
                recent_batters = filtered

        recent_batters["game_date"] = pd.to_datetime(recent_batters["game_date"]).dt.date
        recent_batters["recency_weight"] = recent_batters["game_date"].apply(
            lambda game_date_value: 1.0 / max((target_date - game_date_value).days, 1)
        )

        overall = (
            recent_batters.groupby("player_id")
            .agg(
                recent_score=("recency_weight", "sum"),
                starts=("game_pk", "nunique"),
                avg_order=("batting_order", "mean"),
                last_game_date=("game_date", "max"),
            )
            .reset_index()
            .sort_values(
                ["recent_score", "starts", "last_game_date", "avg_order"],
                ascending=[False, False, False, True],
            )
        )
        if overall.empty:
            return []

        order_scores = (
            recent_batters.groupby(["player_id", "batting_order"])
            .agg(order_score=("recency_weight", "sum"))
            .reset_index()
        )

        metadata_source = pd.concat(
            [
                current_team_rows,
                recent_team_rows,
            ],
            ignore_index=True,
        )
        metadata_source = metadata_source.dropna(subset=["player_id"]).copy()
        if metadata_source.empty:
            return []

        if "game_date" in metadata_source.columns:
            metadata_source["game_date"] = pd.to_datetime(metadata_source["game_date"]).dt.date
            metadata_source = metadata_source.sort_values("game_date")
        latest_meta = metadata_source.groupby("player_id").tail(1).set_index("player_id")

        selected_by_order: dict[int, int] = {}
        selected_player_ids: set[int] = set()

        for batting_order in range(1, 10):
            order_candidates = order_scores[
                (order_scores["batting_order"] == batting_order)
                & (~order_scores["player_id"].isin(selected_player_ids))
            ].merge(
                overall[["player_id", "recent_score", "starts"]],
                on="player_id",
                how="left",
            ).sort_values(
                ["order_score", "recent_score", "starts"],
                ascending=[False, False, False],
            )

            if order_candidates.empty:
                continue

            selected_player_id = int(order_candidates.iloc[0]["player_id"])
            selected_by_order[batting_order] = selected_player_id
            selected_player_ids.add(selected_player_id)

        remaining_orders = [order for order in range(1, 10) if order not in selected_by_order]
        remaining_players = overall[
            ~overall["player_id"].isin(selected_player_ids)
        ].sort_values(
            ["recent_score", "starts", "avg_order"],
            ascending=[False, False, True],
        )

        for batting_order, (_, candidate_row) in zip(remaining_orders, remaining_players.iterrows()):
            selected_player_id = int(candidate_row["player_id"])
            selected_by_order[batting_order] = selected_player_id
            selected_player_ids.add(selected_player_id)

        if len(selected_by_order) < 9:
            return []

        projected_rows = []
        team_id = game.get(f"{team_type}_team_id")
        for batting_order in range(1, 10):
            player_id = selected_by_order[batting_order]
            if player_id not in latest_meta.index:
                continue

            player_meta = latest_meta.loc[player_id]
            projected_rows.append({
                "game_pk": game["game_pk"],
                "game_date": game["game_date"],
                "team_id": team_id,
                "team_type": team_type,
                "player_id": int(player_id),
                "player_name": player_meta.get("player_name", ""),
                "batting_order": batting_order,
                "position": player_meta.get("position", ""),
                "bat_side": player_meta.get("bat_side", ""),
                "pitch_hand": player_meta.get("pitch_hand", ""),
                "is_starter": True,
                "is_probable_pitcher": False,
                "lineup_confirmed": False,
            })

        probable_pitcher_id = game.get(f"{team_type}_probable_pitcher_id")
        if probable_pitcher_id:
            probable_pitcher_id = int(probable_pitcher_id)
            probable_source = metadata_source[
                metadata_source["player_id"] == probable_pitcher_id
            ]
            probable_meta = probable_source.tail(1).iloc[0] if not probable_source.empty else None
            projected_rows.append({
                "game_pk": game["game_pk"],
                "game_date": game["game_date"],
                "team_id": team_id,
                "team_type": team_type,
                "player_id": probable_pitcher_id,
                "player_name": (
                    probable_meta.get("player_name", "")
                    if probable_meta is not None
                    else game.get(f"{team_type}_probable_pitcher", "")
                ),
                "batting_order": None,
                "position": "P",
                "bat_side": probable_meta.get("bat_side", "") if probable_meta is not None else "",
                "pitch_hand": probable_meta.get("pitch_hand", "") if probable_meta is not None else "",
                "is_starter": True,
                "is_probable_pitcher": True,
                "lineup_confirmed": False,
            })

        return projected_rows

    def build_projected_lineups(
        self,
        games: list[dict],
        lineups_df: pd.DataFrame,
        target_date: date,
    ) -> pd.DataFrame:
        team_ids = [
            team_id
            for game in games
            for team_id in (game.get("home_team_id"), game.get("away_team_id"))
            if team_id
        ]
        recent_lineups_df = self.load_recent_confirmed_lineups(team_ids, target_date)
        if recent_lineups_df.empty:
            return pd.DataFrame()

        projected_rows: list[dict] = []
        for game in games:
            game_lineups = lineups_df[lineups_df["game_pk"] == game["game_pk"]] if not lineups_df.empty else pd.DataFrame()
            for team_type in ("home", "away"):
                current_team_rows = game_lineups[game_lineups["team_type"] == team_type] if not game_lineups.empty else pd.DataFrame()
                recent_team_rows = recent_lineups_df[
                    (recent_lineups_df["team_id"] == game.get(f"{team_type}_team_id"))
                    & (recent_lineups_df["team_type"] == team_type)
                ]
                projected_rows.extend(
                    self._project_team_lineup_rows(
                        game,
                        team_type,
                        target_date,
                        current_team_rows,
                        recent_team_rows,
                    )
                )

        if not projected_rows:
            return pd.DataFrame()

        projected_df = pd.DataFrame(projected_rows)
        logger.info(
            "Projected %d lineup rows across %d games",
            len(projected_df),
            projected_df["game_pk"].nunique(),
        )
        return projected_df

    # -----------------------------------------------------------------------
    # Composite matchup advantage
    # -----------------------------------------------------------------------
    def _compute_matchup_advantage(
        self, home_feats: dict, away_feats: dict,
        home_pitcher_feats: dict, away_pitcher_feats: dict
    ) -> float:
        """
        Composite matchup advantage score (positive = home advantage, negative = away).

        Uses DEVIATION from league-neutral values so components contribute positive or
        negative signal relative to average rather than using raw wOBA magnitudes.

        Pitcher direction logic (corrected):
          - Away pitcher high wOBA_allowed  → bad away pitcher → GOOD for home (+)
          - Home pitcher low  wOBA_allowed  → good home pitcher → GOOD for home (+)
          - Away pitcher high K%            → good away pitcher → BAD  for home (−)
          - Home pitcher high K%            → good home pitcher → GOOD for home (+)
        """
        NEUTRAL_WOBA = 0.320
        NEUTRAL_K_PCT = 0.220

        score = 0.0
        weight_total = 0.0

        def _add_dev(val: Optional[float], neutral: float, weight: float, direction: float = 1.0) -> None:
            nonlocal score, weight_total
            if val is not None:
                score += direction * (val - neutral) * weight
                weight_total += weight

        # Home lineup offense vs away pitcher (high wOBA = good for home)
        _add_dev(home_feats.get("lineup_woba_vs_hand"), NEUTRAL_WOBA, 3.0, +1)
        _add_dev(home_feats.get("h2h_woba"), NEUTRAL_WOBA, 2.0, +1)
        _add_dev(home_feats.get("top3_woba_vs_hand"), NEUTRAL_WOBA, 1.5, +1)

        # Away pitcher quality: high wOBA_allowed = bad pitcher = good for home (+)
        # High K% = good pitcher = bad for home (−)
        _add_dev(away_pitcher_feats.get("starter_woba_allowed"), NEUTRAL_WOBA, 2.0, +1)
        _add_dev(away_pitcher_feats.get("starter_k_pct"), NEUTRAL_K_PCT, 1.0, -1)

        # Away lineup offense vs home pitcher (high wOBA = bad for home, direction −)
        _add_dev(away_feats.get("lineup_woba_vs_hand"), NEUTRAL_WOBA, 3.0, -1)
        _add_dev(away_feats.get("h2h_woba"), NEUTRAL_WOBA, 2.0, -1)
        _add_dev(away_feats.get("top3_woba_vs_hand"), NEUTRAL_WOBA, 1.5, -1)

        # Home pitcher quality: low wOBA_allowed = good pitcher = good for home (−dev)
        # High K% = good home pitcher = good for home (+)
        _add_dev(home_pitcher_feats.get("starter_woba_allowed"), NEUTRAL_WOBA, 2.0, -1)
        _add_dev(home_pitcher_feats.get("starter_k_pct"), NEUTRAL_K_PCT, 1.0, +1)

        if weight_total == 0:
            return 0.0

        # raw is a weighted average wOBA deviation (typically ±0.02–0.08 for real games)
        # Normalize: ±0.06 = ±1.0 on the output scale (covers typical game range)
        raw = score / weight_total
        normalized = raw / 0.06
        return round(max(-1.0, min(1.0, normalized)), 4)

    # -----------------------------------------------------------------------
    # Main compute method for one game
    # -----------------------------------------------------------------------
    def compute_for_game(
        self, game: dict, lineups_df: pd.DataFrame,
        statcast_df: pd.DataFrame, pitcher_splits_df: pd.DataFrame,
        platoon_df: pd.DataFrame,
    ) -> Optional[dict]:
        game_pk = game["game_pk"]
        game_date = game["game_date"]

        game_lineups = lineups_df[lineups_df["game_pk"] == game_pk]
        if game_lineups.empty:
            logger.warning("No lineup rows found for game %s", game_pk)
            return None

        home_lineup = game_lineups[game_lineups["team_type"] == "home"]
        away_lineup = game_lineups[game_lineups["team_type"] == "away"]

        # Identify starters (probable pitcher or position == P with batting_order == null)
        home_starter_row = home_lineup[home_lineup["is_probable_pitcher"] == True]
        if home_starter_row.empty:
            home_starter_row = home_lineup[home_lineup["position"] == "P"].head(1)
        away_starter_row = away_lineup[away_lineup["is_probable_pitcher"] == True]
        if away_starter_row.empty:
            away_starter_row = away_lineup[away_lineup["position"] == "P"].head(1)

        home_starter_id = (
            int(home_starter_row["player_id"].iloc[0])
            if not home_starter_row.empty else game.get("home_probable_pitcher_id")
        )
        away_starter_id = (
            int(away_starter_row["player_id"].iloc[0])
            if not away_starter_row.empty else game.get("away_probable_pitcher_id")
        )

        home_starter_hand = (
            str(home_starter_row["pitch_hand"].iloc[0])
            if not home_starter_row.empty and pd.notna(home_starter_row["pitch_hand"].iloc[0])
            else None
        )
        away_starter_hand = (
            str(away_starter_row["pitch_hand"].iloc[0])
            if not away_starter_row.empty and pd.notna(away_starter_row["pitch_hand"].iloc[0])
            else None
        )

        if home_starter_hand is None and not game_pitcher_splits.empty and home_starter_id is not None:
            home_pitcher_hand = game_pitcher_splits[
                game_pitcher_splits["pitcher"] == home_starter_id
            ]
            if not home_pitcher_hand.empty:
                hand = home_pitcher_hand["p_throws"].iloc[0]
                home_starter_hand = str(hand) if pd.notna(hand) else None

        if away_starter_hand is None and not game_pitcher_splits.empty and away_starter_id is not None:
            away_pitcher_hand = game_pitcher_splits[
                game_pitcher_splits["pitcher"] == away_starter_id
            ]
            if not away_pitcher_hand.empty:
                hand = away_pitcher_hand["p_throws"].iloc[0]
                away_starter_hand = str(hand) if pd.notna(hand) else None

        # Filtering statcast to relevant players
        game_batter_ids = (
            list(home_lineup["player_id"].dropna().astype(int).unique()) +
            list(away_lineup["player_id"].dropna().astype(int).unique())
        )
        game_pitcher_ids = [p for p in [home_starter_id, away_starter_id] if p]

        game_statcast = pd.DataFrame()
        if not statcast_df.empty and game_pitcher_ids and game_batter_ids:
            game_statcast = statcast_df[
                (statcast_df["pitcher"].isin(game_pitcher_ids)) &
                (statcast_df["batter"].isin(game_batter_ids))
            ]

        game_pitcher_splits = pd.DataFrame()
        if not pitcher_splits_df.empty and game_pitcher_ids:
            game_pitcher_splits = pitcher_splits_df[
                pitcher_splits_df["pitcher"].isin(game_pitcher_ids)
            ]

        game_platoon = pd.DataFrame()
        if not platoon_df.empty and game_batter_ids:
            game_platoon = platoon_df[
                platoon_df["batter"].isin(game_batter_ids)
            ]

        # Home batters vs away starter
        home_batting_feats = self._compute_lineup_features(
            lineup=home_lineup,
            pitcher_id=away_starter_id,
            pitcher_hand=away_starter_hand,
            statcast_df=game_statcast,
            pitcher_splits_df=game_pitcher_splits,
            platoon_df=game_platoon,
        )

        # Away batters vs home starter
        away_batting_feats = self._compute_lineup_features(
            lineup=away_lineup,
            pitcher_id=home_starter_id,
            pitcher_hand=home_starter_hand,
            statcast_df=game_statcast,
            pitcher_splits_df=game_pitcher_splits,
            platoon_df=game_platoon,
        )

        home_pitcher_feats = self._compute_pitcher_features(
            home_starter_id, game_pitcher_splits
        )
        away_pitcher_feats = self._compute_pitcher_features(
            away_starter_id, game_pitcher_splits
        )

        matchup_advantage = self._compute_matchup_advantage(
            home_batting_feats, away_batting_feats,
            home_pitcher_feats, away_pitcher_feats
        )

        lineup_confirmed = bool(
            game_lineups["lineup_confirmed"].any()
            if "lineup_confirmed" in game_lineups.columns
            else False
        )

        row = {
            "game_pk": game_pk,
            "game_date": game_date,
            "home_team_id": game.get("home_team_id"),
            "away_team_id": game.get("away_team_id"),
            # Home batting vs away starter
            "home_lineup_woba_vs_hand": home_batting_feats["lineup_woba_vs_hand"],
            "home_lineup_k_pct_vs_hand": home_batting_feats["lineup_k_pct_vs_hand"],
            "home_lineup_bb_pct_vs_hand": home_batting_feats["lineup_bb_pct_vs_hand"],
            "home_lineup_iso_vs_hand": home_batting_feats["lineup_iso_vs_hand"],
            "home_top3_woba_vs_hand": home_batting_feats["top3_woba_vs_hand"],
            "home_middle4_woba_vs_hand": home_batting_feats["middle4_woba_vs_hand"],
            "home_bottom2_woba_vs_hand": home_batting_feats["bottom2_woba_vs_hand"],
            "home_pct_same_hand": home_batting_feats["pct_same_hand"],
            "home_h2h_woba": home_batting_feats["h2h_woba"],
            "home_h2h_pa_total": home_batting_feats["h2h_pa_total"],
            "home_h2h_k_pct": home_batting_feats["h2h_k_pct"],
            "home_h2h_hr_rate": home_batting_feats["h2h_hr_rate"],
            # Away batting vs home starter
            "away_lineup_woba_vs_hand": away_batting_feats["lineup_woba_vs_hand"],
            "away_lineup_k_pct_vs_hand": away_batting_feats["lineup_k_pct_vs_hand"],
            "away_lineup_bb_pct_vs_hand": away_batting_feats["lineup_bb_pct_vs_hand"],
            "away_lineup_iso_vs_hand": away_batting_feats["lineup_iso_vs_hand"],
            "away_top3_woba_vs_hand": away_batting_feats["top3_woba_vs_hand"],
            "away_middle4_woba_vs_hand": away_batting_feats["middle4_woba_vs_hand"],
            "away_bottom2_woba_vs_hand": away_batting_feats["bottom2_woba_vs_hand"],
            "away_pct_same_hand": away_batting_feats["pct_same_hand"],
            "away_h2h_woba": away_batting_feats["h2h_woba"],
            "away_h2h_pa_total": away_batting_feats["h2h_pa_total"],
            "away_h2h_k_pct": away_batting_feats["h2h_k_pct"],
            "away_h2h_hr_rate": away_batting_feats["h2h_hr_rate"],
            # Pitchers
            "home_starter_id": home_starter_id,
            "home_starter_hand": home_pitcher_feats.get("starter_hand"),
            "home_starter_woba_allowed": home_pitcher_feats.get("starter_woba_allowed"),
            "home_starter_k_pct": home_pitcher_feats.get("starter_k_pct"),
            "home_starter_whip": home_pitcher_feats.get("starter_whip"),
            "away_starter_id": away_starter_id,
            "away_starter_hand": away_pitcher_feats.get("starter_hand"),
            "away_starter_woba_allowed": away_pitcher_feats.get("starter_woba_allowed"),
            "away_starter_k_pct": away_pitcher_feats.get("starter_k_pct"),
            "away_starter_whip": away_pitcher_feats.get("starter_whip"),
            # Composite
            "matchup_advantage_home": matchup_advantage,
            "lineup_confirmed": lineup_confirmed,
            "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        return row

    # -----------------------------------------------------------------------
    # Public run methods
    # -----------------------------------------------------------------------
    def run_for_date(self, target_date: date) -> dict:
        """Build matchup features for all games on a given date."""
        games = self.fetch_game_metadata(target_date)
        if not games:
            logger.info("No games found for %s", target_date)
            return {"games": 0, "rows_written": 0}

        game_pks = [g["game_pk"] for g in games]
        return self._run_for_games(games, game_pks)

    def run_for_game_pks(
        self, game_pks: list[int], game_date: Optional[date] = None
    ) -> dict:
        target_date = game_date or date.today()
        all_games = self.fetch_game_metadata(target_date)
        games = [g for g in all_games if g["game_pk"] in game_pks]
        return self._run_for_games(games, game_pks)

    def _run_for_games(self, games: list[dict], game_pks: list[int]) -> dict:
        lineups_df = self.load_lineups(game_pks)
        target_date = date.fromisoformat(games[0]["game_date"]) if games else date.today()
        projected_lineups_df = self.build_projected_lineups(games, lineups_df, target_date)
        if not projected_lineups_df.empty:
            lineups_df = pd.concat([lineups_df, projected_lineups_df], ignore_index=True)

        if lineups_df.empty:
            logger.warning("No lineup data found for games %s — did lineups fetch run?", game_pks)
            return {"games": len(games), "rows_written": 0, "warning": "no_lineups"}

        # Collect all pitcher and batter IDs to batch the Statcast query
        pitcher_ids = []
        batter_ids = []
        for game in games:
            game_lineups = lineups_df[lineups_df["game_pk"] == game["game_pk"]]
            probable_pitchers = game_lineups[game_lineups["is_probable_pitcher"] == True]["player_id"]
            all_batters = game_lineups[
                game_lineups["batting_order"].notna()
            ]["player_id"]
            pitcher_ids.extend(probable_pitchers.astype(int).tolist())
            batter_ids.extend(all_batters.astype(int).tolist())

        pitcher_ids = list(set(pitcher_ids))
        batter_ids = list(set(batter_ids))

        logger.info(
            "Loading statcast data for %d pitchers and %d batters",
            len(pitcher_ids), len(batter_ids)
        )

        # Load H2H (pitcher-specific), pitcher overall splits, and batter career platoon splits
        statcast_df = self.load_statcast_splits(pitcher_ids, batter_ids)
        pitcher_splits_df = self.load_pitcher_overall_splits(pitcher_ids)
        platoon_df = self.load_batter_career_splits(batter_ids)

        rows = []
        for game in games:
            row = self.compute_for_game(game, lineups_df, statcast_df, pitcher_splits_df, platoon_df)
            if row:
                rows.append(row)

        if rows and not self.dry_run:
            df = pd.DataFrame(rows)
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
            df["computed_at"] = pd.to_datetime(df["computed_at"], utc=True, errors="coerce")
            job_config = bigquery.LoadJobConfig(
                schema=MATCHUP_SCHEMA,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            )
            job = self.bq.load_table_from_dataframe(df, MATCHUP_TABLE, job_config=job_config)
            job.result()
            logger.info("Inserted %d matchup feature rows", len(rows))
        elif rows and self.dry_run:
            logger.info("[DRY RUN] Would insert %d matchup feature rows", len(rows))

        return {"games": len(games), "rows_written": len(rows)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build pitcher/batter matchup features")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--game-pk", help="Comma-separated game PKs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    builder = MatchupFeatureBuilder(dry_run=args.dry_run)

    if args.game_pk:
        pks = [int(pk.strip()) for pk in args.game_pk.split(",")]
        target = date.fromisoformat(args.date) if args.date else date.today()
        result = builder.run_for_game_pks(pks, target)
    else:
        target = date.fromisoformat(args.date) if args.date else date.today()
        result = builder.run_for_date(target)

    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()
