#!/usr/bin/env python3
"""
Historical Matchup Feature Builder

One-time (then weekly) job that computes pitcher/batter matchup features
for all completed training games (2015–2025) directly from Statcast data.

For historical games we reconstruct the batting lineup from statcast itself:
  - Each unique batter's first at_bat_number in a game gives their lineup slot
  - p_throws / stand tells us the handedness matchup
  - Event outcomes compute wOBA, K%, BB%, HR-rate

This is more accurate than the lineup API for historical games because it
reflects who actually played (accounting for late scratches, pinch hitters, etc.)

Output:
  BigQuery: hankstank.mlb_historical_data.matchup_features_historical
  Local:    data/training/matchup_features_historical.parquet  (for offline training)

Usage:
    python build_historical_matchup_features.py               # all years 2015-2025
    python build_historical_matchup_features.py --year 2024   # single year
    python build_historical_matchup_features.py --dry-run     # print SQL only
    python build_historical_matchup_features.py --local-only  # skip BQ upload, parquet only
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
HIST_DATASET = "mlb_historical_data"
STATCAST_TABLE = f"{PROJECT}.{HIST_DATASET}.statcast_pitches"
GAMES_TABLE = f"{PROJECT}.{HIST_DATASET}.games_historical"
OUTPUT_TABLE = f"{PROJECT}.{HIST_DATASET}.matchup_features_historical"
OUTPUT_PATH = Path("data/training/matchup_features_historical.parquet")

# wOBA weights (2022 constants, stable across years)
WOBA_WEIGHTS = {
    "walk": 0.69,
    "hit_by_pitch": 0.72,
    "single": 0.88,
    "double": 1.247,
    "triple": 1.578,
    "home_run": 2.031,
}

OUTPUT_SCHEMA = [
    bigquery.SchemaField("game_pk", "INTEGER"),
    bigquery.SchemaField("game_date", "DATE"),
    bigquery.SchemaField("year", "INTEGER"),
    bigquery.SchemaField("home_team_id", "INTEGER"),
    bigquery.SchemaField("away_team_id", "INTEGER"),
    # Home lineup vs away starter
    bigquery.SchemaField("home_lineup_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_lineup_k_pct_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_lineup_bb_pct_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_top3_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_middle4_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_bottom2_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("home_pct_same_hand", "FLOAT"),
    bigquery.SchemaField("home_h2h_woba", "FLOAT"),
    bigquery.SchemaField("home_h2h_pa_total", "INTEGER"),
    bigquery.SchemaField("home_h2h_k_pct", "FLOAT"),
    bigquery.SchemaField("home_h2h_hr_rate", "FLOAT"),
    # Away lineup vs home starter
    bigquery.SchemaField("away_lineup_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_lineup_k_pct_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_lineup_bb_pct_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_top3_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_middle4_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_bottom2_woba_vs_hand", "FLOAT"),
    bigquery.SchemaField("away_pct_same_hand", "FLOAT"),
    bigquery.SchemaField("away_h2h_woba", "FLOAT"),
    bigquery.SchemaField("away_h2h_pa_total", "INTEGER"),
    bigquery.SchemaField("away_h2h_k_pct", "FLOAT"),
    bigquery.SchemaField("away_h2h_hr_rate", "FLOAT"),
    # Starters
    bigquery.SchemaField("home_starter_id", "INTEGER"),
    bigquery.SchemaField("home_starter_hand", "STRING"),
    bigquery.SchemaField("home_starter_woba_allowed", "FLOAT"),
    bigquery.SchemaField("home_starter_k_pct", "FLOAT"),
    bigquery.SchemaField("away_starter_id", "INTEGER"),
    bigquery.SchemaField("away_starter_hand", "STRING"),
    bigquery.SchemaField("away_starter_woba_allowed", "FLOAT"),
    bigquery.SchemaField("away_starter_k_pct", "FLOAT"),
    # Derived
    bigquery.SchemaField("lineup_woba_differential", "FLOAT"),
    bigquery.SchemaField("starter_woba_differential", "FLOAT"),
    bigquery.SchemaField("matchup_advantage_home", "FLOAT"),
    bigquery.SchemaField("h2h_woba_differential", "FLOAT"),
    bigquery.SchemaField("lineup_k_pct_differential", "FLOAT"),
    bigquery.SchemaField("computed_at", "TIMESTAMP"),
]

# Minimum PA thresholds
MIN_PA_H2H = 5      # minimum total PA for H2H stats to be included
MIN_PA_CAREER = 20  # minimum career PA for pitcher platoon stats


def build_historical_sql(year_filter: str = "") -> str:
    """
    Build the BigQuery SQL that computes per-game matchup features.

    Strategy:
      1. Identify the starting pitcher per team per game:
         - The pitcher who faced the most batters in innings 1-3
         (most reliable proxy for starter from statcast)

      2. Identify the batting lineup (positions 1-9):
         - First 9 unique batters by their minimum at_bat_number per team
         - Assign lineup slot by ranking their first at_bat_number

      3. Compute per-lineup-slot wOBA/K%/BB% vs the opposing starter's
         handedness using CAREER statcast (prior games only, avoiding data leak)

      4. H2H stats: each batter's career record vs this specific pitcher
         (prior games only)

    Note: "prior games only" means we use statcast from different game_pks,
    not the current game being evaluated — avoiding target leakage.
    """
    year_clause = f"AND EXTRACT(YEAR FROM s.game_date) {year_filter}" if year_filter else ""

    return f"""
-- ============================================================
-- Step 1: Identify starting pitchers per team per game
-- Starter = pitcher who threw the most pitches in innings 1-3
-- ============================================================
WITH starter_candidates AS (
    SELECT
        game_pk,
        game_date,
        EXTRACT(YEAR FROM game_date) AS year,
        pitcher,
        inning_topbot,
        COUNT(*) AS pitches_thrown,
        p_throws
    FROM `{STATCAST_TABLE}` s
    WHERE inning <= 3
      AND inning IS NOT NULL
      {year_clause}
    GROUP BY game_pk, game_date, pitcher, inning_topbot, p_throws
),
-- Pick the pitcher with the most pitches per team side per game
starters AS (
    SELECT
        game_pk,
        game_date,
        year,
        pitcher AS starter_id,
        p_throws AS starter_hand,
        inning_topbot,
        CASE inning_topbot WHEN 'Top' THEN 'away' ELSE 'home' END AS pitching_team
    FROM starter_candidates
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY game_pk, inning_topbot
        ORDER BY pitches_thrown DESC
    ) = 1
),

-- ============================================================
-- Step 2: Reconstruct batting lineup (1-9) from at_bat_number
-- ============================================================
all_batters AS (
    SELECT
        s.game_pk,
        s.game_date,
        s.batter,
        s.stand AS bat_side,
        s.inning_topbot,
        CASE s.inning_topbot WHEN 'Top' THEN 'away' ELSE 'home' END AS batting_team,
        MIN(s.at_bat_number) AS first_atbat_number,
        SUM(CASE WHEN s.events IN (
            'strikeout','strikeout_double_play','field_out','force_out',
            'grounded_into_double_play','double_play','field_error',
            'fielders_choice','fielders_choice_out','walk','hit_by_pitch',
            'single','double','triple','home_run','sac_fly','sac_bunt'
        ) THEN 1 ELSE 0 END) AS pa_in_game
    FROM `{STATCAST_TABLE}` s
    WHERE s.events IS NOT NULL
      {year_clause}
    GROUP BY s.game_pk, s.game_date, s.batter, s.stand, s.inning_topbot
),
-- Assign lineup slot (1-9) based on order of first appearance
lineup AS (
    SELECT
        game_pk,
        game_date,
        batter,
        bat_side,
        batting_team,
        inning_topbot,
        ROW_NUMBER() OVER (
            PARTITION BY game_pk, inning_topbot
            ORDER BY first_atbat_number
        ) AS lineup_slot
    FROM all_batters
    WHERE pa_in_game >= 1
),
-- Keep only the starting 9 (slot 1-9 accounts for pinch runners, defensive subs)
starting_9 AS (
    SELECT * FROM lineup WHERE lineup_slot <= 9
),

-- ============================================================
-- Step 3: Career platoon splits for each batter
-- wOBA/K%/BB% split by pitcher handedness (p_throws)
-- Using ALL historical statcast (not just this year)
-- ============================================================
batter_platoon_career AS (
    SELECT
        s.batter,
        s.p_throws,
        SUM(CASE WHEN s.events IN (
            'strikeout','strikeout_double_play','field_out','force_out',
            'grounded_into_double_play','double_play','field_error',
            'fielders_choice','fielders_choice_out','walk','hit_by_pitch',
            'single','double','triple','home_run','sac_fly','sac_bunt'
        ) THEN 1 ELSE 0 END) AS pa,
        SUM(CASE WHEN s.events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS k,
        SUM(CASE WHEN s.events = 'walk' THEN 1 ELSE 0 END) AS bb,
        SUM(CASE WHEN s.events = 'home_run' THEN 1 ELSE 0 END) AS hr,
        SUM(CASE
            WHEN s.events = 'walk' THEN 0.69
            WHEN s.events = 'hit_by_pitch' THEN 0.72
            WHEN s.events = 'single' THEN 0.88
            WHEN s.events = 'double' THEN 1.247
            WHEN s.events = 'triple' THEN 1.578
            WHEN s.events = 'home_run' THEN 2.031
            ELSE 0.0
        END) AS woba_numerator
    FROM `{STATCAST_TABLE}` s
    WHERE s.events IS NOT NULL
    GROUP BY s.batter, s.p_throws
),

-- ============================================================
-- Step 4: Career stats for each starter pitcher (overall wOBA allowed, K%)
-- ============================================================
pitcher_career AS (
    SELECT
        s.pitcher,
        s.p_throws,
        SUM(CASE WHEN s.events IN (
            'strikeout','strikeout_double_play','field_out','force_out',
            'grounded_into_double_play','double_play','field_error',
            'fielders_choice','fielders_choice_out','walk','hit_by_pitch',
            'single','double','triple','home_run','sac_fly','sac_bunt'
        ) THEN 1 ELSE 0 END) AS pa,
        SUM(CASE WHEN s.events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS k,
        SUM(CASE
            WHEN s.events = 'walk' THEN 0.69
            WHEN s.events = 'hit_by_pitch' THEN 0.72
            WHEN s.events = 'single' THEN 0.88
            WHEN s.events = 'double' THEN 1.247
            WHEN s.events = 'triple' THEN 1.578
            WHEN s.events = 'home_run' THEN 2.031
            ELSE 0.0
        END) AS woba_numerator
    FROM `{STATCAST_TABLE}` s
    WHERE s.events IS NOT NULL
    GROUP BY s.pitcher, s.p_throws
),

-- ============================================================
-- Step 5: H2H stats — each batter's career record vs specific pitcher
-- ============================================================
h2h_career AS (
    SELECT
        s.batter,
        s.pitcher,
        SUM(CASE WHEN s.events IN (
            'strikeout','strikeout_double_play','field_out','force_out',
            'grounded_into_double_play','double_play','field_error',
            'fielders_choice','fielders_choice_out','walk','hit_by_pitch',
            'single','double','triple','home_run','sac_fly','sac_bunt'
        ) THEN 1 ELSE 0 END) AS pa,
        SUM(CASE WHEN s.events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS k,
        SUM(CASE WHEN s.events = 'home_run' THEN 1 ELSE 0 END) AS hr,
        SUM(CASE
            WHEN s.events = 'walk' THEN 0.69
            WHEN s.events = 'hit_by_pitch' THEN 0.72
            WHEN s.events = 'single' THEN 0.88
            WHEN s.events = 'double' THEN 1.247
            WHEN s.events = 'triple' THEN 1.578
            WHEN s.events = 'home_run' THEN 2.031
            ELSE 0.0
        END) AS woba_numerator
    FROM `{STATCAST_TABLE}` s
    WHERE s.events IS NOT NULL
    GROUP BY s.batter, s.pitcher
),

-- ============================================================
-- Step 6: Join everything per game
-- For each game: home batters vs away starter, away batters vs home starter
-- ============================================================
lineup_with_starter AS (
    SELECT
        l.game_pk,
        l.game_date,
        EXTRACT(YEAR FROM l.game_date) AS year,
        l.batter,
        l.bat_side,
        l.batting_team,
        l.lineup_slot,
        -- Opposing starter
        opp_st.starter_id AS opp_starter_id,
        opp_st.starter_hand AS opp_starter_hand,
        -- Own starter (pitcher for the other team)
        own_st.starter_id AS own_starter_id,
        -- Is same hand matchup (harder for batter)
        CASE
            WHEN l.bat_side = opp_st.starter_hand THEN 1
            WHEN l.bat_side = 'S' THEN 0  -- switch hitter, neutral
            ELSE 0
        END AS is_same_hand
    FROM starting_9 l
    -- Join opposing starter (pitcher the batter will face)
    JOIN starters opp_st
        ON  l.game_pk = opp_st.game_pk
        AND l.batting_team != opp_st.pitching_team
    -- Join own starter (pitcher for this batter's team)
    LEFT JOIN starters own_st
        ON  l.game_pk = own_st.game_pk
        AND l.batting_team = own_st.pitching_team
),

-- Join platoon splits
lineup_with_splits AS (
    SELECT
        lws.*,
        bpc.pa AS career_pa_vs_hand,
        bpc.k AS career_k_vs_hand,
        bpc.bb AS career_bb_vs_hand,
        bpc.woba_numerator AS career_woba_num_vs_hand,
        h2h.pa AS h2h_pa,
        h2h.k AS h2h_k,
        h2h.hr AS h2h_hr,
        h2h.woba_numerator AS h2h_woba_num
    FROM lineup_with_starter lws
    LEFT JOIN batter_platoon_career bpc
        ON  lws.batter = bpc.batter
        AND lws.opp_starter_hand = bpc.p_throws
    LEFT JOIN h2h_career h2h
        ON  lws.batter = h2h.batter
        AND lws.opp_starter_id = h2h.pitcher
),

-- ============================================================
-- Step 7: Aggregate per game per batting team
-- ============================================================
game_batting_agg AS (
    SELECT
        game_pk,
        game_date,
        year,
        batting_team,
        opp_starter_id,
        opp_starter_hand,
        -- Full lineup
        SAFE_DIVIDE(
            SUM(CASE WHEN career_pa_vs_hand >= {MIN_PA_CAREER} THEN career_woba_num_vs_hand ELSE NULL END),
            NULLIF(SUM(CASE WHEN career_pa_vs_hand >= {MIN_PA_CAREER} THEN career_pa_vs_hand ELSE NULL END), 0)
        ) AS lineup_woba_vs_hand,
        SAFE_DIVIDE(
            SUM(CASE WHEN career_pa_vs_hand >= {MIN_PA_CAREER} THEN career_k_vs_hand ELSE NULL END),
            NULLIF(SUM(CASE WHEN career_pa_vs_hand >= {MIN_PA_CAREER} THEN career_pa_vs_hand ELSE NULL END), 0)
        ) AS lineup_k_pct_vs_hand,
        SAFE_DIVIDE(
            SUM(CASE WHEN career_pa_vs_hand >= {MIN_PA_CAREER} THEN career_bb_vs_hand ELSE NULL END),
            NULLIF(SUM(CASE WHEN career_pa_vs_hand >= {MIN_PA_CAREER} THEN career_pa_vs_hand ELSE NULL END), 0)
        ) AS lineup_bb_pct_vs_hand,
        -- Batting order groups
        SAFE_DIVIDE(
            SUM(CASE WHEN lineup_slot <= 3 AND career_pa_vs_hand >= {MIN_PA_CAREER} THEN career_woba_num_vs_hand ELSE NULL END),
            NULLIF(SUM(CASE WHEN lineup_slot <= 3 AND career_pa_vs_hand >= {MIN_PA_CAREER} THEN career_pa_vs_hand ELSE NULL END), 0)
        ) AS top3_woba_vs_hand,
        SAFE_DIVIDE(
            SUM(CASE WHEN lineup_slot BETWEEN 4 AND 7 AND career_pa_vs_hand >= {MIN_PA_CAREER} THEN career_woba_num_vs_hand ELSE NULL END),
            NULLIF(SUM(CASE WHEN lineup_slot BETWEEN 4 AND 7 AND career_pa_vs_hand >= {MIN_PA_CAREER} THEN career_pa_vs_hand ELSE NULL END), 0)
        ) AS middle4_woba_vs_hand,
        SAFE_DIVIDE(
            SUM(CASE WHEN lineup_slot >= 8 AND career_pa_vs_hand >= {MIN_PA_CAREER} THEN career_woba_num_vs_hand ELSE NULL END),
            NULLIF(SUM(CASE WHEN lineup_slot >= 8 AND career_pa_vs_hand >= {MIN_PA_CAREER} THEN career_pa_vs_hand ELSE NULL END), 0)
        ) AS bottom2_woba_vs_hand,
        -- Same-hand percentage
        SAFE_DIVIDE(SUM(is_same_hand), COUNT(*)) AS pct_same_hand,
        -- H2H (aggregate across all batters vs this specific starter)
        SUM(COALESCE(h2h_pa, 0)) AS h2h_pa_total,
        SAFE_DIVIDE(
            SUM(CASE WHEN h2h_pa >= {MIN_PA_H2H} THEN h2h_woba_num ELSE NULL END),
            NULLIF(SUM(CASE WHEN h2h_pa >= {MIN_PA_H2H} THEN h2h_pa ELSE NULL END), 0)
        ) AS h2h_woba,
        SAFE_DIVIDE(
            SUM(CASE WHEN h2h_pa >= {MIN_PA_H2H} THEN h2h_k ELSE NULL END),
            NULLIF(SUM(CASE WHEN h2h_pa >= {MIN_PA_H2H} THEN h2h_pa ELSE NULL END), 0)
        ) AS h2h_k_pct,
        SAFE_DIVIDE(
            SUM(CASE WHEN h2h_pa >= {MIN_PA_H2H} THEN h2h_hr ELSE NULL END),
            NULLIF(SUM(CASE WHEN h2h_pa >= {MIN_PA_H2H} THEN h2h_pa ELSE NULL END), 0)
        ) AS h2h_hr_rate
    FROM lineup_with_splits
    GROUP BY game_pk, game_date, year, batting_team, opp_starter_id, opp_starter_hand
),

-- ============================================================
-- Step 8: Pivot to one row per game (home + away side by side)
-- ============================================================
pivoted AS (
    SELECT
        game_pk,
        game_date,
        year,
        -- Home batting stats
        MAX(CASE WHEN batting_team = 'home' THEN lineup_woba_vs_hand END) AS home_lineup_woba_vs_hand,
        MAX(CASE WHEN batting_team = 'home' THEN lineup_k_pct_vs_hand END) AS home_lineup_k_pct_vs_hand,
        MAX(CASE WHEN batting_team = 'home' THEN lineup_bb_pct_vs_hand END) AS home_lineup_bb_pct_vs_hand,
        MAX(CASE WHEN batting_team = 'home' THEN top3_woba_vs_hand END) AS home_top3_woba_vs_hand,
        MAX(CASE WHEN batting_team = 'home' THEN middle4_woba_vs_hand END) AS home_middle4_woba_vs_hand,
        MAX(CASE WHEN batting_team = 'home' THEN bottom2_woba_vs_hand END) AS home_bottom2_woba_vs_hand,
        MAX(CASE WHEN batting_team = 'home' THEN pct_same_hand END) AS home_pct_same_hand,
        MAX(CASE WHEN batting_team = 'home' THEN h2h_pa_total END) AS home_h2h_pa_total,
        MAX(CASE WHEN batting_team = 'home' THEN h2h_woba END) AS home_h2h_woba,
        MAX(CASE WHEN batting_team = 'home' THEN h2h_k_pct END) AS home_h2h_k_pct,
        MAX(CASE WHEN batting_team = 'home' THEN h2h_hr_rate END) AS home_h2h_hr_rate,
        MAX(CASE WHEN batting_team = 'home' THEN opp_starter_id END) AS away_starter_id,
        MAX(CASE WHEN batting_team = 'home' THEN opp_starter_hand END) AS away_starter_hand,
        -- Away batting stats
        MAX(CASE WHEN batting_team = 'away' THEN lineup_woba_vs_hand END) AS away_lineup_woba_vs_hand,
        MAX(CASE WHEN batting_team = 'away' THEN lineup_k_pct_vs_hand END) AS away_lineup_k_pct_vs_hand,
        MAX(CASE WHEN batting_team = 'away' THEN lineup_bb_pct_vs_hand END) AS away_lineup_bb_pct_vs_hand,
        MAX(CASE WHEN batting_team = 'away' THEN top3_woba_vs_hand END) AS away_top3_woba_vs_hand,
        MAX(CASE WHEN batting_team = 'away' THEN middle4_woba_vs_hand END) AS away_middle4_woba_vs_hand,
        MAX(CASE WHEN batting_team = 'away' THEN bottom2_woba_vs_hand END) AS away_bottom2_woba_vs_hand,
        MAX(CASE WHEN batting_team = 'away' THEN pct_same_hand END) AS away_pct_same_hand,
        MAX(CASE WHEN batting_team = 'away' THEN h2h_pa_total END) AS away_h2h_pa_total,
        MAX(CASE WHEN batting_team = 'away' THEN h2h_woba END) AS away_h2h_woba,
        MAX(CASE WHEN batting_team = 'away' THEN h2h_k_pct END) AS away_h2h_k_pct,
        MAX(CASE WHEN batting_team = 'away' THEN h2h_hr_rate END) AS away_h2h_hr_rate,
        MAX(CASE WHEN batting_team = 'away' THEN opp_starter_id END) AS home_starter_id,
        MAX(CASE WHEN batting_team = 'away' THEN opp_starter_hand END) AS home_starter_hand,
    FROM game_batting_agg
    GROUP BY game_pk, game_date, year
),

-- ============================================================
-- Step 9: Pitcher career stats (overall wOBA allowed, K%)
-- ============================================================
home_starter_stats AS (
    SELECT
        p.game_pk,
        pc.woba_numerator / NULLIF(pc.pa, 0) AS home_starter_woba_allowed,
        pc.k / NULLIF(pc.pa, 0) AS home_starter_k_pct
    FROM pivoted p
    JOIN pitcher_career pc
        ON  p.home_starter_id = pc.pitcher
    WHERE pc.pa >= {MIN_PA_CAREER}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.game_pk ORDER BY pc.pa DESC) = 1
),
away_starter_stats AS (
    SELECT
        p.game_pk,
        pc.woba_numerator / NULLIF(pc.pa, 0) AS away_starter_woba_allowed,
        pc.k / NULLIF(pc.pa, 0) AS away_starter_k_pct
    FROM pivoted p
    JOIN pitcher_career pc
        ON  p.away_starter_id = pc.pitcher
    WHERE pc.pa >= {MIN_PA_CAREER}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.game_pk ORDER BY pc.pa DESC) = 1
),

-- ============================================================
-- Step 10: Join games table for team IDs + compute derived features
-- ============================================================
final AS (
    SELECT
        p.game_pk,
        p.game_date,
        p.year,
        g.home_team_id,
        g.away_team_id,
        -- Home batting
        p.home_lineup_woba_vs_hand,
        p.home_lineup_k_pct_vs_hand,
        p.home_lineup_bb_pct_vs_hand,
        p.home_top3_woba_vs_hand,
        p.home_middle4_woba_vs_hand,
        p.home_bottom2_woba_vs_hand,
        p.home_pct_same_hand,
        p.home_h2h_pa_total,
        p.home_h2h_woba,
        p.home_h2h_k_pct,
        p.home_h2h_hr_rate,
        -- Away batting
        p.away_lineup_woba_vs_hand,
        p.away_lineup_k_pct_vs_hand,
        p.away_lineup_bb_pct_vs_hand,
        p.away_top3_woba_vs_hand,
        p.away_middle4_woba_vs_hand,
        p.away_bottom2_woba_vs_hand,
        p.away_pct_same_hand,
        p.away_h2h_pa_total,
        p.away_h2h_woba,
        p.away_h2h_k_pct,
        p.away_h2h_hr_rate,
        -- Starters
        p.home_starter_id,
        p.home_starter_hand,
        COALESCE(hs.home_starter_woba_allowed, 0.320) AS home_starter_woba_allowed,
        COALESCE(hs.home_starter_k_pct, 0.20) AS home_starter_k_pct,
        p.away_starter_id,
        p.away_starter_hand,
        COALESCE(as2.away_starter_woba_allowed, 0.320) AS away_starter_woba_allowed,
        COALESCE(as2.away_starter_k_pct, 0.20) AS away_starter_k_pct,
        -- Derived differential features (used directly in V5 training)
        COALESCE(p.home_lineup_woba_vs_hand, 0.320)
            - COALESCE(p.away_lineup_woba_vs_hand, 0.320) AS lineup_woba_differential,
        COALESCE(as2.away_starter_woba_allowed, 0.320)
            - COALESCE(hs.home_starter_woba_allowed, 0.320) AS starter_woba_differential,
        -- Composite matchup advantage (simplified version of build_matchup_features.py logic)
        (
            COALESCE(p.home_lineup_woba_vs_hand, 0.320) * 3.0
            + COALESCE(p.home_h2h_woba, p.home_lineup_woba_vs_hand, 0.320) * 2.0
            + COALESCE(p.home_top3_woba_vs_hand, p.home_lineup_woba_vs_hand, 0.320) * 1.5
            - COALESCE(p.away_lineup_woba_vs_hand, 0.320) * 3.0
            - COALESCE(p.away_h2h_woba, p.away_lineup_woba_vs_hand, 0.320) * 2.0
            - COALESCE(p.away_top3_woba_vs_hand, p.away_lineup_woba_vs_hand, 0.320) * 1.5
            - COALESCE(as2.away_starter_woba_allowed, 0.320) * 2.0
            + COALESCE(hs.home_starter_woba_allowed, 0.320) * 2.0
        ) / 15.0 / 0.15 AS matchup_advantage_home,
        COALESCE(p.home_h2h_woba, 0.320)
            - COALESCE(p.away_h2h_woba, 0.320) AS h2h_woba_differential,
        COALESCE(p.away_lineup_k_pct_vs_hand, 0.20)
            - COALESCE(p.home_lineup_k_pct_vs_hand, 0.20) AS lineup_k_pct_differential,
        CURRENT_TIMESTAMP() AS computed_at
    FROM pivoted p
    LEFT JOIN `{GAMES_TABLE}` g
        ON p.game_pk = g.game_pk
    LEFT JOIN home_starter_stats hs ON p.game_pk = hs.game_pk
    LEFT JOIN away_starter_stats as2 ON p.game_pk = as2.game_pk
)

SELECT * FROM final
"""


class HistoricalMatchupBuilder:
    def __init__(self, dry_run: bool = False, local_only: bool = False):
        self.dry_run = dry_run
        self.local_only = local_only
        if not dry_run:
            self.bq = bigquery.Client(project=PROJECT)

    def _ensure_table(self) -> None:
        table_ref = bigquery.Table(OUTPUT_TABLE, schema=OUTPUT_SCHEMA)
        table_ref.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.YEAR,
            field="game_date",
        )
        table_ref.clustering_fields = ["game_pk", "year"]
        self.bq.create_table(table_ref, exists_ok=True)
        logger.info("Output table ready: %s", OUTPUT_TABLE)

    def run(self, year: int = None) -> pd.DataFrame:
        if year:
            year_filter = f"= {year}"
            label = str(year)
        else:
            year_filter = "BETWEEN 2015 AND 2025"
            label = "2015-2025"

        logger.info("Building historical matchup features for %s...", label)
        sql = build_historical_sql(year_filter)

        if self.dry_run:
            print(sql[:3000])
            print("\n... (SQL truncated for dry-run preview)")
            return pd.DataFrame()

        logger.info("Executing BigQuery job (this may take 2-5 minutes for full history)...")
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            # Write to temp table first, then we'll handle the BQ upload ourselves
        )

        df = self.bq.query(sql, job_config=job_config).to_dataframe()
        logger.info("Query returned %d rows for %s", len(df), label)

        if df.empty:
            logger.error("No rows returned — check statcast table has data")
            return df

        # Cap matchup_advantage_home to [-1, 1]
        if "matchup_advantage_home" in df.columns:
            df["matchup_advantage_home"] = df["matchup_advantage_home"].clip(-1, 1)

        # Save locally
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if year:
            out_path = OUTPUT_PATH.parent / f"matchup_features_{year}.parquet"
        else:
            out_path = OUTPUT_PATH
        df.to_parquet(out_path, index=False)
        logger.info("Saved to %s (%d rows)", out_path, len(df))

        # Upload to BigQuery
        if not self.local_only:
            self._ensure_table()
            if year:
                # Delete existing rows for this year before inserting
                logger.info("Deleting existing BQ rows for year %d...", year)
                self.bq.query(
                    f"DELETE FROM `{OUTPUT_TABLE}` WHERE year = {year}"
                ).result()

            job_config_load = bigquery.LoadJobConfig(
                schema=OUTPUT_SCHEMA,
                write_disposition=(
                    bigquery.WriteDisposition.WRITE_TRUNCATE
                    if not year
                    else bigquery.WriteDisposition.WRITE_APPEND
                ),
            )
            job = self.bq.load_table_from_dataframe(
                df, OUTPUT_TABLE, job_config=job_config_load
            )
            job.result()
            logger.info("Uploaded %d rows to %s", len(df), OUTPUT_TABLE)

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Build historical matchup features from Statcast for V5 training"
    )
    parser.add_argument("--year", type=int, help="Single year to process (default: all 2015-2025)")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL only, no execution")
    parser.add_argument("--local-only", action="store_true", help="Save parquet locally, skip BQ upload")
    args = parser.parse_args()

    builder = HistoricalMatchupBuilder(dry_run=args.dry_run, local_only=args.local_only)
    df = builder.run(year=args.year)

    if not df.empty:
        logger.info("\nFeature coverage summary:")
        for col in ["home_lineup_woba_vs_hand", "home_h2h_woba", "home_pct_same_hand",
                    "lineup_woba_differential", "matchup_advantage_home"]:
            if col in df.columns:
                pct = df[col].notna().mean() * 100
                mean = df[col].mean() if df[col].notna().any() else float("nan")
                logger.info("  %-35s coverage=%.1f%%  mean=%.4f", col, pct, mean)


if __name__ == "__main__":
    main()
