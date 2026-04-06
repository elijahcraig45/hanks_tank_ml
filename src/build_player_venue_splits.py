#!/usr/bin/env python3
"""
Build player venue split tables from Statcast data.

Computes per-batter performance at each ballpark (using historical statcast
pitch-by-pitch data joined with games_historical for venue_id).

Also computes per-batter overall wOBA (all parks, per season) for comparison,
so we can derive a venue_advantage feature = wOBA_at_park - wOBA_overall.

Sources:
  mlb_historical_data.statcast_pitches  (batter-level events)
  mlb_historical_data.games_historical  (game → venue_id mapping)

Outputs:
  mlb_historical_data.player_venue_splits   (per batter per venue, career)
  mlb_historical_data.player_season_splits  (per batter per season, all parks)

Usage:
    python build_player_venue_splits.py              # full rebuild
    python build_player_venue_splits.py --dry-run    # print SQL only
"""

import argparse
import logging

from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
HIST_DS = "mlb_historical_data"
SEASON_DS = "mlb_2026_season"
VENUE_TABLE = f"{PROJECT}.{HIST_DS}.player_venue_splits"
SEASON_TABLE = f"{PROJECT}.{HIST_DS}.player_season_splits"

# Minimum PA at venue for the split to be usable (avoid tiny sample noise)
MIN_PA_VENUE = 15
MIN_PA_SEASON = 30


VENUE_SPLITS_SQL = f"""
CREATE OR REPLACE TABLE `{VENUE_TABLE}`
CLUSTER BY player_id, venue_id
AS
WITH combined_pitches AS (
  -- Historical 2015-2025
  SELECT sc.batter, sc.stand, sc.game_pk, sc.game_year,
         sc.woba_value, sc.woba_denom,
         sc.estimated_woba_using_speedangle AS xwoba,
         sc.events, sc.launch_speed, sc.launch_angle
  FROM `{PROJECT}.{HIST_DS}.statcast_pitches` sc
  WHERE sc.woba_denom = 1 AND sc.batter IS NOT NULL
  UNION ALL
  -- 2026 season
  SELECT sc.batter, sc.stand, sc.game_pk,
         EXTRACT(YEAR FROM sc.game_date) AS game_year,
         sc.woba_value, sc.woba_denom,
         sc.estimated_woba_using_speedangle AS xwoba,
         sc.events, sc.launch_speed, sc.launch_angle
  FROM `{PROJECT}.{SEASON_DS}.statcast_pitches` sc
  WHERE sc.woba_denom = 1 AND sc.batter IS NOT NULL
),
combined_games AS (
  SELECT game_pk, venue_id, venue_name
  FROM `{PROJECT}.{HIST_DS}.games_historical`
  UNION ALL
  SELECT game_pk, venue_id, venue_name
  FROM `{PROJECT}.{SEASON_DS}.games`
  WHERE venue_id IS NOT NULL
),
pa_events AS (
  SELECT
    sc.batter                                     AS player_id,
    sc.stand                                      AS bat_side,
    g.venue_id,
    g.venue_name,
    sc.game_year,
    sc.woba_value,
    sc.woba_denom,
    sc.xwoba,
    CASE WHEN sc.events IN ('strikeout','strikeout_double_play')
         THEN 1 ELSE 0 END                        AS is_k,
    CASE WHEN sc.events IN ('walk','intent_walk')
         THEN 1 ELSE 0 END                        AS is_bb,
    CASE WHEN sc.events = 'home_run'
         THEN 1 ELSE 0 END                        AS is_hr,
    sc.launch_speed                               AS exit_velo,
    sc.launch_angle
  FROM combined_pitches sc
  JOIN combined_games g ON sc.game_pk = g.game_pk
  WHERE g.venue_id IS NOT NULL
)
SELECT
  player_id,
  venue_id,
  ANY_VALUE(venue_name)                                AS venue_name,
  COUNT(1)                                             AS pa_total,
  SUM(woba_denom)                                      AS weighted_pa,
  ROUND(SAFE_DIVIDE(SUM(woba_value), SUM(woba_denom)), 4)
                                                       AS woba,
  ROUND(AVG(xwoba), 4)                                AS xwoba,
  ROUND(SAFE_DIVIDE(SUM(is_k),  COUNT(1)), 4)         AS k_pct,
  ROUND(SAFE_DIVIDE(SUM(is_bb), COUNT(1)), 4)         AS bb_pct,
  ROUND(SAFE_DIVIDE(SUM(is_hr), COUNT(1)), 4)         AS hr_per_pa,
  ROUND(AVG(exit_velo), 1)                             AS avg_exit_velo,
  ROUND(AVG(launch_angle), 1)                         AS avg_launch_angle,
  MIN(game_year)                                       AS first_year,
  MAX(game_year)                                       AS last_year
FROM pa_events
GROUP BY player_id, venue_id
HAVING pa_total >= {MIN_PA_VENUE}
"""


SEASON_SPLITS_SQL = f"""
CREATE OR REPLACE TABLE `{SEASON_TABLE}`
PARTITION BY RANGE_BUCKET(game_year, GENERATE_ARRAY(2015, 2027, 1))
CLUSTER BY player_id
AS
WITH pa_events AS (
  SELECT
    batter                                        AS player_id,
    stand                                         AS bat_side,
    game_year,
    woba_value,
    woba_denom,
    estimated_woba_using_speedangle               AS xwoba,
    CASE WHEN events IN ('strikeout','strikeout_double_play')
         THEN 1 ELSE 0 END                        AS is_k,
    CASE WHEN events IN ('walk','intent_walk')
         THEN 1 ELSE 0 END                        AS is_bb,
    CASE WHEN events = 'home_run'
         THEN 1 ELSE 0 END                        AS is_hr
  FROM `{PROJECT}.{HIST_DS}.statcast_pitches`
  WHERE woba_denom = 1 AND batter IS NOT NULL
  UNION ALL
  SELECT
    batter                                        AS player_id,
    stand                                         AS bat_side,
    EXTRACT(YEAR FROM game_date)                  AS game_year,
    woba_value,
    woba_denom,
    estimated_woba_using_speedangle               AS xwoba,
    CASE WHEN events IN ('strikeout','strikeout_double_play')
         THEN 1 ELSE 0 END                        AS is_k,
    CASE WHEN events IN ('walk','intent_walk')
         THEN 1 ELSE 0 END                        AS is_bb,
    CASE WHEN events = 'home_run'
         THEN 1 ELSE 0 END                        AS is_hr
  FROM `{PROJECT}.{SEASON_DS}.statcast_pitches`
  WHERE woba_denom = 1 AND batter IS NOT NULL
)
SELECT
  player_id,
  game_year,
  COUNT(1)                                        AS pa_total,
  ROUND(SAFE_DIVIDE(SUM(woba_value), SUM(woba_denom)), 4)
                                                  AS woba,
  ROUND(AVG(xwoba), 4)                           AS xwoba,
  ROUND(SAFE_DIVIDE(SUM(is_k),  COUNT(1)), 4)    AS k_pct,
  ROUND(SAFE_DIVIDE(SUM(is_bb), COUNT(1)), 4)    AS bb_pct,
  ROUND(SAFE_DIVIDE(SUM(is_hr), COUNT(1)), 4)    AS hr_per_pa
FROM pa_events
GROUP BY player_id, game_year
HAVING pa_total >= {MIN_PA_SEASON}
"""


def run(dry_run: bool = False) -> None:
    bq = bigquery.Client(project=PROJECT)

    for label, sql in [
        ("player_venue_splits",  VENUE_SPLITS_SQL),
        ("player_season_splits", SEASON_SPLITS_SQL),
    ]:
        if dry_run:
            logger.info("[DRY RUN] %s SQL (first 1500 chars):\n%s", label, sql[:1500])
            continue

        logger.info("Building %s...", label)
        job = bq.query(sql)
        job.result()
        logger.info("%s build complete.", label)

    if not dry_run:
        for tbl in [VENUE_TABLE, SEASON_TABLE]:
            cnt = next(bq.query(f"SELECT COUNT(1) AS n FROM `{tbl}`").result())
            logger.info("%s: %d rows", tbl, cnt["n"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.dry_run)
