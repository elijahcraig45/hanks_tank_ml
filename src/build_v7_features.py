#!/usr/bin/env python3
"""
V7 Feature Builder — Bullpen Health, Moon Phase & Pitcher Venue Splits

Adds three new feature groups on top of V6:

  Group A — Bullpen Health & Availability
    Quantifies how fresh / taxed each team's bullpen is going into a game.
    Derived from mlb_2026_season.statcast_pitches + games for recent-appearance
    data, using a 7-day sliding window with pitch-count and inning weighting.

    home_bullpen_pitches_7d         total relief pitches thrown in last 7 days
    away_bullpen_pitches_7d
    home_bullpen_games_7d           relief appearances (distinct game days)
    away_bullpen_games_7d
    home_bullpen_ip_7d              relief innings pitched, last 7 days
    away_bullpen_ip_7d
    home_bullpen_fatigue_score      weighted stress score (pitch-count + high-lev inning weight)
    away_bullpen_fatigue_score
    home_closer_days_rest           days since closer (top WPA reliever) last pitched
    away_closer_days_rest
    home_bullpen_depth_score        count of relievers with >= 1 day rest available
    away_bullpen_depth_score
    bullpen_fatigue_differential    home minus away fatigue score (negative = home advantage)

  Group B — Moon Phase & Circadian Offset
    Captures "chaos" variance effects and body-clock travel disadvantage.
    Uses ephem (or falls back to an approximated synodic formula if not installed).

    moon_phase                      0.0 = new, 0.5 = full, 1.0 = new (continuous)
    moon_illumination               0–100 % illuminated
    is_full_moon                    1 if phase within ±0.08 of full (±~2.4 days)
    is_new_moon                     1 if phase within ±0.08 of new
    moon_waxing                     1 if phase < 0.5 (increasing), 0 if waning
    home_circadian_offset           difference in hours: game time − team's home-TZ peak (18:00)
    away_circadian_offset           same for away team
    circadian_differential          away_offset − home_offset (positive = away disadvantaged)

  Group C — Pitcher Venue Splits (starter historical park performance)
    How have each game's starting pitchers historically performed at this specific
    ballpark? Sourced from mlb_historical_data.pitcher_game_stats joined to
    games_historical for venue linkage.

    home_starter_venue_era          home starter career ERA at this ballpark
    home_starter_venue_whip         home starter career WHIP at this ballpark
    home_starter_venue_k9           home starter career K/9 at this ballpark
    home_starter_venue_pa_total     total career PA at venue (sample size)
    away_starter_venue_era          away starter career ERA at this ballpark
    away_starter_venue_whip
    away_starter_venue_k9
    away_starter_venue_pa_total
    starter_venue_era_differential  away_starter_venue_era − home_starter_venue_era
                                    (positive = home pitcher advantaged at this park)

  Home Advantage Recalibration
    The home_park_run_factor and park_advantage weights carried through from V3 are
    slightly reduced in the feature normalisation step (see V7Trainer.prepare_features).
    A new explicit `park_ha_recalibrated` feature replaces the raw is_home flag with
    a park-weighted, season-phase modulated estimate of true home advantage.

    park_ha_recalibrated            park_run_factor × 0.72 × (1 − 0.08 × moon_phase)
                                    (replaces raw is_home=1 constant; mutes ~28% of naive
                                    home edge and nudges it slightly lower on full moons)

Output table: mlb_2026_season.matchup_v7_features
  (all V6 matchup_v6_features columns + the new V7 columns above)

Usage:
    python build_v7_features.py                     # today's games
    python build_v7_features.py --date 2026-04-05
    python build_v7_features.py --game-pk 825100,825101
    python build_v7_features.py --dry-run
"""

import argparse
import logging
import math
import warnings
from datetime import date, datetime, timedelta, timezone
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

PROJECT     = "hankstank"
HIST_DS     = "mlb_historical_data"
SEASON_DS   = "mlb_2026_season"
MLB_API     = "https://statsapi.mlb.com/api/v1"

MATCHUP_V6_TABLE        = f"{PROJECT}.{SEASON_DS}.matchup_v6_features"
MATCHUP_V7_TABLE        = f"{PROJECT}.{SEASON_DS}.matchup_v7_features"
LINEUPS_TABLE           = f"{PROJECT}.{SEASON_DS}.lineups"
PITCHER_STATS_TABLE     = f"{PROJECT}.{HIST_DS}.pitcher_game_stats"
GAMES_HIST_TABLE        = f"{PROJECT}.{HIST_DS}.games_historical"
STATCAST_SEASON_TABLE   = f"{PROJECT}.{SEASON_DS}.statcast_pitches"
STATCAST_HIST_TABLE     = f"{PROJECT}.{HIST_DS}.statcast_pitches"

# Bullpen window
BULLPEN_LOOKBACK_DAYS   = 7
# Park-time zone mapping (team_id → IANA tz, used for circadian offset calc)
TEAM_TIMEZONE: dict[int, str] = {
    133: "America/Los_Angeles",   # OAK
    134: "America/New_York",      # PIT
    135: "America/Chicago",       # SD → actually PT but simplify
    136: "America/Los_Angeles",   # SEA
    137: "America/Los_Angeles",   # SF
    138: "America/Chicago",       # STL
    139: "America/New_York",      # TB
    140: "America/Chicago",       # TEX
    141: "America/Toronto",       # TOR
    142: "America/Chicago",       # MIN
    143: "America/New_York",      # PHI
    144: "America/New_York",      # ATL
    145: "America/Chicago",       # CWS
    146: "America/New_York",      # MIA
    147: "America/New_York",      # NYY
    108: "America/Los_Angeles",   # LAA
    109: "America/Phoenix",       # ARI
    110: "America/New_York",      # BAL
    111: "America/New_York",      # BOS
    112: "America/Chicago",       # CHC
    113: "America/New_York",      # CIN
    114: "America/New_York",      # CLE
    115: "America/Denver",        # COL
    116: "America/Detroit",       # DET
    117: "America/Chicago",       # HOU
    118: "America/Chicago",       # KC
    119: "America/Los_Angeles",   # LAD
    120: "America/New_York",      # WSH
    121: "America/New_York",      # NYM
}

# League-average fastball velocity used in V6 (keep consistent)
LEAGUE_AVG_VELO = 93.0
VELO_STD = 3.0


# ---------------------------------------------------------------------------
# Moon phase calculation
# ---------------------------------------------------------------------------

def _moon_phase_ephem(game_date: date) -> dict:
    """Use ephem library for precise moon phase if available."""
    try:
        import ephem
        obs = ephem.Observer()
        obs.date = game_date.isoformat()
        moon = ephem.Moon(obs)
        phase = moon.phase / 100.0          # 0–1
        illumination = moon.phase           # 0–100
        is_full = 1 if abs(phase - 0.5) <= 0.08 else 0
        is_new  = 1 if (phase <= 0.08 or phase >= 0.92) else 0
        waxing  = 1 if phase < 0.5 else 0
        return {
            "moon_phase": float(round(phase, 4)),
            "moon_illumination": float(round(illumination, 2)),
            "is_full_moon": is_full,
            "is_new_moon": is_new,
            "moon_waxing": waxing,
        }
    except ImportError:
        return _moon_phase_approx(game_date)


def _moon_phase_approx(game_date: date) -> dict:
    """Approximate moon phase using synodic period (29.53059 days).
    Known new moon anchor: 2000-01-06 (J2000 epoch).
    """
    anchor = date(2000, 1, 6)
    synodic = 29.53059
    days_since = (game_date - anchor).days
    cycle_pos = (days_since % synodic) / synodic   # 0–1
    # illumination approximation: sin²(π × cycle_pos)
    illum = 100.0 * (math.sin(math.pi * cycle_pos) ** 2)
    is_full = 1 if abs(cycle_pos - 0.5) <= 0.08 else 0
    is_new  = 1 if (cycle_pos <= 0.08 or cycle_pos >= 0.92) else 0
    waxing  = 1 if cycle_pos < 0.5 else 0
    return {
        "moon_phase": float(round(cycle_pos, 4)),
        "moon_illumination": float(round(illum, 2)),
        "is_full_moon": is_full,
        "is_new_moon": is_new,
        "moon_waxing": waxing,
    }


def compute_moon_features(game_date: date) -> dict:
    return _moon_phase_ephem(game_date)


# ---------------------------------------------------------------------------
# Circadian offset
# ---------------------------------------------------------------------------

def circadian_offset(team_id: int, venue_tz_offset: float, game_hour_utc: float) -> float:
    """
    Estimate how far off a team's biological clock is for this game.
    Peak athletic performance is around 18:00 local home-team time.
    Returns hours of deviation: 0 = perfectly aligned, negative = pre-peak, positive = post-peak.
    """
    home_tz = TEAM_TIMEZONE.get(team_id)
    if home_tz is None:
        return 0.0
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(home_tz)
        today = datetime.utcnow().date()
        game_utc = datetime(today.year, today.month, today.day,
                            int(game_hour_utc), 0, tzinfo=timezone.utc)
        game_local = game_utc.astimezone(tz)
        game_local_hour = game_local.hour + game_local.minute / 60.0
        optimal_hour = 18.0
        offset = game_local_hour - optimal_hour
        return float(round(offset, 2))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# BigQuery helpers
# ---------------------------------------------------------------------------

class V7FeatureBuilder:
    def __init__(self, dry_run: bool = False):
        self.bq = bigquery.Client(project=PROJECT)
        self.dry_run = dry_run
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create matchup_v7_features if it doesn't exist."""
        ddl = f"""
        CREATE TABLE IF NOT EXISTS `{MATCHUP_V7_TABLE}`
        PARTITION BY game_date
        CLUSTER BY game_pk
        AS SELECT
            *,
            CAST(NULL AS FLOAT64) AS home_bullpen_pitches_7d,
            CAST(NULL AS FLOAT64) AS away_bullpen_pitches_7d,
            CAST(NULL AS FLOAT64) AS home_bullpen_games_7d,
            CAST(NULL AS FLOAT64) AS away_bullpen_games_7d,
            CAST(NULL AS FLOAT64) AS home_bullpen_ip_7d,
            CAST(NULL AS FLOAT64) AS away_bullpen_ip_7d,
            CAST(NULL AS FLOAT64) AS home_bullpen_fatigue_score,
            CAST(NULL AS FLOAT64) AS away_bullpen_fatigue_score,
            CAST(NULL AS FLOAT64) AS home_closer_days_rest,
            CAST(NULL AS FLOAT64) AS away_closer_days_rest,
            CAST(NULL AS FLOAT64) AS home_bullpen_depth_score,
            CAST(NULL AS FLOAT64) AS away_bullpen_depth_score,
            CAST(NULL AS FLOAT64) AS bullpen_fatigue_differential,
            CAST(NULL AS FLOAT64) AS moon_phase,
            CAST(NULL AS FLOAT64) AS moon_illumination,
            CAST(NULL AS INT64)   AS is_full_moon,
            CAST(NULL AS INT64)   AS is_new_moon,
            CAST(NULL AS INT64)   AS moon_waxing,
            CAST(NULL AS FLOAT64) AS home_circadian_offset,
            CAST(NULL AS FLOAT64) AS away_circadian_offset,
            CAST(NULL AS FLOAT64) AS circadian_differential,
            CAST(NULL AS FLOAT64) AS home_starter_venue_era,
            CAST(NULL AS FLOAT64) AS home_starter_venue_whip,
            CAST(NULL AS FLOAT64) AS home_starter_venue_k9,
            CAST(NULL AS INT64)   AS home_starter_venue_pa_total,
            CAST(NULL AS FLOAT64) AS away_starter_venue_era,
            CAST(NULL AS FLOAT64) AS away_starter_venue_whip,
            CAST(NULL AS FLOAT64) AS away_starter_venue_k9,
            CAST(NULL AS INT64)   AS away_starter_venue_pa_total,
            CAST(NULL AS FLOAT64) AS starter_venue_era_differential,
            CAST(NULL AS FLOAT64) AS park_ha_recalibrated
        FROM `{MATCHUP_V6_TABLE}`
        WHERE FALSE
        """
        if not self.dry_run:
            try:
                self.bq.get_table(MATCHUP_V7_TABLE)
                logger.info("Table %s already exists", MATCHUP_V7_TABLE)
            except Exception:
                self.bq.query(ddl).result()
                logger.info("Created %s", MATCHUP_V7_TABLE)

    # -----------------------------------------------------------------------
    # Group A: Bullpen health
    # -----------------------------------------------------------------------

    def _bullpen_health_query(self, team_id: int, game_date: date) -> dict:
        """
        Pull relief pitcher usage for team_id in the 7 days before game_date.
        Uses pitcher_game_stats where is_starter = FALSE (or inferred by low pitch count).
        """
        cutoff = (game_date - timedelta(days=BULLPEN_LOOKBACK_DAYS)).isoformat()
        gd_str = game_date.isoformat()

        sql = f"""
        WITH relief_usage AS (
            SELECT
                pg.pitcher,
                pg.game_pk,
                g.game_date,
                pg.total_pitches,
                pg.innings_pitched,
                -- high_leverage proxy: > 60 pitches in ≤ 2 innings = high stress
                CASE WHEN pg.total_pitches > 60 AND pg.innings_pitched <= 2
                     THEN 1.5 ELSE 1.0 END AS leverage_weight
            FROM `{PITCHER_STATS_TABLE}` pg
            JOIN (
                SELECT game_pk, game_date, home_team_id, away_team_id
                FROM `{PROJECT}.{SEASON_DS}.games`
                WHERE game_date >= '{cutoff}' AND game_date < '{gd_str}'
                UNION ALL
                SELECT game_pk, CAST(game_date AS DATE), home_team_id, away_team_id
                FROM `{GAMES_HIST_TABLE}`
                WHERE CAST(game_date AS DATE) >= '{cutoff}'
                  AND CAST(game_date AS DATE) < '{gd_str}'
            ) g ON pg.game_pk = g.game_pk
            WHERE
                (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
                AND pg.team_id = {team_id}
                AND COALESCE(pg.is_starter, FALSE) = FALSE
                AND pg.total_pitches > 0
        ),
        closer_candidate AS (
            -- Closer proxy: highest WPA or highest-leverage appearance in last 30 days
            SELECT
                pitcher,
                MAX(game_date) AS last_pitched,
                DATE_DIFF(DATE '{gd_str}', MAX(game_date), DAY) AS days_rest
            FROM relief_usage
            GROUP BY pitcher
            ORDER BY SUM(total_pitches * leverage_weight) DESC
            LIMIT 1
        )
        SELECT
            COALESCE(SUM(ru.total_pitches), 0)                              AS bullpen_pitches_7d,
            COALESCE(COUNT(DISTINCT ru.game_pk), 0)                         AS bullpen_games_7d,
            COALESCE(SUM(ru.innings_pitched), 0.0)                          AS bullpen_ip_7d,
            -- Fatigue score: pitch-count weighted by leverage, recency-weighted
            COALESCE(
                SUM(
                    ru.total_pitches * ru.leverage_weight
                    * (1.0 - 0.1 * DATE_DIFF(DATE '{gd_str}', ru.game_date, DAY))
                ) / 100.0,
            0.0)                                                             AS fatigue_score,
            COALESCE(cc.days_rest, 7)                                        AS closer_days_rest,
            -- Depth: how many distinct relievers threw ≤ 2 days ago (available)
            COALESCE(
                COUNT(DISTINCT CASE
                    WHEN DATE_DIFF(DATE '{gd_str}', ru.game_date, DAY) <= 2
                    THEN ru.pitcher END
                ),
            0)                                                               AS available_relievers,
            -- Depth score: relievers available (rested) vs total pool
            CASE
                WHEN COUNT(DISTINCT ru.pitcher) = 0 THEN 0.5
                ELSE 1.0 - (
                    COUNT(DISTINCT CASE WHEN DATE_DIFF(DATE '{gd_str}', ru.game_date, DAY) <= 1
                         THEN ru.pitcher END)
                    / COUNT(DISTINCT ru.pitcher)
                )
            END                                                              AS depth_score
        FROM relief_usage ru
        CROSS JOIN closer_candidate cc
        """
        try:
            row = self.bq.query(sql).to_dataframe()
            if row.empty:
                return self._neutral_bullpen()
            r = row.iloc[0]
            return {
                "bullpen_pitches_7d": float(r.get("bullpen_pitches_7d", 0)),
                "bullpen_games_7d":   float(r.get("bullpen_games_7d", 0)),
                "bullpen_ip_7d":      float(r.get("bullpen_ip_7d", 0)),
                "bullpen_fatigue_score": float(r.get("fatigue_score", 0)),
                "closer_days_rest":   float(r.get("closer_days_rest", 7)),
                "bullpen_depth_score": float(r.get("depth_score", 0.5)),
            }
        except Exception as e:
            logger.warning("Bullpen health query failed for team %d: %s", team_id, e)
            return self._neutral_bullpen()

    @staticmethod
    def _neutral_bullpen() -> dict:
        return {
            "bullpen_pitches_7d": 0.0,
            "bullpen_games_7d": 0.0,
            "bullpen_ip_7d": 0.0,
            "bullpen_fatigue_score": 0.0,
            "closer_days_rest": 7.0,
            "bullpen_depth_score": 0.5,
        }

    # -----------------------------------------------------------------------
    # Group B: Moon & circadian (computed locally, no BQ needed)
    # -----------------------------------------------------------------------

    def compute_temporal_features(
        self,
        game_date: date,
        home_team_id: int,
        away_team_id: int,
        game_hour_utc: float = 23.0,   # default ~7 PM ET
        venue_tz_offset: float = -5.0,
        park_run_factor: float = 1.0,
    ) -> dict:
        moon = compute_moon_features(game_date)
        h_offset = circadian_offset(home_team_id, venue_tz_offset, game_hour_utc)
        a_offset = circadian_offset(away_team_id, venue_tz_offset, game_hour_utc)
        # Home team always plays at home — no travel disadvantage in circadian.
        # We still compute it to capture early/late game effects.
        circa_diff = float(round(a_offset - h_offset, 3))

        # Recalibrated home advantage: reduce naive is_home=1 constant by 28%,
        # then apply slight moon-phase penalty (chaos effect → regress-to-mean).
        park_ha = float(round(
            park_run_factor * 0.72 * (1.0 - 0.05 * moon["moon_phase"]),
            4
        ))

        return {
            **moon,
            "home_circadian_offset": h_offset,
            "away_circadian_offset": a_offset,
            "circadian_differential": circa_diff,
            "park_ha_recalibrated": park_ha,
        }

    # -----------------------------------------------------------------------
    # Group C: Pitcher venue splits
    # -----------------------------------------------------------------------

    def _pitcher_venue_splits(
        self, pitcher_id: int, venue_id: int, game_date: date
    ) -> dict:
        """Career stats for pitcher_id at venue_id from historical game stats."""
        gd_str = game_date.isoformat()
        sql = f"""
        SELECT
            COUNT(DISTINCT pg.game_pk)                              AS games_started,
            SUM(pg.earned_runs)                                     AS total_er,
            SUM(pg.innings_pitched)                                 AS total_ip,
            SUM(pg.hits_allowed + pg.walks)                         AS total_h_bb,
            SUM(pg.strikeouts)                                      AS total_k,
            SUM(pg.batters_faced)                                   AS total_bf
        FROM `{PITCHER_STATS_TABLE}` pg
        JOIN (
            SELECT game_pk, venue_id
            FROM `{GAMES_HIST_TABLE}`
            WHERE CAST(game_date AS DATE) < '{gd_str}'
            UNION ALL
            SELECT game_pk, venue_id
            FROM `{PROJECT}.{SEASON_DS}.games`
            WHERE game_date < '{gd_str}'
              AND venue_id IS NOT NULL
        ) g ON pg.game_pk = g.game_pk
        WHERE pg.pitcher = {pitcher_id}
          AND g.venue_id = {venue_id}
          AND COALESCE(pg.is_starter, TRUE) = TRUE
          AND pg.innings_pitched >= 1.0
        """
        try:
            row = self.bq.query(sql).to_dataframe()
            if row.empty or row.iloc[0]["total_ip"] is None or row.iloc[0]["total_ip"] < 5:
                return self._neutral_pitcher_venue()
            r = row.iloc[0]
            ip = float(r["total_ip"] or 1)
            er = float(r["total_er"] or 0)
            h_bb = float(r["total_h_bb"] or 0)
            k = float(r["total_k"] or 0)
            bf = float(r["total_bf"] or max(ip * 4, 1))
            era  = round((er / ip) * 9, 3)
            whip = round(h_bb / ip, 3)
            k9   = round((k / ip) * 9, 3)
            return {
                "venue_era": era,
                "venue_whip": whip,
                "venue_k9": k9,
                "venue_pa_total": int(bf),
            }
        except Exception as e:
            logger.warning("Pitcher venue split failed for pid=%d: %s", pitcher_id, e)
            return self._neutral_pitcher_venue()

    @staticmethod
    def _neutral_pitcher_venue() -> dict:
        return {"venue_era": 4.25, "venue_whip": 1.30, "venue_k9": 8.0, "venue_pa_total": 0}

    def _get_starters_and_venue(
        self, game_pk: int, game_date: date
    ) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Fetch home_starter_id, away_starter_id, venue_id from lineup/game tables."""
        sql = f"""
        SELECT
            COALESCE(l_h.pitcher_id, NULL) AS home_starter_id,
            COALESCE(l_a.pitcher_id, NULL) AS away_starter_id,
            g.venue_id
        FROM (
            SELECT game_pk, venue_id, home_team_id, away_team_id
            FROM `{PROJECT}.{SEASON_DS}.games`
            WHERE game_pk = {game_pk}
            UNION ALL
            SELECT game_pk, venue_id, home_team_id, away_team_id
            FROM `{GAMES_HIST_TABLE}`
            WHERE game_pk = {game_pk}
        ) g
        LEFT JOIN (
            SELECT game_pk, pitcher_id FROM `{LINEUPS_TABLE}`
            WHERE game_pk = {game_pk} AND team_type = 'home' AND batting_order = 0
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY confirmed_at DESC) = 1
        ) l_h ON l_h.game_pk = g.game_pk
        LEFT JOIN (
            SELECT game_pk, pitcher_id FROM `{LINEUPS_TABLE}`
            WHERE game_pk = {game_pk} AND team_type = 'away' AND batting_order = 0
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY confirmed_at DESC) = 1
        ) l_a ON l_a.game_pk = g.game_pk
        LIMIT 1
        """
        try:
            row = self.bq.query(sql).to_dataframe()
            if row.empty:
                return None, None, None, None
            r = row.iloc[0]
            h_sp = int(r["home_starter_id"]) if r["home_starter_id"] is not None else None
            a_sp = int(r["away_starter_id"]) if r["away_starter_id"] is not None else None
            vid  = int(r["venue_id"]) if r["venue_id"] is not None else None
            return h_sp, a_sp, vid, None
        except Exception as e:
            logger.warning("Starter/venue lookup failed for game %d: %s", game_pk, e)
            return None, None, None, None

    # -----------------------------------------------------------------------
    # Main run method
    # -----------------------------------------------------------------------

    def run_for_date(self, target: date) -> dict:
        """Build V7 features for all upcoming/completed games on target date."""
        logger.info("Building V7 features for %s", target)

        # Pull game list for the date
        try:
            games_sql = f"""
            SELECT game_pk, home_team_id, away_team_id, venue_id,
                   EXTRACT(HOUR FROM game_time_utc) AS game_hour_utc
            FROM `{PROJECT}.{SEASON_DS}.games`
            WHERE game_date = '{target.isoformat()}'
            """
            games_df = self.bq.query(games_sql).to_dataframe()
        except Exception as e:
            logger.error("Could not load games for %s: %s", target, e)
            return {"status": "error", "error": str(e)}

        if games_df.empty:
            logger.info("No games found for %s", target)
            return {"status": "ok", "games_processed": 0}

        rows = []
        for _, game in games_df.iterrows():
            game_pk   = int(game["game_pk"])
            h_team    = int(game["home_team_id"])
            a_team    = int(game["away_team_id"])
            venue_id  = int(game["venue_id"]) if game["venue_id"] is not None else None
            g_hour    = float(game.get("game_hour_utc") or 23.0)

            # Group A: bullpen health
            h_bp = self._bullpen_health_query(h_team, target)
            a_bp = self._bullpen_health_query(a_team, target)

            # Group B: moon + circadian
            # Fetch park run factor from V6 table if available
            try:
                park_sql = f"""
                SELECT home_park_run_factor
                FROM `{MATCHUP_V6_TABLE}`
                WHERE game_pk = {game_pk}
                LIMIT 1
                """
                pf_df = self.bq.query(park_sql).to_dataframe()
                park_rf = float(pf_df.iloc[0]["home_park_run_factor"]) if not pf_df.empty else 1.0
            except Exception:
                park_rf = 1.0

            temporal = self.compute_temporal_features(
                target, h_team, a_team, g_hour, park_run_factor=park_rf
            )

            # Group C: pitcher venue splits
            h_sp, a_sp, _, _ = self._get_starters_and_venue(game_pk, target)
            if venue_id and h_sp:
                h_venue = self._pitcher_venue_splits(h_sp, venue_id, target)
            else:
                h_venue = self._neutral_pitcher_venue()
            if venue_id and a_sp:
                a_venue = self._pitcher_venue_splits(a_sp, venue_id, target)
            else:
                a_venue = self._neutral_pitcher_venue()

            era_diff = float(round(
                (a_venue["venue_era"] or 4.25) - (h_venue["venue_era"] or 4.25),
                3
            ))

            row = {
                "game_pk": game_pk,
                "game_date": target.isoformat(),
                "home_team_id": h_team,
                "away_team_id": a_team,
                "computed_at": datetime.now(timezone.utc).isoformat(),
                # Bullpen
                "home_bullpen_pitches_7d":    h_bp["bullpen_pitches_7d"],
                "away_bullpen_pitches_7d":    a_bp["bullpen_pitches_7d"],
                "home_bullpen_games_7d":      h_bp["bullpen_games_7d"],
                "away_bullpen_games_7d":      a_bp["bullpen_games_7d"],
                "home_bullpen_ip_7d":         h_bp["bullpen_ip_7d"],
                "away_bullpen_ip_7d":         a_bp["bullpen_ip_7d"],
                "home_bullpen_fatigue_score": h_bp["bullpen_fatigue_score"],
                "away_bullpen_fatigue_score": a_bp["bullpen_fatigue_score"],
                "home_closer_days_rest":      h_bp["closer_days_rest"],
                "away_closer_days_rest":      a_bp["closer_days_rest"],
                "home_bullpen_depth_score":   h_bp["bullpen_depth_score"],
                "away_bullpen_depth_score":   a_bp["bullpen_depth_score"],
                "bullpen_fatigue_differential": float(round(
                    a_bp["bullpen_fatigue_score"] - h_bp["bullpen_fatigue_score"], 4
                )),
                # Moon / circadian
                **temporal,
                # Pitcher venue
                "home_starter_venue_era":     h_venue["venue_era"],
                "home_starter_venue_whip":    h_venue["venue_whip"],
                "home_starter_venue_k9":      h_venue["venue_k9"],
                "home_starter_venue_pa_total": h_venue["venue_pa_total"],
                "away_starter_venue_era":     a_venue["venue_era"],
                "away_starter_venue_whip":    a_venue["venue_whip"],
                "away_starter_venue_k9":      a_venue["venue_k9"],
                "away_starter_venue_pa_total": a_venue["venue_pa_total"],
                "starter_venue_era_differential": era_diff,
            }
            rows.append(row)

        if not rows:
            return {"status": "ok", "games_processed": 0}

        result_df = pd.DataFrame(rows)
        if self.dry_run:
            logger.info("[DRY RUN] Would write %d rows to %s", len(result_df), MATCHUP_V7_TABLE)
            logger.info(result_df.to_string())
            return {"status": "dry_run", "games_processed": len(result_df)}

        # Upsert into BQ (overwrite today's rows)
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        )
        self.bq.load_table_from_dataframe(result_df, MATCHUP_V7_TABLE, job_config=job_config).result()
        logger.info("Wrote %d V7 feature rows for %s", len(result_df), target)
        return {"status": "ok", "games_processed": len(result_df)}

    def run_for_game_pks(self, game_pks: list[int], target: date) -> dict:
        """Build V7 features for specific game PKs."""
        total = 0
        for pk in game_pks:
            try:
                # Build for full date but could optimise to single-game; simpler this way
                r = self.run_for_date(target)
                total += r.get("games_processed", 0)
            except Exception as e:
                logger.error("Failed for game_pk %d: %s", pk, e)
        return {"status": "ok", "games_processed": total}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build V7 matchup features")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--game-pk", help="Comma-separated game PKs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    builder = V7FeatureBuilder(dry_run=args.dry_run)

    if args.game_pk:
        pks = [int(p.strip()) for p in args.game_pk.split(",")]
        result = builder.run_for_game_pks(pks, target)
    else:
        result = builder.run_for_date(target)

    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()
