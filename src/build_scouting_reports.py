#!/usr/bin/env python3
"""
Daily Scouting Report Builder
==============================
Runs once per day after the pregame pipeline completes.
For every game on the target date, assembles a structured scouting report from:
  - game_predictions        (win probabilities, model signals, pitcher names)
  - game_v8_features        (Elo, Pythagorean, streaks, H2H)
  - matchup_v7_features     (probable pitchers, arsenal, bullpen, venue)
  - mlb_historical_data     (player hot/cold streaks from last 14 days)
  - news_articles           (relevant team/player headlines from last 7 days)

Output: one row per game_pk in mlb_2026_season.game_scouting_reports
The JSON report is pre-computed so the frontend never does heavy BQ work at
page-load time — it just fetches a single row.

Usage:
    python3 build_scouting_reports.py [--date YYYY-MM-DD] [--dry-run]
"""

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
DATASET = "mlb_2026_season"
HIST_DS = "mlb_historical_data"
REPORTS_TABLE = f"{PROJECT}.{DATASET}.game_scouting_reports"


def _q(bq: bigquery.Client, sql: str) -> list[dict]:
    rows = list(bq.query(sql).result())
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def fetch_games_on_date(bq: bigquery.Client, game_date: date) -> list[dict]:
    # Try games table first; fall back to game_predictions if empty (e.g. same-day before nightly sync)
    sql = f"""
    SELECT game_pk, game_date,
           home_team_id, away_team_id,
           home_team_name, away_team_name,
           home_score, away_score, status,
           venue_name
    FROM `{PROJECT}.{DATASET}.games`
    WHERE game_date = '{game_date}'
    ORDER BY game_pk
    """
    rows = _q(bq, sql)
    if rows:
        return rows

    logger.info("games table empty for %s — falling back to game_predictions", game_date)
    sql2 = f"""
    SELECT DISTINCT
        game_pk, game_date,
        home_team_id, away_team_id,
        home_team_name, away_team_name,
        NULL AS home_score, NULL AS away_score,
        'Preview' AS status,
        NULL AS venue_name
    FROM `{PROJECT}.{DATASET}.game_predictions`
    WHERE game_date = '{game_date}'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY predicted_at DESC) = 1
    ORDER BY game_pk
    """
    return _q(bq, sql2)


def fetch_predictions(bq: bigquery.Client, game_date: date) -> dict[int, dict]:
    sql = f"""
    SELECT *
    FROM `{PROJECT}.{DATASET}.game_predictions`
    WHERE game_date = '{game_date}'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY predicted_at DESC) = 1
    """
    rows = _q(bq, sql)
    return {int(r["game_pk"]): r for r in rows}


def fetch_v8_features(bq: bigquery.Client, game_date: date) -> dict[int, dict]:
    sql = f"""
    SELECT *
    FROM `{PROJECT}.{DATASET}.game_v8_features`
    WHERE game_date = '{game_date}'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) = 1
    """
    rows = _q(bq, sql)
    return {int(r["game_pk"]): r for r in rows}


def fetch_matchup_features(bq: bigquery.Client, game_date: date) -> dict[int, dict]:
    sql = f"""
    SELECT *
    FROM `{PROJECT}.{DATASET}.matchup_v7_features`
    WHERE game_date = '{game_date}'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) = 1
    """
    rows = _q(bq, sql)
    return {int(r["game_pk"]): r for r in rows}


def fetch_hot_cold_players(bq: bigquery.Client, team_ids: list[int], as_of: date) -> dict[int, list[dict]]:
    """
    For each team, find batters hot or cold over the last 14 days.
    Returns {team_id: [player_entry, ...]} — players sorted by wOBA delta.
    Uses player_game_stats from the historical dataset.
    """
    if not team_ids:
        return {}

    cutoff = (as_of - timedelta(days=14)).isoformat()
    ids_str = ", ".join(str(t) for t in team_ids)

    sql = f"""
    WITH recent AS (
        SELECT
            player_id, player_name, team_id,
            SUM(at_bats)  AS ab,
            SUM(hits)     AS h,
            SUM(home_runs) AS hr,
            SUM(rbi)      AS rbi,
            SUM(walks + IFNULL(hit_by_pitch, 0)) AS bb,
            SUM(strikeouts) AS so,
            -- wOBA numerator approximation (without full linear weights)
            ROUND(
                (0.69 * SUM(walks) + 0.72 * SUM(IFNULL(hit_by_pitch,0))
                 + 0.89 * (SUM(hits) - SUM(IFNULL(doubles,0)) - SUM(IFNULL(triples,0)) - SUM(home_runs))
                 + 1.27 * SUM(IFNULL(doubles,0))
                 + 1.62 * SUM(IFNULL(triples,0))
                 + 2.10 * SUM(home_runs))
                / NULLIF(SUM(at_bats + walks + IFNULL(hit_by_pitch,0)), 0),
            3) AS woba_14d,
            COUNT(DISTINCT game_date) AS games_played
        FROM `{PROJECT}.{HIST_DS}.player_game_stats`
        WHERE team_id IN ({ids_str})
          AND game_date BETWEEN '{cutoff}' AND '{as_of}'
          AND at_bats > 0
        GROUP BY player_id, player_name, team_id
        HAVING games_played >= 4 AND ab >= 10
    )
    SELECT *,
        woba_14d - 0.320 AS woba_delta
    FROM recent
    ORDER BY team_id, woba_delta DESC
    """
    try:
        rows = _q(bq, sql)
    except Exception as e:
        logger.warning("hot/cold query failed (table may be empty early season): %s", e)
        return {}

    result: dict[int, list] = {}
    for r in rows:
        tid = int(r["team_id"])
        result.setdefault(tid, []).append(r)
    return result


def fetch_team_news(bq: bigquery.Client, team_names: list[str], since_days: int = 7) -> dict[str, list[dict]]:
    """
    Fetch recent news articles from news_articles that mention any of the team names.
    Returns {search_term: [articles]} grouped by team name keyword.
    """
    cutoff = (date.today() - timedelta(days=since_days)).isoformat()
    # Build a LIKE clause for each team name keyword
    keywords = []
    for name in team_names:
        # Use last word of team name (e.g. "Atlanta Braves" → "Braves")
        keyword = name.split()[-1]
        keywords.append(keyword)

    # Deduplicate
    keywords = list(dict.fromkeys(keywords))

    conditions = " OR ".join(
        f"LOWER(title) LIKE '%{k.lower()}%' OR LOWER(description) LIKE '%{k.lower()}%'"
        for k in keywords
    )

    sql = f"""
    SELECT title, description, url, published_at, source_name, news_type
    FROM `{PROJECT}.{DATASET}.news_articles`
    WHERE fetched_at >= '{cutoff}'
      AND ({conditions})
    ORDER BY published_at DESC
    LIMIT 40
    """
    try:
        rows = _q(bq, sql)
    except Exception as e:
        logger.warning("news query failed: %s", e)
        return {}

    # Group articles by which keyword they match
    grouped: dict[str, list] = {k: [] for k in keywords}
    for r in rows:
        title_desc = ((r.get("title") or "") + " " + (r.get("description") or "")).lower()
        for k in keywords:
            if k.lower() in title_desc:
                if len(grouped[k]) < 4:  # cap at 4 per team
                    grouped[k].append({
                        "title": r["title"],
                        "description": r.get("description"),
                        "url": r["url"],
                        "published_at": str(r["published_at"]) if r.get("published_at") else None,
                        "source": r.get("source_name"),
                    })
    return grouped


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _safe_float(v, digits: int = 3):
    try:
        return round(float(v), digits) if v is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _streak_label(val) -> str | None:
    if val is None:
        return None
    v = int(val)
    return f"W{v}" if v > 0 else f"L{abs(v)}"


def build_hot_cold_section(players: list[dict]) -> dict:
    """Separate player list into hot (top 3) and cold (bottom 3)."""
    sorted_p = sorted(players, key=lambda p: p.get("woba_delta", 0) or 0, reverse=True)
    hot  = []
    cold = []
    for p in sorted_p[:3]:
        if (p.get("woba_delta") or 0) >= 0.015:
            hot.append({
                "name":       p["player_name"],
                "woba_14d":   _safe_float(p.get("woba_14d")),
                "woba_delta": _safe_float(p.get("woba_delta")),
                "ab":         _safe_int(p.get("ab")),
                "hr":         _safe_int(p.get("hr")),
                "rbi":        _safe_int(p.get("rbi")),
                "games":      _safe_int(p.get("games_played")),
            })
    for p in sorted_p[-3:]:
        if (p.get("woba_delta") or 0) <= -0.015:
            cold.append({
                "name":       p["player_name"],
                "woba_14d":   _safe_float(p.get("woba_14d")),
                "woba_delta": _safe_float(p.get("woba_delta")),
                "ab":         _safe_int(p.get("ab")),
                "games":      _safe_int(p.get("games_played")),
            })
    return {"hot": hot, "cold": cold}


def fetch_team_abbrevs(bq: bigquery.Client, team_ids: list[int]) -> dict[int, str]:
    """Return {team_id: team_code} for each team — used to map to statcast abbreviations."""
    if not team_ids:
        return {}
    ids_str = ", ".join(str(t) for t in team_ids)
    sql = f"SELECT team_id, team_code FROM `{PROJECT}.{DATASET}.teams` WHERE team_id IN ({ids_str})"
    return {int(r["team_id"]): r["team_code"] for r in _q(bq, sql)}


def fetch_batter_vs_team_matchups(
    bq: bigquery.Client,
    home_abbr: str,
    away_abbr: str,
    home_tid: int,
    away_tid: int,
    game_date: date,
) -> dict[str, list[dict]]:
    """
    Query 2026 + 2024-2025 statcast for current roster players batting against
    the opposing team's pitchers.  Returns {"home": [...], "away": [...]}.
    Only players on current rosters, min 8 PA, sorted notable-first (wOBA desc).
    """
    gd = game_date.isoformat()
    sql = f"""
    WITH recent_pa AS (
        SELECT batter, player_name,
               IF(inning_topbot = 'Top', away_team, home_team) AS batting_team,
               IF(inning_topbot = 'Top', home_team, away_team) AS pitching_team,
               woba_value, events, game_date
        FROM `{PROJECT}.{DATASET}.statcast_pitches`
        WHERE game_date < '{gd}'
          AND game_year = 2026
          AND woba_denom = 1
          AND (  (home_team = '{home_abbr}' AND away_team = '{away_abbr}')
              OR (home_team = '{away_abbr}' AND away_team = '{home_abbr}'))
        UNION ALL
        SELECT batter, player_name,
               IF(inning_topbot = 'Top', away_team, home_team) AS batting_team,
               IF(inning_topbot = 'Top', home_team, away_team) AS pitching_team,
               woba_value, events, game_date
        FROM `{PROJECT}.{HIST_DS}.statcast_pitches`
        WHERE game_date >= '2024-03-01'
          AND woba_denom = 1
          AND (  (home_team = '{home_abbr}' AND away_team = '{away_abbr}')
              OR (home_team = '{away_abbr}' AND away_team = '{home_abbr}'))
    ),
    roster AS (
        SELECT player_id, team_id
        FROM (
            SELECT player_id, team_id,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC) AS rn
            FROM `{PROJECT}.{DATASET}.rosters`
            WHERE team_id IN ({home_tid}, {away_tid})
              AND position_type IN ('Hitter', 'Two-Way Player')
        ) WHERE rn = 1
    ),
    stats AS (
        SELECT p.batter,
               MAX(p.player_name)  AS player_name,
               r.team_id,
               p.batting_team,
               p.pitching_team,
               COUNT(*)            AS pa,
               ROUND(SUM(p.woba_value) / NULLIF(COUNT(*), 0), 3) AS woba,
               COUNTIF(p.events = 'home_run')                    AS hr,
               COUNTIF(p.events IN ('single','double','triple','home_run')) AS hits,
               COUNTIF(p.game_date >= DATE_SUB('{gd}', INTERVAL 365 DAY)) AS pa_recent_yr
        FROM recent_pa p
        JOIN roster r ON r.player_id = p.batter
        GROUP BY p.batter, r.team_id, p.batting_team, p.pitching_team
        HAVING COUNT(*) >= 8
    )
    SELECT * FROM stats
    ORDER BY team_id, woba DESC
    LIMIT 60
    """
    try:
        rows = _q(bq, sql)
    except Exception as e:
        logger.warning("batter vs team matchup query failed: %s", e)
        return {"home": [], "away": []}

    home_list, away_list = [], []
    for r in rows:
        entry = {
            "player_name": r["player_name"],
            "pa":          int(r["pa"]),
            "woba":        float(r["woba"]) if r["woba"] is not None else None,
            "hr":          int(r["hr"]),
            "hits":        int(r["hits"]),
        }
        if int(r["team_id"]) == home_tid:
            home_list.append(entry)
        else:
            away_list.append(entry)
    return {"home": home_list, "away": away_list}


def fetch_yearly_h2h_records(
    bq: bigquery.Client,
    home_tid: int,
    away_tid: int,
) -> list[dict]:
    """
    Return year-by-year H2H records between two teams (2022–2026).
    Returns [{'year': int, 'home_wins': int, 'away_wins': int, 'total_games': int}]
    where home/away refer to the FIXED teams (not game site).
    """
    sql = f"""
    WITH results AS (
        SELECT
            EXTRACT(YEAR FROM game_date) AS year,
            home_team_id,
            away_team_id,
            home_score,
            away_score
        FROM `{PROJECT}.{DATASET}.games`
        WHERE (
            (home_team_id = {home_tid} AND away_team_id = {away_tid})
            OR (home_team_id = {away_tid} AND away_team_id = {home_tid})
        )
          AND EXTRACT(YEAR FROM game_date) >= 2022
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
    )
    SELECT
        year,
        COUNTIF(
            (home_team_id = {home_tid} AND home_score > away_score)
            OR (away_team_id = {home_tid} AND away_score > home_score)
        ) AS team1_wins,
        COUNTIF(
            (home_team_id = {away_tid} AND home_score > away_score)
            OR (away_team_id = {away_tid} AND away_score > home_score)
        ) AS team2_wins,
        COUNT(*) AS total_games
    FROM results
    GROUP BY year
    ORDER BY year DESC
    """
    try:
        rows = _q(bq, sql)
        return [{"year": int(r["year"]), "team1_wins": int(r["team1_wins"]),
                 "team2_wins": int(r["team2_wins"]), "total_games": int(r["total_games"])} for r in rows]
    except Exception as e:
        logger.warning("yearly H2H query failed: %s", e)
        return []


def fetch_batter_vs_pitcher(
    bq: bigquery.Client,
    home_pitcher_id: int | None,
    away_pitcher_id: int | None,
    home_tid: int,
    away_tid: int,
    game_date: date,
) -> dict[str, list[dict]]:
    """
    Return career batter vs specific pitcher matchup data.
    home_pitcher_id = the home team's SP (away batters face him)
    away_pitcher_id = the away team's SP (home batters face him)
    Returns {"home": [...], "away": [...]}  (home = home batters vs away SP)
    """
    pitcher_ids = [p for p in [home_pitcher_id, away_pitcher_id] if p is not None]
    if not pitcher_ids:
        return {"home": [], "away": []}

    roster_sql = f"""
        SELECT player_id, team_id
        FROM (
            SELECT player_id, team_id,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC) AS rn
            FROM `{PROJECT}.{DATASET}.rosters`
            WHERE team_id IN ({home_tid}, {away_tid})
              AND position_type IN ('Hitter', 'Two-Way Player')
        ) WHERE rn = 1
    """

    results = {"home": [], "away": []}
    for pitcher_id, batting_tid, key in [
        (away_pitcher_id, home_tid, "home"),
        (home_pitcher_id, away_tid, "away"),
    ]:
        if pitcher_id is None:
            continue
        gd = game_date.isoformat()
        sql = f"""
        WITH roster AS (
            {roster_sql}
        ),
        pa AS (
            SELECT batter, MAX(player_name) AS player_name,
                   COUNT(*) AS pa,
                   ROUND(SUM(woba_value) / NULLIF(COUNT(*), 0), 3) AS woba,
                   COUNTIF(events = 'home_run') AS hr,
                   COUNTIF(events IN ('single','double','triple','home_run')) AS hits,
                   COUNTIF(events = 'strikeout') AS k
            FROM (
                SELECT batter, player_name, pitcher, woba_value, events, game_date
                FROM `{PROJECT}.{DATASET}.statcast_pitches`
                WHERE pitcher = {pitcher_id}
                  AND game_date < '{gd}'
                  AND game_year = 2026
                  AND woba_denom = 1
                UNION ALL
                SELECT batter, player_name, pitcher, woba_value, events, game_date
                FROM `{PROJECT}.{HIST_DS}.statcast_pitches`
                WHERE pitcher = {pitcher_id}
                  AND game_date >= '2022-01-01'
                  AND woba_denom = 1
            )
            GROUP BY batter
            HAVING COUNT(*) >= 5
        )
        SELECT s.batter, s.player_name, s.pa, s.woba, s.hr, s.hits, s.k,
               r.team_id
        FROM pa s
        JOIN roster r ON r.player_id = s.batter
        WHERE r.team_id = {batting_tid}
        ORDER BY ABS(s.woba - 0.320) DESC
        LIMIT 8
        """
        try:
            rows = _q(bq, sql)
            results[key] = [{
                "player_name": r["player_name"],
                "pa":          int(r["pa"]),
                "woba":        float(r["woba"]) if r["woba"] is not None else None,
                "hr":          int(r["hr"]),
                "hits":        int(r["hits"]),
                "k":           int(r["k"]),
                "k_pct":       round(int(r["k"]) / int(r["pa"]), 3) if int(r["pa"]) > 0 else None,
            } for r in rows]
        except Exception as e:
            logger.warning("batter vs pitcher query failed (pitcher=%s): %s", pitcher_id, e)

    return results


def fetch_venue_batter_stats(
    bq: bigquery.Client,
    venue_team_abbr: str,
    home_tid: int,
    away_tid: int,
    game_date: date,
) -> dict[str, list[dict]]:
    """
    Return batter performance at this specific ballpark (home_team = venue team)
    for both rosters. Returns {"home": [...], "away": [...]} top performers/struggles.
    """
    if not venue_team_abbr:
        return {"home": [], "away": []}
    gd = game_date.isoformat()
    sql = f"""
    WITH roster AS (
        SELECT player_id, team_id
        FROM (
            SELECT player_id, team_id,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC) AS rn
            FROM `{PROJECT}.{DATASET}.rosters`
            WHERE team_id IN ({home_tid}, {away_tid})
              AND position_type IN ('Hitter', 'Two-Way Player')
        ) WHERE rn = 1
    ),
    venue_pa AS (
        SELECT batter, MAX(player_name) AS player_name,
               COUNT(*) AS pa,
               ROUND(SUM(woba_value) / NULLIF(COUNT(*), 0), 3) AS woba,
               COUNTIF(events = 'home_run') AS hr,
               COUNTIF(events IN ('single','double','triple','home_run')) AS hits
        FROM (
            SELECT batter, player_name, home_team, woba_value, events, game_date
            FROM `{PROJECT}.{DATASET}.statcast_pitches`
            WHERE home_team = '{venue_team_abbr}'
              AND game_date < '{gd}'
              AND game_year = 2026
              AND woba_denom = 1
            UNION ALL
            SELECT batter, player_name, home_team, woba_value, events, game_date
            FROM `{PROJECT}.{HIST_DS}.statcast_pitches`
            WHERE home_team = '{venue_team_abbr}'
              AND game_date >= '2022-01-01'
              AND woba_denom = 1
        )
        GROUP BY batter
        HAVING COUNT(*) >= 8
    )
    SELECT vp.batter, vp.player_name, vp.pa, vp.woba, vp.hr, vp.hits, r.team_id
    FROM venue_pa vp
    JOIN roster r ON r.player_id = vp.batter
    ORDER BY r.team_id, ABS(vp.woba - 0.320) DESC
    LIMIT 20
    """
    try:
        rows = _q(bq, sql)
    except Exception as e:
        logger.warning("venue batter stats query failed: %s", e)
        return {"home": [], "away": []}

    home_list, away_list = [], []
    for r in rows:
        entry = {"player_name": r["player_name"], "pa": int(r["pa"]),
                 "woba": float(r["woba"]) if r["woba"] else None,
                 "hr": int(r["hr"]), "hits": int(r["hits"])}
        if int(r["team_id"]) == home_tid:
            home_list.append(entry)
        else:
            away_list.append(entry)
    return {"home": home_list[:5], "away": away_list[:5]}


def compute_hit_streaks(
    bq: bigquery.Client,
    home_tid: int,
    away_tid: int,
    home_abbr: str,
    away_abbr: str,
    game_date: date,
) -> dict[int, list[dict]]:
    """
    Compute current consecutive-game hit streaks in 2026 for hitters on both teams.
    Returns {team_id: [{player_name, hit_streak}]} for streaks >= 7.
    """
    gd = game_date.isoformat()
    sql = f"""
    WITH roster AS (
        SELECT player_id, team_id
        FROM (
            SELECT player_id, team_id,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC) AS rn
            FROM `{PROJECT}.{DATASET}.rosters`
            WHERE team_id IN ({home_tid}, {away_tid})
              AND position_type IN ('Hitter', 'Two-Way Player')
        ) WHERE rn = 1
    ),
    game_appearances AS (
        SELECT batter, MAX(player_name) AS player_name, game_date,
               MAX(IF(events IN ('single','double','triple','home_run'), 1, 0)) AS got_hit
        FROM `{PROJECT}.{DATASET}.statcast_pitches`
        WHERE game_date < '{gd}'
          AND game_year = 2026
          AND woba_denom = 1
          AND (home_team IN ('{home_abbr}', '{away_abbr}')
               OR away_team IN ('{home_abbr}', '{away_abbr}'))
        GROUP BY batter, game_date
    ),
    ranked AS (
        SELECT ga.batter, ga.player_name, r.team_id, ga.game_date, ga.got_hit,
               ROW_NUMBER() OVER (PARTITION BY ga.batter ORDER BY ga.game_date DESC) AS rn_all
        FROM game_appearances ga
        JOIN roster r ON r.player_id = ga.batter
    ),
    streaks AS (
        SELECT batter, MAX(player_name) AS player_name, team_id,
               COALESCE(
                   MIN(IF(got_hit = 0, rn_all, NULL)) - 1,
                   COUNT(*)
               ) AS hit_streak
        FROM ranked
        GROUP BY batter, team_id
    )
    SELECT batter, player_name, team_id, hit_streak
    FROM streaks
    WHERE hit_streak >= 7
    ORDER BY hit_streak DESC
    LIMIT 20
    """
    try:
        rows = _q(bq, sql)
    except Exception as e:
        logger.warning("hit streak query failed: %s", e)
        return {}

    result: dict[int, list] = {}
    for r in rows:
        tid = int(r["team_id"])
        result.setdefault(tid, []).append({
            "player_name": r["player_name"],
            "hit_streak":  int(r["hit_streak"]),
        })
    return result


def generate_fun_facts(
    matchup: dict,
    hit_streaks: dict,
    v8: dict | None,
    home_name: str,
    away_name: str,
    home_tid: int,
    away_tid: int,
) -> list[str]:
    """Build a list of narrative fun-fact strings from matchup + streak data."""
    facts: list[str] = []
    home_word = home_name.split()[-1]
    away_word = away_name.split()[-1]

    # --- Hit streaks ---
    for tid, team_name in [(home_tid, home_name), (away_tid, away_name)]:
        for p in hit_streaks.get(tid, []):
            s = p.get("hit_streak", 0)
            n = p.get("player_name", "")
            if s >= 15:
                facts.append(f"🔥 {n} is on fire — hits in {s} consecutive games this season")
            elif s >= 10:
                facts.append(f"🔥 {n} has hit in {s} straight games in 2026")
            elif s >= 7:
                facts.append(f"📈 {n} has a {s}-game hit streak")

    # --- Batter vs opponent history ---
    home_name_kw = home_word   # home batters hit vs away pitchers
    away_name_kw = away_word   # away batters hit vs home pitchers

    for side, team_batters, opp_word in [
        ("home", matchup.get("home", []), away_word),
        ("away", matchup.get("away", []), home_word),
    ]:
        for p in team_batters[:5]:
            woba_val = p.get("woba") or 0
            pa       = p.get("pa") or 0
            hr       = p.get("hr") or 0
            hits     = p.get("hits") or 0
            name     = p.get("player_name", "")
            if woba_val >= 0.420 and pa >= 10:
                facts.append(f"💥 {name} is mashing vs {opp_word} pitching — .{round(woba_val * 1000)} wOBA in {pa} career PA")
            elif woba_val >= 0.375 and pa >= 15:
                facts.append(f"👀 {name} has owned {opp_word} pitchers — .{round(woba_val * 1000)} wOBA, {hr} HR in {pa} PA")
            elif hr >= 4:
                facts.append(f"🏠 {name} has {hr} career HR vs {opp_word} pitching ({pa} PA)")
            elif woba_val <= 0.220 and pa >= 15:
                facts.append(f"❄️ {name} struggles vs {opp_word} pitching — .{round(woba_val * 1000)} wOBA in {pa} PA")
            if len([f for f in facts if opp_word in f]) >= 3:
                break

    # --- H2H team dominance ---
    if v8:
        h2h_pct = _safe_float(v8.get("h2h_win_pct_3yr"))
        h2h_g   = _safe_int(v8.get("h2h_games_3yr")) or 0
        if h2h_pct is not None and h2h_g >= 8:
            wins   = round(h2h_pct * h2h_g)
            losses = h2h_g - wins
            if h2h_pct >= 0.625:
                facts.append(f"📊 {home_word} have dominated these meetings lately — {wins}–{losses} in last {h2h_g} matchups")
            elif h2h_pct <= 0.375:
                facts.append(f"📊 {away_word} own this rivalry — {losses}–{wins} record against {home_word} (last {h2h_g} games)")

    return facts[:8]


def assemble_report(
    game: dict,
    pred: dict | None,
    v8: dict | None,
    mv7: dict | None,
    home_hot_cold: dict,
    away_hot_cold: dict,
    home_news: list[dict],
    away_news: list[dict],
    yearly_h2h: list[dict] | None = None,
    batter_vs_sp: dict | None = None,
    venue_stats: dict | None = None,
    matchup_vs_team: dict | None = None,
    hit_streaks: dict | None = None,
) -> dict:
    """Build the full scouting report JSON for one game."""

    home_name = game.get("home_team_name", "")
    away_name = game.get("away_team_name", "")
    home_kw   = home_name.split()[-1]
    away_kw   = away_name.split()[-1]

    # --- Prediction section ---
    prediction = None
    if pred:
        prediction = {
            "home_win_probability": _safe_float(pred.get("home_win_probability")),
            "away_win_probability": _safe_float(pred.get("away_win_probability")),
            "predicted_winner":     pred.get("predicted_winner"),
            "confidence_tier":      pred.get("confidence_tier"),
            "model_version":        pred.get("model_version"),
            "lineup_confirmed":     pred.get("lineup_confirmed"),
        }

    # --- Probable starters ---
    starters = {}
    if pred:
        starters["home"] = {
            "id":   _safe_int(pred.get("home_starter_id")),
            "name": pred.get("home_starter_name"),
            "hand": pred.get("home_starter_hand"),
        }
        starters["away"] = {
            "id":   _safe_int(pred.get("away_starter_id")),
            "name": pred.get("away_starter_name"),
            "hand": pred.get("away_starter_hand"),
        }
    # --- Pitcher arsenal (V10 SP percentile ranks from predictions) ---
    arsenal = {}
    for side in ("home", "away"):
        base = {}
        if mv7:
            base = {
                "mean_velo":     _safe_float(mv7.get(f"{side}_starter_mean_velo"), 1),
                "k_bb_pct":      _safe_float(mv7.get(f"{side}_starter_k_bb_pct")),
                "xwoba_allowed": _safe_float(mv7.get(f"{side}_starter_xwoba_allowed")),
                "venue_era":     _safe_float(mv7.get(f"{side}_starter_venue_era"), 2),
            }
        if pred:
            # V10 percentile ranks (Baseball Savant 0-100)
            base["xera_pct"]   = _safe_int(pred.get(f"{side}_sp_xera"))
            base["fbv_pct"]    = _safe_int(pred.get(f"{side}_sp_fbv_pct"))
            base["k_pct"]      = _safe_int(pred.get(f"{side}_sp_k_pct"))
            base["bb_pct"]     = _safe_int(pred.get(f"{side}_sp_bb_pct"))
            base["whiff_pct"]  = _safe_int(pred.get(f"{side}_sp_whiff_pct"))
            base["sp_known"]   = pred.get(f"{side}_sp_known")
        arsenal[side] = base

    # --- Team momentum (V8 features) ---
    momentum = {}
    if v8:
        for side in ("home", "away"):
            momentum[side] = {
                "elo":           _safe_int(v8.get(f"{side}_elo")),
                "pythag_pct":    _safe_float(v8.get(f"{side}_pythag_season")),
                "streak":        _streak_label(v8.get(f"{side}_current_streak")),
                "run_diff_10g":  _safe_float(v8.get(f"{side}_run_diff_10g"), 1),
                "record_10g":    None,
            }
        momentum["elo_differential"]    = _safe_int(v8.get("elo_differential"))
        momentum["h2h_win_pct_3yr"]     = _safe_float(v8.get("h2h_win_pct_3yr"))
        momentum["h2h_game_count_3yr"]  = _safe_int(v8.get("h2h_games_3yr"))
        momentum["is_divisional"]       = bool(v8.get("is_divisional"))

    # --- Hot/cold players ---
    hot_cold = {
        "home": build_hot_cold_section(home_hot_cold),
        "away": build_hot_cold_section(away_hot_cold),
    }

    # --- Bullpen ---
    bullpen = {}
    if mv7:
        bullpen = {
            "home_fatigue":        _safe_float(mv7.get("home_bullpen_fatigue_score"), 2),
            "away_fatigue":        _safe_float(mv7.get("away_bullpen_fatigue_score"), 2),
            "home_closer_rest":    _safe_int(mv7.get("home_closer_days_rest")),
            "away_closer_rest":    _safe_int(mv7.get("away_closer_days_rest")),
        }

    # --- Watcher callouts (key matchup to watch) ---
    watch_list = []
    if mv7:
        h2h_woba = _safe_float(mv7.get("home_h2h_woba"))
        h2h_pa   = _safe_int(mv7.get("home_h2h_pa_total")) or 0
        if h2h_woba and h2h_pa >= 15:
            delta = h2h_woba - 0.320
            direction = "familiar" if abs(delta) < 0.015 else ("dominates" if delta < -0.015 else "struggles vs")
            watch_list.append({
                "type": "h2h",
                "label": f"{away_kw} lineup {direction} {starters.get('home', {}).get('name', 'home starter')} ({h2h_pa} PA, .{round(h2h_woba*1000)} wOBA)",
            })
        a_h2h_woba = _safe_float(mv7.get("away_h2h_woba"))
        a_h2h_pa   = _safe_int(mv7.get("away_h2h_pa_total")) or 0
        if a_h2h_woba and a_h2h_pa >= 15:
            delta = a_h2h_woba - 0.320
            direction = "familiar" if abs(delta) < 0.015 else ("dominates" if delta < -0.015 else "struggles vs")
            watch_list.append({
                "type": "h2h",
                "label": f"{home_kw} lineup {direction} {starters.get('away', {}).get('name', 'away starter')} ({a_h2h_pa} PA, .{round(a_h2h_woba*1000)} wOBA)",
            })

    if v8:
        streak_h = _safe_int(v8.get("home_current_streak"))
        streak_a = _safe_int(v8.get("away_current_streak"))
        for team_name, streak_val in [(home_name, streak_h), (away_name, streak_a)]:
            if streak_val and abs(streak_val) >= 4:
                label = f"{team_name.split()[-1]} riding a {_streak_label(streak_val)} streak"
                watch_list.append({"type": "streak", "label": label})

    # Top hot hitters from each side
    for side_kw, hot_cold_data in [(home_kw, hot_cold["home"]), (away_kw, hot_cold["away"])]:
        for p in hot_cold_data.get("hot", [])[:1]:
            watch_list.append({
                "type": "hot_player",
                "label": f"{p['name']} 🔥 .{round((p['woba_14d'] or 0)*1000)} wOBA last 14 days",
            })

    # --- News ---
    news = {
        "home": home_news[:4],
        "away": away_news[:4],
    }

    return {
        "game_pk":        _safe_int(game["game_pk"]),
        "game_date":      str(game["game_date"]),
        "game_time_utc":  None,
        "venue_name":     game.get("venue_name"),
        "home_team_id":   _safe_int(game["home_team_id"]),
        "away_team_id":   _safe_int(game["away_team_id"]),
        "home_team_name": home_name,
        "away_team_name": away_name,
        "status":         game.get("status"),
        "home_score":     _safe_int(game.get("home_score")),
        "away_score":     _safe_int(game.get("away_score")),
        "prediction":     prediction,
        "starters":       starters,
        "arsenal":        arsenal,
        "momentum":       momentum,
        "hot_cold":       hot_cold,
        "bullpen":        bullpen,
        "watch_list":       watch_list[:6],
        "news":             news,
        "matchup_vs_team":  matchup_vs_team or {"home": [], "away": []},
        "h2h_yearly":       yearly_h2h or [],
        "batter_vs_sp":     batter_vs_sp or {"home": [], "away": []},
        "venue_stats":      venue_stats or {"home": [], "away": []},
        "fun_facts":      generate_fun_facts(
            matchup     = matchup_vs_team or {},
            hit_streaks = hit_streaks or {},
            v8          = v8,
            home_name   = home_name,
            away_name   = away_name,
            home_tid    = int(game.get("home_team_id", 0)),
            away_tid    = int(game.get("away_team_id", 0)),
        ),
        "generated_at":   datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# BQ write
# ---------------------------------------------------------------------------

def upsert_reports(bq: bigquery.Client, reports: list[dict], dry_run: bool = False) -> dict:
    if dry_run:
        logger.info("[DRY RUN] Would write %d reports", len(reports))
        return {"reports_written": 0, "dry_run": True}

    game_date = reports[0]["game_date"] if reports else None

    rows = []
    for r in reports:
        rows.append({
            "game_pk":        r["game_pk"],
            "game_date":      r["game_date"],
            "home_team_id":   r["home_team_id"],
            "away_team_id":   r["away_team_id"],
            "home_team_name": r["home_team_name"],
            "away_team_name": r["away_team_name"],
            "report":         json.dumps(r),
            "generated_at":   r["generated_at"],
        })

    # Use load job with partition overwrite to avoid streaming-buffer DELETE conflicts.
    # The $YYYYMMDD partition decorator overwrites exactly that day's partition atomically.
    partition_suffix = (game_date or "").replace("-", "")
    table_ref = f"{REPORTS_TABLE}${partition_suffix}"

    schema = [
        bigquery.SchemaField("game_pk",        "INTEGER"),
        bigquery.SchemaField("game_date",       "DATE"),
        bigquery.SchemaField("home_team_id",    "INTEGER"),
        bigquery.SchemaField("away_team_id",    "INTEGER"),
        bigquery.SchemaField("home_team_name",  "STRING"),
        bigquery.SchemaField("away_team_name",  "STRING"),
        bigquery.SchemaField("report",          "JSON"),
        bigquery.SchemaField("generated_at",    "TIMESTAMP"),
    ]
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=schema,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    job = bq.load_table_from_json(rows, table_ref, job_config=job_config)
    job.result()
    logger.info("Wrote %d scouting reports to BQ for %s (partition overwrite)", len(rows), game_date)

    return {"reports_written": len(rows), "game_date": game_date}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(target_date: date, dry_run: bool = False) -> dict:
    bq = bigquery.Client(project=PROJECT)

    logger.info("Building scouting reports for %s", target_date)

    games = fetch_games_on_date(bq, target_date)
    if not games:
        logger.info("No games on %s", target_date)
        return {"reports_written": 0, "game_date": str(target_date)}

    logger.info("Found %d games", len(games))

    predictions  = fetch_predictions(bq, target_date)
    v8_features  = fetch_v8_features(bq, target_date)
    mv7_features = fetch_matchup_features(bq, target_date)

    # Collect all team IDs and names
    team_ids   = list({int(g["home_team_id"]) for g in games} | {int(g["away_team_id"]) for g in games})
    team_names = list({g["home_team_name"] for g in games} | {g["away_team_name"] for g in games})

    hot_cold_by_team = fetch_hot_cold_players(bq, team_ids, target_date)
    news_by_keyword  = fetch_team_news(bq, team_names)
    team_abbrevs     = fetch_team_abbrevs(bq, team_ids)

    reports = []
    for game in games:
        gpk      = int(game["game_pk"])
        home_tid = int(game["home_team_id"])
        away_tid = int(game["away_team_id"])
        home_kw  = game["home_team_name"].split()[-1]
        away_kw  = game["away_team_name"].split()[-1]
        home_abbr = team_abbrevs.get(home_tid, "")
        away_abbr = team_abbrevs.get(away_tid, "")

        pred_row = predictions.get(gpk)
        home_sp_id = int(pred_row.get("home_starter_id") or 0) if pred_row else None
        away_sp_id = int(pred_row.get("away_starter_id") or 0) if pred_row else None
        home_sp_id = home_sp_id or None
        away_sp_id = away_sp_id or None

        matchup_vs_team = fetch_batter_vs_team_matchups(
            bq, home_abbr, away_abbr, home_tid, away_tid, target_date,
        ) if home_abbr and away_abbr else {"home": [], "away": []}

        hit_streaks = compute_hit_streaks(
            bq, home_tid, away_tid, home_abbr, away_abbr, target_date,
        ) if home_abbr and away_abbr else {}

        yearly_h2h = fetch_yearly_h2h_records(bq, home_tid, away_tid)

        batter_vs_sp = fetch_batter_vs_pitcher(
            bq, home_sp_id, away_sp_id, home_tid, away_tid, target_date
        )

        venue_stats = fetch_venue_batter_stats(
            bq, home_abbr, home_tid, away_tid, target_date
        ) if home_abbr else {"home": [], "away": []}

        report = assemble_report(
            game             = game,
            pred             = pred_row,
            v8               = v8_features.get(gpk),
            mv7              = mv7_features.get(gpk),
            home_hot_cold    = hot_cold_by_team.get(home_tid, []),
            away_hot_cold    = hot_cold_by_team.get(away_tid, []),
            home_news        = news_by_keyword.get(home_kw, []),
            away_news        = news_by_keyword.get(away_kw, []),
            matchup_vs_team  = matchup_vs_team,
            hit_streaks      = hit_streaks,
            yearly_h2h       = yearly_h2h,
            batter_vs_sp     = batter_vs_sp,
            venue_stats      = venue_stats,
        )
        reports.append(report)
        logger.info("  %s @ %s — %d watch callouts, %d fun facts, %d news items",
                    game["away_team_name"], game["home_team_name"],
                    len(report["watch_list"]),
                    len(report["fun_facts"]),
                    len(report["news"]["home"]) + len(report["news"]["away"]))

    result = upsert_reports(bq, reports, dry_run=dry_run)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    result = run(target, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
