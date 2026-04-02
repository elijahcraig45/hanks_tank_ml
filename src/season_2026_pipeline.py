#!/usr/bin/env python3
"""
2026 Season Daily Data Pipeline

Collects all MLB data for the current day (or a date range for backfill)
and loads it into the mlb_2026_season BigQuery dataset.

Data sources:
  - MLB Stats API (games, standings, rosters, team/player stats, transactions)
  - Baseball Savant (statcast pitch-by-pitch)

Usage:
    python season_2026_pipeline.py                       # yesterday's data
    python season_2026_pipeline.py --date 2026-03-27     # specific date
    python season_2026_pipeline.py --backfill --start 2026-02-20  # backfill range
    python season_2026_pipeline.py --dry-run             # preview only
"""

import argparse
import logging
import time
from datetime import datetime, date, timedelta, timezone
from typing import Optional

import pandas as pd
import numpy as np
import requests
import urllib3
from io import StringIO
from google.cloud import bigquery

# Disable SSL warnings for local dev (macOS cert issues with statsapi.mlb.com)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
DATASET = "mlb_2026_season"
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"
SEASON = 2026
BATCH_SIZE = 500
REQUEST_HEADERS = {"User-Agent": "HanksTank-Pipeline/2.0"}


class SeasonPipeline:
    """Collects and loads 2026 MLB data into BigQuery."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.bq = bigquery.Client(project=PROJECT)
        self.stats = {"games": 0, "team_stats": 0, "player_stats": 0,
                      "standings": 0, "rosters": 0, "transactions": 0,
                      "statcast": 0, "errors": []}

    # ------------------------------------------------------------------
    # MLB Stats API helpers
    # ------------------------------------------------------------------
    def _api_get(self, path: str, params: dict | None = None) -> dict:
        url = f"{MLB_API_BASE}{path}"
        resp = requests.get(url, params=params, headers=REQUEST_HEADERS,
                            timeout=30, verify=False)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Games
    # ------------------------------------------------------------------
    def collect_games(self, target_date: date) -> pd.DataFrame:
        ds = target_date.isoformat()
        data = self._api_get("/schedule", params={
            "date": ds, "sportId": 1,
            "hydrate": "linescore,decisions,team",
            "fields": (
                "dates,date,games,gamePk,gameDate,gameType,status,detailedState,"
                "statusCode,teams,home,away,team,id,name,score,leagueRecord,wins,"
                "losses,venue,id,name,linescore,currentInning,decisions,"
                "winner,id,fullName,loser,id,fullName,save,id,fullName"
            ),
        })

        rows = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})
                dec = g.get("decisions", {})
                ls = g.get("linescore", {})
                rows.append({
                    "game_pk": g["gamePk"],
                    "game_date": target_date.isoformat(),
                    "game_type": g.get("gameType", "R"),
                    "home_team_id": home.get("team", {}).get("id"),
                    "home_team_name": home.get("team", {}).get("name"),
                    "away_team_id": away.get("team", {}).get("id"),
                    "away_team_name": away.get("team", {}).get("name"),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                    "venue_id": g.get("venue", {}).get("id"),
                    "venue_name": g.get("venue", {}).get("name"),
                    "status": g.get("status", {}).get("detailedState"),
                    "status_code": g.get("status", {}).get("statusCode"),
                    "innings": ls.get("currentInning"),
                    "winning_pitcher_id": dec.get("winner", {}).get("id"),
                    "winning_pitcher_name": dec.get("winner", {}).get("fullName"),
                    "losing_pitcher_id": dec.get("loser", {}).get("id"),
                    "losing_pitcher_name": dec.get("loser", {}).get("fullName"),
                    "save_pitcher_id": dec.get("save", {}).get("id"),
                    "save_pitcher_name": dec.get("save", {}).get("fullName"),
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        logger.info("  games: %d rows for %s", len(df), ds)
        return df

    # ------------------------------------------------------------------
    # Standings
    # ------------------------------------------------------------------
    def collect_standings(self, target_date: date) -> pd.DataFrame:
        ds = target_date.isoformat()
        rows = []
        for league_id in (103, 104):  # AL, NL
            data = self._api_get("/standings", params={
                "leagueId": league_id, "season": SEASON,
                "date": ds, "standingsType": "regularSeason",
                "hydrate": "team",
            })
            for rec in data.get("records", []):
                div = rec.get("division", {})
                lg = rec.get("league", {})
                for tr in rec.get("teamRecords", []):
                    team = tr.get("team", {})
                    split = tr.get("records", {}).get("splitRecords", [])
                    home_rec = next((s for s in split if s.get("type") == "home"), {})
                    away_rec = next((s for s in split if s.get("type") == "away"), {})
                    rows.append({
                        "team_id": team.get("id"),
                        "team_name": team.get("name"),
                        "league_id": lg.get("id", league_id),
                        "league_name": lg.get("name"),
                        "division_id": div.get("id"),
                        "division_name": div.get("name"),
                        "wins": tr.get("wins", 0),
                        "losses": tr.get("losses", 0),
                        "win_percentage": float(tr.get("winningPercentage", ".000").replace(".", "0.", 1) if tr.get("winningPercentage", ".000").startswith(".") else tr.get("winningPercentage", "0")),
                        "games_back": float(tr.get("gamesBack", "0").replace("-", "0")),
                        "wildcard_games_back": float(tr.get("wildCardGamesBack", "0").replace("-", "0")),
                        "division_rank": tr.get("divisionRank"),
                        "league_rank": tr.get("leagueRank"),
                        "runs_scored": tr.get("runsScored", 0),
                        "runs_allowed": tr.get("runsAllowed", 0),
                        "run_differential": tr.get("runDifferential", 0),
                        "home_wins": home_rec.get("wins", 0),
                        "home_losses": home_rec.get("losses", 0),
                        "away_wins": away_rec.get("wins", 0),
                        "away_losses": away_rec.get("losses", 0),
                        "streak": tr.get("streak", {}).get("streakCode"),
                        "last_ten": f"{tr.get('records', {}).get('splitRecords', [{}])[0].get('wins', 0)}-{tr.get('records', {}).get('splitRecords', [{}])[0].get('losses', 0)}",
                        "snapshot_date": ds,
                        "synced_at": datetime.now(timezone.utc).isoformat(),
                    })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
        logger.info("  standings: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Team Stats (batting + pitching)
    # ------------------------------------------------------------------
    def collect_team_stats(self, target_date: date) -> pd.DataFrame:
        rows = []
        for group, stat_type in [("hitting", "batting"), ("pitching", "pitching")]:
            data = self._api_get(
                "/teams/stats",
                params={"season": SEASON, "group": group,
                        "stats": "season", "sportIds": 1},
            )
            for split in data.get("stats", [{}])[0].get("splits", []):
                team = split.get("team", {})
                s = split.get("stat", {})
                row = {
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "stat_type": stat_type,
                    "games_played": s.get("gamesPlayed", 0),
                    "snapshot_date": target_date.isoformat(),
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                }
                if stat_type == "batting":
                    row.update({
                        "at_bats": s.get("atBats", 0),
                        "runs": s.get("runs", 0),
                        "hits": s.get("hits", 0),
                        "doubles": s.get("doubles", 0),
                        "triples": s.get("triples", 0),
                        "home_runs": s.get("homeRuns", 0),
                        "rbi": s.get("rbi", 0),
                        "stolen_bases": s.get("stolenBases", 0),
                        "caught_stealing": s.get("caughtStealing", 0),
                        "walks": s.get("baseOnBalls", 0),
                        "strikeouts": s.get("strikeOuts", 0),
                        "batting_avg": float(s.get("avg", "0") or "0"),
                        "obp": float(s.get("obp", "0") or "0"),
                        "slg": float(s.get("slg", "0") or "0"),
                        "ops": float(s.get("ops", "0") or "0"),
                        "total_bases": s.get("totalBases", 0),
                        "hit_by_pitch": s.get("hitByPitch", 0),
                        "sac_flies": s.get("sacFlies", 0),
                        "sac_bunts": s.get("sacBunts", 0),
                        "left_on_base": s.get("leftOnBase", 0),
                    })
                else:
                    row.update({
                        "wins": s.get("wins", 0),
                        "losses": s.get("losses", 0),
                        "win_percentage": float(s.get("winPercentage", "0") or "0"),
                        "era": float(s.get("era", "0") or "0"),
                        "games_started": s.get("gamesStarted", 0),
                        "games_finished": s.get("gamesFinished", 0),
                        "complete_games": s.get("completeGames", 0),
                        "shutouts": s.get("shutouts", 0),
                        "saves": s.get("saves", 0),
                        "save_opportunities": s.get("saveOpportunities", 0),
                        "holds": s.get("holds", 0),
                        "blown_saves": s.get("blownSaves", 0),
                        "innings_pitched": float(s.get("inningsPitched", "0") or "0"),
                        "hits_allowed": s.get("hits", 0),
                        "runs_allowed": s.get("runs", 0),
                        "earned_runs": s.get("earnedRuns", 0),
                        "home_runs_allowed": s.get("homeRuns", 0),
                        "walks_allowed": s.get("baseOnBalls", 0),
                        "pitching_strikeouts": s.get("strikeOuts", 0),
                        "whip": float(s.get("whip", "0") or "0"),
                        "batters_faced": s.get("battersFaced", 0),
                        "wild_pitches": s.get("wildPitches", 0),
                        "hit_batsmen": s.get("hitBatsmen", 0),
                        "balks": s.get("balks", 0),
                    })
                rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
        logger.info("  team_stats: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Player Stats (leaderboard: top 250 batting + pitching)
    # ------------------------------------------------------------------
    def collect_player_stats(self, target_date: date) -> pd.DataFrame:
        rows = []
        for group, stat_type in [("hitting", "batting"), ("pitching", "pitching")]:
            data = self._api_get(
                "/stats/leaders",
                params={
                    "season": SEASON,
                    "leaderCategories": "homeRuns" if group == "hitting" else "earnedRunAverage",
                    "statGroup": group,
                    "limit": 250,
                    "sportId": 1,
                    "hydrate": "person,team",
                    "fields": "leagueLeaders,leaders,rank,value,person,id,fullName,team,id,name,stat",
                },
            )
            for cat in data.get("leagueLeaders", []):
                for ldr in cat.get("leaders", []):
                    person = ldr.get("person", {})
                    team = ldr.get("team", {})
                    s = ldr.get("stat", {}) or {}
                    row = {
                        "player_id": person.get("id"),
                        "player_name": person.get("fullName"),
                        "team_id": team.get("id"),
                        "team_name": team.get("name"),
                        "stat_type": stat_type,
                        "games_played": s.get("gamesPlayed", 0),
                        "snapshot_date": target_date.isoformat(),
                        "synced_at": datetime.now(timezone.utc).isoformat(),
                    }
                    if stat_type == "batting":
                        row.update({
                            "at_bats": s.get("atBats", 0),
                            "runs": s.get("runs", 0),
                            "hits": s.get("hits", 0),
                            "home_runs": s.get("homeRuns", 0),
                            "rbi": s.get("rbi", 0),
                            "stolen_bases": s.get("stolenBases", 0),
                            "batting_avg": float(s.get("avg", "0") or "0"),
                            "obp": float(s.get("obp", "0") or "0"),
                            "slg": float(s.get("slg", "0") or "0"),
                            "ops": float(s.get("ops", "0") or "0"),
                        })
                    else:
                        row.update({
                            "wins": s.get("wins", 0),
                            "losses": s.get("losses", 0),
                            "era": float(s.get("era", "0") or "0"),
                            "innings_pitched": float(s.get("inningsPitched", "0") or "0"),
                            "pitching_strikeouts": s.get("strikeOuts", 0),
                            "whip": float(s.get("whip", "0") or "0"),
                        })
                    rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
        logger.info("  player_stats: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Rosters
    # ------------------------------------------------------------------
    def collect_rosters(self, target_date: date) -> pd.DataFrame:
        teams = self._api_get("/teams", params={"season": SEASON, "sportId": 1})
        rows = []
        for team in teams.get("teams", []):
            tid = team["id"]
            tname = team["name"]
            try:
                roster = self._api_get(f"/teams/{tid}/roster",
                                       params={"season": SEASON, "rosterType": "active"})
            except requests.HTTPError:
                continue
            for p in roster.get("roster", []):
                person = p.get("person", {})
                pos = p.get("position", {})
                rows.append({
                    "team_id": tid,
                    "team_name": tname,
                    "player_id": person.get("id"),
                    "player_name": person.get("fullName"),
                    "jersey_number": p.get("jerseyNumber"),
                    "position_code": pos.get("code"),
                    "position_name": pos.get("name"),
                    "position_type": pos.get("type"),
                    "status": p.get("status", {}).get("code"),
                    "snapshot_date": target_date.isoformat(),
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                })
            time.sleep(0.25)  # rate-limit
        df = pd.DataFrame(rows)
        if not df.empty:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
        logger.info("  rosters: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------
    def collect_transactions(self, target_date: date) -> pd.DataFrame:
        ds = target_date.isoformat()
        data = self._api_get("/transactions", params={"date": ds})
        rows = []
        for tx in data.get("transactions", []):
            person = tx.get("person", {})
            from_team = tx.get("fromTeam", {})
            to_team = tx.get("toTeam", {})
            rows.append({
                "transaction_id": tx.get("id"),
                "date": tx.get("date", ds)[:10],
                "type_code": tx.get("typeCode"),
                "type_desc": tx.get("typeDesc"),
                "description": tx.get("description"),
                "from_team_id": from_team.get("id"),
                "from_team_name": from_team.get("name"),
                "from_team_abbreviation": from_team.get("abbreviation"),
                "to_team_id": to_team.get("id"),
                "to_team_name": to_team.get("name"),
                "to_team_abbreviation": to_team.get("abbreviation"),
                "person_id": person.get("id"),
                "person_full_name": person.get("fullName"),
                "person_link": person.get("link"),
                "resolution": tx.get("resolutionDate"),
                "notes": tx.get("note"),
                "synced_at": datetime.now(timezone.utc).isoformat(),
            })
        df = pd.DataFrame(rows)
        # Some transactions have no person (team-level moves) — drop nulls for BQ required fields
        if not df.empty:
            df = df.dropna(subset=["person_id", "person_full_name"])
            df["person_id"] = df["person_id"].astype(int)
        logger.info("  transactions: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Statcast pitches (Baseball Savant CSV)
    # ------------------------------------------------------------------
    def collect_statcast(self, target_date: date) -> pd.DataFrame:
        ds = target_date.isoformat()
        params = {
            "all": "true",
            "hfGT": "R|S|",  # regular + spring training
            "hfSea": f"{SEASON}|",
            "player_type": "pitcher",
            "game_date_gt": ds,
            "game_date_lt": ds,
            "min_pitches": "0",
            "min_results": "0",
            "min_pas": "0",
            "group_by": "name",
            "sort_col": "pitches",
            "sort_order": "desc",
            "type": "details",
        }
        try:
            resp = requests.get(SAVANT_CSV, params=params,
                                headers=REQUEST_HEADERS, timeout=60,
                                verify=False)
            resp.raise_for_status()
            if len(resp.content) < 200:
                logger.info("  statcast: no data for %s", ds)
                return pd.DataFrame()
            df = pd.read_csv(StringIO(resp.text))
            # Standardise columns
            renames = {"game_year": "game_year"}
            df = df.rename(columns=renames)
            if "game_date" in df.columns:
                df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
            df["synced_at"] = datetime.now(timezone.utc).isoformat()
            logger.info("  statcast: %d pitches for %s", len(df), ds)
            return df
        except Exception as e:
            logger.warning("  statcast fetch failed for %s: %s", ds, e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Teams reference (run once or weekly)
    # ------------------------------------------------------------------
    def collect_teams(self) -> pd.DataFrame:
        data = self._api_get("/teams", params={"season": SEASON, "sportId": 1})
        rows = []
        for t in data.get("teams", []):
            rows.append({
                "team_id": t["id"],
                "team_name": t["name"],
                "team_code": t.get("abbreviation"),
                "location_name": t.get("locationName"),
                "team_name_full": t.get("teamName"),
                "league_id": t.get("league", {}).get("id"),
                "league_name": t.get("league", {}).get("name"),
                "division_id": t.get("division", {}).get("id"),
                "division_name": t.get("division", {}).get("name"),
                "venue_id": t.get("venue", {}).get("id"),
                "venue_name": t.get("venue", {}).get("name"),
                "first_year_of_play": t.get("firstYearOfPlay"),
                "active": t.get("active", True),
                "synced_at": datetime.now(timezone.utc).isoformat(),
            })
        df = pd.DataFrame(rows)
        logger.info("  teams: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # BigQuery load helpers
    # ------------------------------------------------------------------
    def _load_to_bq(self, df: pd.DataFrame, table_name: str,
                     write_mode: str = "WRITE_APPEND"):
        if df.empty:
            return
        if self.dry_run:
            logger.info("  [DRY RUN] would load %d rows → %s.%s.%s",
                        len(df), PROJECT, DATASET, table_name)
            return

        # Convert synced_at from string to proper datetime for BQ TIMESTAMP
        if "synced_at" in df.columns:
            df["synced_at"] = pd.to_datetime(df["synced_at"], utc=True)

        table_id = f"{PROJECT}.{DATASET}.{table_name}"
        job_kwargs = {"write_disposition": write_mode}
        if write_mode != "WRITE_TRUNCATE":
            job_kwargs["schema_update_options"] = [
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
            ]
        job_config = bigquery.LoadJobConfig(**job_kwargs)
        job = self.bq.load_table_from_dataframe(df, table_id,
                                                  job_config=job_config)
        job.result()
        logger.info("  ✓ loaded %d rows → %s", job.output_rows, table_id)

    def _delete_date(self, table_name: str, date_col: str, target_date: date):
        """Delete existing rows for a date before re-inserting (idempotent)."""
        if self.dry_run:
            return
        query = (
            f"DELETE FROM `{PROJECT}.{DATASET}.{table_name}` "
            f"WHERE {date_col} = '{target_date.isoformat()}'"
        )
        self.bq.query(query).result()

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_daily(self, target_date: date):
        logger.info("=" * 60)
        logger.info("Pipeline run for %s", target_date)
        logger.info("=" * 60)

        # Games — idempotent via delete-then-insert
        try:
            games = self.collect_games(target_date)
            if not games.empty:
                self._delete_date("games", "game_date", target_date)
                self._load_to_bq(games, "games")
                self.stats["games"] += len(games)
        except Exception as e:
            logger.error("games failed: %s", e)
            self.stats["errors"].append(f"games: {e}")

        # Standings snapshot
        try:
            standings = self.collect_standings(target_date)
            if not standings.empty:
                self._delete_date("standings", "snapshot_date", target_date)
                self._load_to_bq(standings, "standings")
                self.stats["standings"] += len(standings)
        except Exception as e:
            logger.error("standings failed: %s", e)
            self.stats["errors"].append(f"standings: {e}")

        # Team stats snapshot
        try:
            tstats = self.collect_team_stats(target_date)
            if not tstats.empty:
                self._delete_date("team_stats", "snapshot_date", target_date)
                self._load_to_bq(tstats, "team_stats")
                self.stats["team_stats"] += len(tstats)
        except Exception as e:
            logger.error("team_stats failed: %s", e)
            self.stats["errors"].append(f"team_stats: {e}")

        # Player stats
        try:
            pstats = self.collect_player_stats(target_date)
            if not pstats.empty:
                self._delete_date("player_stats", "snapshot_date", target_date)
                self._load_to_bq(pstats, "player_stats")
                self.stats["player_stats"] += len(pstats)
        except Exception as e:
            logger.error("player_stats failed: %s", e)
            self.stats["errors"].append(f"player_stats: {e}")

        # Transactions
        try:
            txns = self.collect_transactions(target_date)
            if not txns.empty:
                self._load_to_bq(txns, "transactions")
                self.stats["transactions"] += len(txns)
        except Exception as e:
            logger.error("transactions failed: %s", e)
            self.stats["errors"].append(f"transactions: {e}")

        # Statcast — pull two days to compensate for Baseball Savant's ~24-48h
        # publication lag.  target_date is "yesterday"; target_date-1 is the
        # day Savant has typically just finished publishing.
        for sc_date in (target_date - timedelta(days=1), target_date):
            try:
                sc = self.collect_statcast(sc_date)
                if not sc.empty:
                    self._delete_date("statcast_pitches", "game_date", sc_date)
                    # Only keep columns that exist in the BQ schema
                    keep_cols = [
                        "pitch_type", "game_date", "game_year", "game_pk",
                        "pitcher", "batter", "player_name", "events", "description",
                        "release_speed", "release_pos_x", "release_pos_z",
                        "release_spin_rate", "spin_axis", "release_extension",
                        "effective_speed", "zone", "plate_x", "plate_z",
                        "pfx_x", "pfx_z", "game_type", "stand", "p_throws",
                        "home_team", "away_team", "balls", "strikes",
                        "inning", "inning_topbot", "outs_when_up",
                        "hit_distance_sc", "launch_speed", "launch_angle",
                        "estimated_ba_using_speedangle",
                        "estimated_woba_using_speedangle",
                        "estimated_slg_using_speedangle",
                        "woba_value", "woba_denom", "babip_value", "iso_value",
                        "vx0", "vy0", "vz0", "ax", "ay", "az",
                        "sz_top", "sz_bot", "bat_speed", "swing_length",
                        "attack_angle", "delta_home_win_exp", "delta_run_exp",
                        "pitch_name", "home_score", "away_score", "synced_at",
                    ]
                    existing = [c for c in keep_cols if c in sc.columns]
                    sc = sc[existing]
                    self._load_to_bq(sc, "statcast_pitches")
                    self.stats["statcast"] += len(sc)
            except Exception as e:
                logger.error("statcast failed for %s: %s", sc_date, e)
                self.stats["errors"].append(f"statcast({sc_date}): {e}")

    def run_rosters(self, target_date: date):
        """Rosters only need weekly refresh."""
        try:
            rosters = self.collect_rosters(target_date)
            if not rosters.empty:
                self._delete_date("rosters", "snapshot_date", target_date)
                self._load_to_bq(rosters, "rosters")
                self.stats["rosters"] += len(rosters)
        except Exception as e:
            logger.error("rosters failed: %s", e)
            self.stats["errors"].append(f"rosters: {e}")

    def run_teams_reference(self):
        """Teams reference — run once at setup or after expansion."""
        try:
            teams = self.collect_teams()
            if not teams.empty:
                self._load_to_bq(teams, "teams", write_mode="WRITE_TRUNCATE")
        except Exception as e:
            logger.error("teams reference failed: %s", e)

    def run_backfill(self, start: date, end: date):
        logger.info("Backfilling %s → %s", start, end)
        current = start
        while current <= end:
            self.run_daily(current)
            # Roster snapshot once per week
            if current.weekday() == 0:  # Monday
                self.run_rosters(current)
            current += timedelta(days=1)
            time.sleep(1)  # be gentle on APIs
        # Always get latest rosters
        self.run_rosters(end)
        self.run_teams_reference()

    def print_summary(self):
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        for k, v in self.stats.items():
            if k != "errors":
                logger.info("  %-20s %s rows", k, v)
        if self.stats["errors"]:
            logger.warning("  ERRORS: %d", len(self.stats["errors"]))
            for e in self.stats["errors"]:
                logger.warning("    • %s", e)
        else:
            logger.info("  ✅ No errors")
        logger.info("=" * 60)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="2026 MLB Season Pipeline")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--backfill", action="store_true", help="Backfill mode")
    parser.add_argument("--start", help="Backfill start date YYYY-MM-DD")
    parser.add_argument("--end", help="Backfill end date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pipeline = SeasonPipeline(dry_run=args.dry_run)

    if args.backfill:
        start = date.fromisoformat(args.start) if args.start else date(2026, 2, 20)
        end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
        pipeline.run_backfill(start, end)
    else:
        target = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
        pipeline.run_daily(target)
        # Rosters on Mondays
        if target.weekday() == 0:
            pipeline.run_rosters(target)

    pipeline.print_summary()


if __name__ == "__main__":
    main()
