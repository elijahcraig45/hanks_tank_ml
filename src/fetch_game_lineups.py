#!/usr/bin/env python3
"""
Lineup Fetcher — fetches confirmed game lineups from the MLB Stats API
and stores them in BigQuery (mlb_2026_season.lineups).

Designed to run ~90 minutes before the first pitch of each game.
Can be called:
  - As a Cloud Function (mode=lineups) with a specific game_pk list
  - Standalone via CLI for testing

MLB API endpoint:
  GET /api/v1/game/{game_pk}/boxscore
    → teams.home.players, teams.away.players with battingOrder set pre-game

  GET /api/v1/schedule?date=YYYY-MM-DD&hydrate=lineups
    → returns lineups if officially released by the club

Usage:
    python fetch_game_lineups.py                    # today's games
    python fetch_game_lineups.py --date 2026-04-02  # specific date
    python fetch_game_lineups.py --game-pk 745612   # single game (comma-separated)
    python fetch_game_lineups.py --dry-run
"""

import argparse
import logging
import time
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
import requests
import urllib3
from google.cloud import bigquery

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
DATASET = "mlb_2026_season"
MLB_API = "https://statsapi.mlb.com/api/v1"
LINEUPS_TABLE = f"{PROJECT}.{DATASET}.lineups"
HEADERS = {"User-Agent": "HanksTank/2.0"}


# ---------------------------------------------------------------------------
# BigQuery schema
# ---------------------------------------------------------------------------
LINEUP_SCHEMA = [
    bigquery.SchemaField("game_pk", "INTEGER"),
    bigquery.SchemaField("game_date", "DATE"),
    bigquery.SchemaField("team_id", "INTEGER"),
    bigquery.SchemaField("team_name", "STRING"),
    bigquery.SchemaField("team_type", "STRING"),      # "home" | "away"
    bigquery.SchemaField("player_id", "INTEGER"),
    bigquery.SchemaField("player_name", "STRING"),
    bigquery.SchemaField("batting_order", "INTEGER"), # 1–9, null for pitchers not batting
    bigquery.SchemaField("position", "STRING"),       # P, C, 1B, … DH, PH
    bigquery.SchemaField("bat_side", "STRING"),       # L | R | S
    bigquery.SchemaField("pitch_hand", "STRING"),     # L | R | S (for pitchers)
    bigquery.SchemaField("is_starter", "BOOLEAN"),
    bigquery.SchemaField("is_probable_pitcher", "BOOLEAN"),
    bigquery.SchemaField("status", "STRING"),         # Active, On Bench, etc.
    bigquery.SchemaField("game_time_utc", "TIMESTAMP"),
    bigquery.SchemaField("fetched_at", "TIMESTAMP"),
    bigquery.SchemaField("lineup_confirmed", "BOOLEAN"),
]


class LineupFetcher:
    def __init__(self, dry_run: bool = False):
        self.bq = bigquery.Client(project=PROJECT)
        self.dry_run = dry_run
        self._handedness_cache: dict[int, dict] = {}  # player_id → {bat_side, pitch_hand}
        self._ensure_table()

    # -----------------------------------------------------------------------
    # Table management
    # -----------------------------------------------------------------------
    def _ensure_table(self) -> None:
        """Create the lineups table if it doesn't exist."""
        table_ref = bigquery.Table(LINEUPS_TABLE, schema=LINEUP_SCHEMA)
        table_ref.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="game_date",
        )
        table_ref.clustering_fields = ["game_pk", "team_type"]
        try:
            self.bq.get_table(LINEUPS_TABLE)
        except Exception:
            if not self.dry_run:
                self.bq.create_table(table_ref, exists_ok=True)
                logger.info("Created table %s", LINEUPS_TABLE)

    # -----------------------------------------------------------------------
    # Handedness lookup (batch)
    # -----------------------------------------------------------------------
    def fetch_handedness(self, player_ids: list[int]) -> None:
        """
        Batch-fetch bat_side and pitch_hand for a list of player IDs.
        Populates self._handedness_cache.  Skips already-cached IDs.
        The MLB people API accepts up to ~200 IDs per call.
        """
        to_fetch = [pid for pid in player_ids if pid not in self._handedness_cache]
        if not to_fetch:
            return

        chunk_size = 200
        for i in range(0, len(to_fetch), chunk_size):
            chunk = to_fetch[i : i + chunk_size]
            ids_str = ",".join(str(pid) for pid in chunk)
            url = f"{MLB_API}/people"
            try:
                resp = requests.get(
                    url,
                    params={"personIds": ids_str},
                    headers=HEADERS,
                    timeout=30,
                    verify=False,
                )
                resp.raise_for_status()
                for person in resp.json().get("people", []):
                    pid = person.get("id")
                    if pid:
                        self._handedness_cache[pid] = {
                            "bat_side": (person.get("batSide") or {}).get("code", ""),
                            "pitch_hand": (person.get("pitchHand") or {}).get("code", ""),
                        }
            except Exception as e:
                logger.warning("Handedness fetch failed for chunk: %s", e)

    # -----------------------------------------------------------------------
    # Schedule fetching
    # -----------------------------------------------------------------------
    def fetch_schedule(self, target_date: date) -> list[dict]:
        """Return list of {game_pk, game_date, game_time_utc, home_team_id, away_team_id}."""
        url = f"{MLB_API}/schedule"
        params = {
            "date": target_date.isoformat(),
            "sportId": 1,
            "hydrate": "team,probablePitcher",
            "gameType": "R,F,D,L,W",  # Regular + postseason
        }
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()

        games = []
        for day in data.get("dates", []):
            for g in day.get("games", []):
                if g.get("status", {}).get("abstractGameState") == "Final":
                    continue  # skip completed games
                game_datetime_str = g.get("gameDate", "")
                try:
                    game_time_utc = datetime.fromisoformat(
                        game_datetime_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    game_time_utc = None

                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})
                games.append({
                    "game_pk": g["gamePk"],
                    "game_date": day["date"],
                    "game_time_utc": game_time_utc,
                    "home_team_id": home.get("team", {}).get("id"),
                    "home_team_name": home.get("team", {}).get("name", ""),
                    "away_team_id": away.get("team", {}).get("id"),
                    "away_team_name": away.get("team", {}).get("name", ""),
                    "home_probable_pitcher_id": home.get("probablePitcher", {}).get("id"),
                    "away_probable_pitcher_id": away.get("probablePitcher", {}).get("id"),
                })
        return games

    # -----------------------------------------------------------------------
    # Lineup fetching for a single game
    # -----------------------------------------------------------------------
    def fetch_lineup_for_game(self, game_info: dict) -> list[dict]:
        """
        Fetch confirmed lineup via /game/{pk}/boxscore.
        Falls back to /schedule?hydrate=lineups when boxscore has no batting order.
        Returns list of player-row dicts ready for BQ insertion.
        """
        game_pk = game_info["game_pk"]
        rows = []
        fetched_at = datetime.now(tz=timezone.utc).isoformat()
        lineup_confirmed = False

        # Primary: boxscore endpoint (has batting order pre-game)
        url = f"{MLB_API}/game/{game_pk}/boxscore"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30, verify=False)
            resp.raise_for_status()
            boxscore = resp.json()
            lineup_confirmed, rows = self._parse_boxscore(
                boxscore, game_info, fetched_at
            )
        except Exception as e:
            logger.warning("Boxscore fetch failed for game %s: %s", game_pk, e)

        # Fallback: if no batting order found try the schedule lineups hydration
        if not lineup_confirmed:
            rows = self._fetch_lineup_from_schedule(game_info, fetched_at)
            lineup_confirmed = len(rows) > 0

        if not rows:
            logger.info("No lineup data available yet for game %s", game_pk)
            return []

        logger.info(
            "Fetched %d player rows for game %s (confirmed=%s)",
            len(rows), game_pk, lineup_confirmed
        )
        return rows

    def _parse_boxscore(
        self, boxscore: dict, game_info: dict, fetched_at: str
    ) -> tuple[bool, list[dict]]:
        """Parse boxscore JSON into lineup rows. Returns (lineup_confirmed, rows)."""
        rows = []
        lineup_confirmed = False
        game_pk = game_info["game_pk"]
        game_date = game_info["game_date"]
        game_time_utc = (
            game_info["game_time_utc"].isoformat()
            if game_info.get("game_time_utc")
            else None
        )

        # Collect all player IDs first so we can batch-fetch handedness
        all_player_ids = [
            p.get("person", {}).get("id")
            for side in ("home", "away")
            for p in boxscore.get("teams", {}).get(side, {}).get("players", {}).values()
            if p.get("person", {}).get("id")
        ]
        self.fetch_handedness(all_player_ids)

        for team_type in ("home", "away"):
            team_data = boxscore.get("teams", {}).get(team_type, {})
            team_info = team_data.get("team", {})
            team_id_key = f"{team_type}_team_id"
            team_name_key = f"{team_type}_team_name"
            team_id = game_info.get(team_id_key)
            team_name = game_info.get(team_name_key, "")

            players = team_data.get("players", {})
            for player_key, player_data in players.items():
                pid = player_data.get("person", {}).get("id")
                pname = player_data.get("person", {}).get("fullName", "")
                pos = player_data.get("position", {}).get("abbreviation", "")
                status = player_data.get("status", {}).get("description", "")
                batting_order_raw = player_data.get("battingOrder")

                # Handedness from cache (populated by batch fetch above)
                hand_info = self._handedness_cache.get(pid, {})
                bat_side = hand_info.get("bat_side", "")
                pitch_hand = hand_info.get("pitch_hand", "")

                # Determine batting order (1–9 within each team, stored as 1–9)
                batting_order = None
                if batting_order_raw is not None:
                    try:
                        batting_order = int(str(batting_order_raw).strip("0")) if batting_order_raw != 0 else None
                        # MLB API uses 100,200,...900 for batting order
                        if isinstance(batting_order_raw, (int, float)) and batting_order_raw >= 100:
                            batting_order = int(batting_order_raw) // 100
                    except (ValueError, TypeError):
                        batting_order = None

                is_starter = batting_order is not None or pos == "P"
                if batting_order is not None:
                    lineup_confirmed = True

                # Only include active players (starters + bench)
                if status not in ("", "Active", "On Bench", "On 1st Base", "On 2nd Base", "On 3rd Base"):
                    continue

                is_probable_pitcher = (
                    pid == game_info.get(f"{team_type}_probable_pitcher_id")
                )

                rows.append({
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "team_id": team_id,
                    "team_name": team_name,
                    "team_type": team_type,
                    "player_id": pid,
                    "player_name": pname,
                    "batting_order": batting_order,
                    "position": pos,
                    "bat_side": bat_side,
                    "pitch_hand": pitch_hand,
                    "is_starter": is_starter,
                    "is_probable_pitcher": is_probable_pitcher,
                    "status": status,
                    "game_time_utc": game_time_utc,
                    "fetched_at": fetched_at,
                    "lineup_confirmed": lineup_confirmed,
                })

        return lineup_confirmed, rows

    def _fetch_lineup_from_schedule(
        self, game_info: dict, fetched_at: str
    ) -> list[dict]:
        """Try schedule endpoint with lineups hydration as fallback."""
        game_pk = game_info["game_pk"]
        game_date = game_info["game_date"]
        url = f"{MLB_API}/schedule"
        params = {
            "gamePk": game_pk,
            "hydrate": "lineups",
        }
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30, verify=False)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Schedule lineup fallback failed for game %s: %s", game_pk, e)
            return []

        rows = []
        game_time_utc = (
            game_info["game_time_utc"].isoformat()
            if game_info.get("game_time_utc")
            else None
        )

        for day in data.get("dates", []):
            for g in day.get("games", []):
                if g.get("gamePk") != game_pk:
                    continue
                lineups = g.get("lineups", {})
                for team_type in ("homePlayers", "awayPlayers"):
                    side = "home" if team_type.startswith("home") else "away"
                    team_id = game_info.get(f"{side}_team_id")
                    team_name = game_info.get(f"{side}_team_name", "")
                    players = lineups.get(team_type, [])
                    for i, player in enumerate(players):
                        pid = player.get("id")
                        pname = player.get("fullName", "")
                        rows.append({
                            "game_pk": game_pk,
                            "game_date": game_date,
                            "team_id": team_id,
                            "team_name": team_name,
                            "team_type": side,
                            "player_id": pid,
                            "player_name": pname,
                            "batting_order": i + 1,
                            "position": player.get("primaryPosition", {}).get("abbreviation", ""),
                            "bat_side": player.get("batSide", {}).get("code", ""),
                            "pitch_hand": player.get("pitchHand", {}).get("code", ""),
                            "is_starter": True,
                            "is_probable_pitcher": False,
                            "status": "Active",
                            "game_time_utc": game_time_utc,
                            "fetched_at": fetched_at,
                            "lineup_confirmed": True,
                        })
        return rows

    # -----------------------------------------------------------------------
    # BigQuery write
    # -----------------------------------------------------------------------
    def upsert_lineups(self, rows: list[dict]) -> None:
        """Append lineup rows. BQ reads deduplicate via QUALIFY."""
        if not rows:
            return

        game_pks = list({r["game_pk"] for r in rows})
        pk_list = ", ".join(str(pk) for pk in game_pks)

        if not self.dry_run:
            df = pd.DataFrame(rows)
            # Parse date/timestamp columns for BQ schema compatibility
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
            df["game_time_utc"] = pd.to_datetime(df["game_time_utc"], utc=True, errors="coerce")
            df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce")

            job_config = bigquery.LoadJobConfig(
                schema=LINEUP_SCHEMA,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            )
            job = self.bq.load_table_from_dataframe(df, LINEUPS_TABLE, job_config=job_config)
            job.result()
            logger.info("Inserted %d lineup rows for games %s", len(rows), pk_list)
        else:
            logger.info("[DRY RUN] Would insert %d lineup rows for games %s", len(rows), pk_list)

    # -----------------------------------------------------------------------
    # Main entry points
    # -----------------------------------------------------------------------
    def run_for_date(self, target_date: date) -> dict:
        """Fetch lineups for all games on a given date."""
        games = self.fetch_schedule(target_date)
        logger.info("Found %d upcoming/in-progress games on %s", len(games), target_date)

        all_rows = []
        for game in games:
            rows = self.fetch_lineup_for_game(game)
            all_rows.extend(rows)
            time.sleep(0.5)  # polite rate limiting

        self.upsert_lineups(all_rows)
        confirmed = sum(1 for r in all_rows if r.get("lineup_confirmed") and r.get("batting_order") == 1)
        return {
            "games_checked": len(games),
            "player_rows": len(all_rows),
            "lineups_confirmed": confirmed,
        }

    def run_for_game_pks(self, game_pks: list[int], game_date: Optional[date] = None) -> dict:
        """Fetch lineups for specific game PKs (used by Cloud Tasks)."""
        target_date = game_date or date.today()
        schedule = self.fetch_schedule(target_date)
        schedule_map = {g["game_pk"]: g for g in schedule}

        all_rows = []
        for pk in game_pks:
            game_info = schedule_map.get(pk)
            if not game_info:
                # Build minimal game_info from API if not in today's schedule
                game_info = {"game_pk": pk, "game_date": target_date.isoformat(),
                             "game_time_utc": None,
                             "home_team_id": None, "home_team_name": "",
                             "away_team_id": None, "away_team_name": "",
                             "home_probable_pitcher_id": None, "away_probable_pitcher_id": None}
            rows = self.fetch_lineup_for_game(game_info)
            all_rows.extend(rows)
            time.sleep(0.5)

        self.upsert_lineups(all_rows)
        return {"game_pks_processed": len(game_pks), "player_rows": len(all_rows)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch MLB game lineups into BigQuery")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--game-pk", help="Comma-separated game PKs to fetch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fetcher = LineupFetcher(dry_run=args.dry_run)

    if args.game_pk:
        pks = [int(pk.strip()) for pk in args.game_pk.split(",")]
        target = date.fromisoformat(args.date) if args.date else date.today()
        result = fetcher.run_for_game_pks(pks, target)
    else:
        target = date.fromisoformat(args.date) if args.date else date.today()
        result = fetcher.run_for_date(target)

    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()
