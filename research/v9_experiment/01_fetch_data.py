#!/usr/bin/env python3
"""
V9 Data Fetcher — pulls all external data needed for V9 features.

Sources (FanGraphs blocked 403; uses working alternatives):
  1. MLB Stats API /teams/stats    -> Team pitching/batting stats per season
  2. Statcast pitcher percentile ranks -> SP quality (xERA, K%, BB%, whiff%, fbv)
  3. Statcast batter percentile ranks  -> Batter quality (xwOBA, EV, hard-hit%)
  4. Statcast pitcher expected stats   -> Individual xERA, xwOBA
  5. MLB Stats API /schedule           -> 2020-2026 game results + scores

Outputs (saved to data/v9/raw/):
  mlb_team_pitching_{year}.parquet    MLB API team pitching
  mlb_team_batting_{year}.parquet     MLB API team batting
  statcast_sp_pct_{year}.parquet      Statcast SP percentile ranks
  statcast_batter_pct_{year}.parquet  Statcast batter percentile ranks
  statcast_sp_xera_{year}.parquet     Statcast SP expected stats
  statcast_team_pitching_{year}.parquet  Team-level SP Statcast aggregates
  statcast_team_batting_{year}.parquet   Team-level batter Statcast aggregates
  games_{year}.parquet                Game results (MLB API)

Usage:
    python 01_fetch_data.py
    python 01_fetch_data.py --years 2020 2021 2022 2023 2024 2025 2026
    python 01_fetch_data.py --skip-statcast   # skip Statcast (faster)
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "v9" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MLB_API = "https://statsapi.mlb.com/api/v1"

ALL_TEAM_IDS = [
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 133, 134, 135, 136, 137, 138,
    139, 140, 141, 142, 143, 144, 145, 146, 147, 158,
]


def _sleep(s: float = 0.5) -> None:
    time.sleep(s)


# ─────────────────────────────────────────────────────────────────────────────
# MLB Stats API helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_mlb_team_stats(years: list) -> None:
    """Fetch team pitching and batting stats from MLB Stats API."""
    # MLB API uses "hitting" not "batting" as the group name
    GROUP_MAP = {"pitching": "pitching", "batting": "hitting"}
    for year in years:
        for stat_group, api_group in GROUP_MAP.items():
            out_path = DATA_DIR / f"mlb_team_{stat_group}_{year}.parquet"
            if out_path.exists():
                logger.info(f"  mlb_team_{stat_group}_{year}: cached")
                continue

            logger.info(f"  Fetching MLB API team {stat_group} {year}...")
            try:
                resp = requests.get(
                    f"{MLB_API}/teams/stats",
                    params={
                        "season": year, "sportId": 1, "stats": "season",
                        "group": api_group, "gameType": "R",
                    },
                    timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()

                rows = []
                for stat_block in data.get("stats", []):
                    for split in stat_block.get("splits", []):
                        row = {
                            "team_id": split["team"]["id"],
                            "team_name": split["team"]["name"],
                            "season": year,
                        }
                        row.update(split.get("stat", {}))
                        rows.append(row)

                df = pd.DataFrame(rows)
                for col in df.columns:
                    if col not in ("team_id", "team_name", "season"):
                        df[col] = pd.to_numeric(df[col], errors="ignore")

                df.to_parquet(out_path, index=False)
                logger.info(f"    -> {len(df)} teams, {len(df.columns)} cols")
                _sleep(0.3)
            except Exception as e:
                logger.warning(f"  mlb_team_{stat_group}_{year}: {e}")


def fetch_mlb_games(years: list) -> None:
    """Fetch completed game results from MLB Stats API."""
    for year in years:
        out_path = DATA_DIR / f"games_{year}.parquet"
        if out_path.exists():
            logger.info(f"  games_{year}: cached")
            continue

        logger.info(f"  Fetching MLB API games {year}...")
        rows = []
        try:
            resp = requests.get(
                f"{MLB_API}/schedule",
                params={
                    "sportId": 1, "season": year, "gameType": "R",
                    "startDate": f"{year}-03-15",
                    "endDate": f"{year}-10-10",
                    "hydrate": "linescore",
                    "fields": (
                        "dates,date,games,gamePk,gameDate,status,abstractGameState,"
                        "teams,home,away,team,id,name,score,isWinner"
                    ),
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            for date_obj in data.get("dates", []):
                game_date = date_obj["date"]
                for game in date_obj.get("games", []):
                    status = game.get("status", {}).get("abstractGameState", "")
                    if status != "Final":
                        continue
                    home = game["teams"]["home"]
                    away = game["teams"]["away"]
                    rows.append({
                        "game_pk": game["gamePk"],
                        "game_date": game_date,
                        "season": year,
                        "home_team_id": home["team"]["id"],
                        "home_team_name": home["team"].get("name", ""),
                        "away_team_id": away["team"]["id"],
                        "away_team_name": away["team"].get("name", ""),
                        "home_score": home.get("score"),
                        "away_score": away.get("score"),
                        "home_won": int(home.get("isWinner", False)),
                    })

            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values("game_date").reset_index(drop=True)
                df.to_parquet(out_path, index=False)
                logger.info(f"    -> {len(df)} games")
            else:
                logger.warning(f"  No completed games for {year}")
            _sleep(0.5)
        except Exception as e:
            logger.warning(f"  games_{year}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Statcast data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_statcast_pitcher_data(years: list) -> None:
    """Fetch Statcast pitcher percentile ranks and expected stats."""
    import pybaseball as pb
    pb.cache.enable()

    for year in years:
        # Percentile ranks
        pct_path = DATA_DIR / f"statcast_sp_pct_{year}.parquet"
        if pct_path.exists():
            logger.info(f"  statcast_sp_pct_{year}: cached")
        else:
            logger.info(f"  Fetching Statcast SP percentile ranks {year}...")
            try:
                df = pb.statcast_pitcher_percentile_ranks(year)
                df["season"] = year
                df.to_parquet(pct_path, index=False)
                logger.info(f"    -> {len(df)} pitchers")
                _sleep(1.0)
            except Exception as e:
                logger.warning(f"  statcast_sp_pct_{year}: {e}")

        # Expected stats (xERA, xwOBA)
        xera_path = DATA_DIR / f"statcast_sp_xera_{year}.parquet"
        if xera_path.exists():
            logger.info(f"  statcast_sp_xera_{year}: cached")
        else:
            logger.info(f"  Fetching Statcast SP expected stats {year}...")
            try:
                df = pb.statcast_pitcher_expected_stats(year)
                df["season"] = year
                df.to_parquet(xera_path, index=False)
                logger.info(f"    -> {len(df)} pitchers")
                _sleep(1.0)
            except Exception as e:
                logger.warning(f"  statcast_sp_xera_{year}: {e}")


def fetch_statcast_batter_data(years: list) -> None:
    """Fetch Statcast batter percentile ranks per year."""
    import pybaseball as pb
    pb.cache.enable()

    for year in years:
        out_path = DATA_DIR / f"statcast_batter_pct_{year}.parquet"
        if out_path.exists():
            logger.info(f"  statcast_batter_pct_{year}: cached")
            continue
        logger.info(f"  Fetching Statcast batter percentile ranks {year}...")
        try:
            df = pb.statcast_batter_percentile_ranks(year)
            df["season"] = year
            df.to_parquet(out_path, index=False)
            logger.info(f"    -> {len(df)} batters")
            _sleep(1.0)
        except Exception as e:
            logger.warning(f"  statcast_batter_pct_{year}: {e}")


def _get_team_pitcher_ids(year: int) -> list:
    """Get pitcher IDs by team from MLB API rosters."""
    rows = []
    for team_id in ALL_TEAM_IDS:
        try:
            resp = requests.get(
                f"{MLB_API}/teams/{team_id}/roster",
                params={"season": year, "rosterType": "active"},
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            for player in data.get("roster", []):
                pos = player.get("position", {}).get("abbreviation", "")
                if pos in ("SP", "RP", "P", "TWP"):  # active roster returns "P" for all pitchers
                    rows.append({
                        "team_id": team_id,
                        "player_id": player["person"]["id"],
                        "is_pitcher": True,
                    })
            _sleep(0.1)
        except Exception:
            pass
    return rows


def _get_team_batter_ids(year: int) -> list:
    """Get batter IDs by team from MLB API rosters."""
    rows = []
    for team_id in ALL_TEAM_IDS:
        try:
            resp = requests.get(
                f"{MLB_API}/teams/{team_id}/roster",
                params={"season": year, "rosterType": "active"},
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            for player in data.get("roster", []):
                pos = player.get("position", {}).get("abbreviation", "")
                if pos not in ("SP", "RP", "P", "TWP"):
                    rows.append({
                        "team_id": team_id,
                        "player_id": player["person"]["id"],
                        "position": pos,
                    })
            _sleep(0.1)
        except Exception:
            pass
    return rows


def build_team_statcast_aggregates(years: list) -> None:
    """
    Aggregate individual Statcast percentile ranks to team level.
    Produces team-level averages for SP quality and batting quality.
    """
    for year in years:
        # Pitcher aggregates
        out_pitch = DATA_DIR / f"statcast_team_pitching_{year}.parquet"
        if not out_pitch.exists():
            pct_path = DATA_DIR / f"statcast_sp_pct_{year}.parquet"
            if pct_path.exists():
                logger.info(f"  Building Statcast team pitching aggregates {year}...")
                pct_df = pd.read_parquet(pct_path)
                roster = _get_team_pitcher_ids(year)
                if roster:
                    roster_df = pd.DataFrame(roster)
                    merged = roster_df.merge(pct_df, on="player_id", how="inner")
                    # Use all pitchers on the roster (active roster doesn't distinguish SP/RP)
                    agg_df = merged.groupby("team_id").agg(
                        sc_sp_xera_pct=("xera", "mean"),
                        sc_sp_k_pct=("k_percent", "mean"),
                        sc_sp_bb_pct=("bb_percent", "mean"),
                        sc_sp_whiff_pct=("whiff_percent", "mean"),
                        sc_sp_fbv_pct=("fb_velocity", "mean"),
                        sc_sp_xwoba_pct=("xwoba", "mean"),
                        sc_sp_count=("player_id", "count"),
                    ).reset_index()
                    agg_df["season"] = year
                    agg_df.to_parquet(out_pitch, index=False)
                    logger.info(f"    -> {len(agg_df)} teams")

        # Batter aggregates
        out_bat = DATA_DIR / f"statcast_team_batting_{year}.parquet"
        if not out_bat.exists():
            bat_path = DATA_DIR / f"statcast_batter_pct_{year}.parquet"
            if bat_path.exists():
                logger.info(f"  Building Statcast team batting aggregates {year}...")
                bat_df = pd.read_parquet(bat_path)
                roster = _get_team_batter_ids(year)
                if roster:
                    roster_df = pd.DataFrame(roster)
                    merged = roster_df.merge(bat_df, on="player_id", how="inner")
                    agg_df = merged.groupby("team_id").agg(
                        sc_bat_xwoba_pct=("xwoba", "mean"),
                        sc_bat_ev_pct=("exit_velocity", "mean"),
                        sc_bat_hh_pct=("hard_hit_percent", "mean"),
                        sc_bat_brl_pct=("brl_percent", "mean"),
                        sc_bat_k_pct=("k_percent", "mean"),
                        sc_bat_bb_pct=("bb_percent", "mean"),
                        sc_bat_count=("player_id", "count"),
                    ).reset_index()
                    agg_df["season"] = year
                    agg_df.to_parquet(out_bat, index=False)
                    logger.info(f"    -> {len(agg_df)} teams")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V9 data fetcher (MLB API + Statcast)")
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(2018, 2027)),
                        help="Years to fetch stats for")
    parser.add_argument("--skip-statcast", action="store_true",
                        help="Skip Statcast fetches (just use MLB API team stats)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("V9 DATA FETCHER (MLB Stats API + Statcast)")
    logger.info(f"Output dir: {DATA_DIR}")
    logger.info(f"Years: {args.years}")
    logger.info("=" * 60)

    # 1. MLB API team stats
    logger.info("\n[1/4] MLB Stats API team pitching + batting...")
    fetch_mlb_team_stats(args.years)

    # 2. MLB API game results
    logger.info("\n[2/4] MLB Stats API game results (2020-2026)...")
    fetch_mlb_games([y for y in args.years if y >= 2020])

    # 3 & 4. Statcast
    if not args.skip_statcast:
        statcast_years = [y for y in args.years if 2018 <= y <= 2026]

        logger.info("\n[3/4] Statcast SP data...")
        fetch_statcast_pitcher_data(statcast_years)

        logger.info("\n[4/4] Statcast batter data + team aggregates...")
        fetch_statcast_batter_data(statcast_years)
        build_team_statcast_aggregates(statcast_years)
    else:
        logger.info("\n[3-4] Skipping Statcast (--skip-statcast)")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DATA FETCH COMPLETE")
    files = sorted(DATA_DIR.iterdir())
    for f in files:
        logger.info(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
    logger.info("=" * 60)
    logger.info("\nNext: python 02_build_v9_dataset.py")


if __name__ == "__main__":
    main()
