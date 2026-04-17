"""
V10 Data Fetcher
================
Fetches the three new data types needed for V10:
  1. Per-game probable pitcher assignments (MLB Stats API, 2018-2026)
  2. Venue run-scoring data for park factors (computed from existing games CSV)
  3. Nothing else — rest/travel features are computed from existing game dates

Outputs to data/v10/raw/:
  - game_starters_{year}.parquet     — game_pk, home_sp_id, away_sp_id, venue_id
  - park_factors.parquet             — venue_id, venue_name, park_factor (0-100 scale, 100=neutral)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_DIR   = REPO_ROOT / "data" / "v10" / "raw"
V9_RAW_DIR = REPO_ROOT / "data" / "v9" / "raw"
GAMES_CSV  = REPO_ROOT.parent / "hanks_tank_backend" / "data" / "games" / "games_2015_2024.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MLB_API = "https://statsapi.mlb.com/api/v1"

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _sleep(s: float = 0.3) -> None:
    time.sleep(s)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-game probable pitcher assignments
# ─────────────────────────────────────────────────────────────────────────────

def fetch_game_starters(years: list) -> None:
    """
    Fetch probable pitcher for each game via:
      GET /schedule?sportId=1&season={year}&gameType=R&hydrate=probablePitcher

    The API returns the pitcher that was listed as probable at game time.
    For historical games, this is the actual starter in the vast majority of cases.

    Saves: game_starters_{year}.parquet
      Columns: game_pk, game_date, season, venue_id,
               home_team_id, away_team_id,
               home_sp_id, away_sp_id
    """
    for year in years:
        out_path = DATA_DIR / f"game_starters_{year}.parquet"
        if out_path.exists():
            logger.info(f"  game_starters_{year}: cached")
            continue

        logger.info(f"  Fetching probable pitchers {year}...")
        rows = []
        try:
            resp = requests.get(
                f"{MLB_API}/schedule",
                params={
                    "sportId": 1,
                    "season": year,
                    "gameType": "R",
                    "hydrate": "probablePitcher",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            for date_block in data.get("dates", []):
                for game in date_block.get("games", []):
                    status = game.get("status", {}).get("abstractGameState", "")
                    if status != "Final":
                        # For 2026 YTD include all scheduled/live games too
                        if year < 2026:
                            continue

                    teams = game.get("teams", {})
                    home = teams.get("home", {})
                    away = teams.get("away", {})

                    home_sp = home.get("probablePitcher", {})
                    away_sp = away.get("probablePitcher", {})

                    rows.append({
                        "game_pk":     game["gamePk"],
                        "game_date":   game.get("officialDate", date_block.get("date", "")),
                        "season":      year,
                        "venue_id":    game.get("venue", {}).get("id"),
                        "venue_name":  game.get("venue", {}).get("name", ""),
                        "home_team_id": home.get("team", {}).get("id"),
                        "away_team_id": away.get("team", {}).get("id"),
                        "home_sp_id":  home_sp.get("id"),
                        "away_sp_id":  away_sp.get("id"),
                        "home_sp_name": home_sp.get("fullName", ""),
                        "away_sp_name": away_sp.get("fullName", ""),
                        "series_game_number": game.get("seriesGameNumber", 1),
                        "games_in_series":    game.get("gamesInSeries", 3),
                    })

            df = pd.DataFrame(rows)
            if not df.empty:
                df["game_pk"]     = df["game_pk"].astype("Int64")
                df["season"]      = df["season"].astype(int)
                df["venue_id"]    = pd.to_numeric(df["venue_id"], errors="coerce")
                df["home_sp_id"]  = pd.to_numeric(df["home_sp_id"], errors="coerce")
                df["away_sp_id"]  = pd.to_numeric(df["away_sp_id"], errors="coerce")

            df.to_parquet(out_path, index=False)
            assigned = df[["home_sp_id", "away_sp_id"]].notna().all(axis=1).sum()
            logger.info(f"    -> {len(df)} games, {assigned} with both SPs assigned "
                        f"({100*assigned/max(len(df),1):.0f}%)")
            _sleep(0.5)

        except Exception as e:
            logger.warning(f"  game_starters_{year}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Park factors from historical run-scoring
# ─────────────────────────────────────────────────────────────────────────────

def build_park_factors() -> None:
    """
    Compute park factors from historical run-scoring data.

    Method: For each venue, compare average runs per game played there
    against the overall MLB average. A park factor of 1.10 means 10% more
    runs score there than average (hitter-friendly). Neutral = 1.00.

    Use 2015-2024 data for stability (larger sample). Use only regular-season
    completed games. Apply 3-year minimum sample cutoff.

    Saves: data/v10/raw/park_factors.parquet
      Columns: venue_id, venue_name, park_factor, park_factor_100, games_count
        park_factor:     raw ratio (1.00 = neutral)
        park_factor_100: scaled to 100 = neutral (FanGraphs style)
    """
    out_path = DATA_DIR / "park_factors.parquet"
    # Also copy to v9/raw so the V9 feature builder can use it
    v9_path  = V9_RAW_DIR / "park_factors.parquet"

    if out_path.exists():
        logger.info("  park_factors: cached")
        return

    logger.info("  Building park factors from historical run-scoring...")

    # Load games CSV
    if not GAMES_CSV.exists():
        logger.warning(f"  games CSV not found: {GAMES_CSV}")
        return

    df = pd.read_csv(GAMES_CSV)
    df = df[(df["status_code"] == "F") & (df["game_type"] == "R")].copy()
    df = df.dropna(subset=["home_score", "away_score", "venue_id"])
    df["total_runs"] = df["home_score"] + df["away_score"]

    # Overall MLB average runs per game
    mlb_avg = df["total_runs"].mean()
    logger.info(f"  MLB avg runs/game (2015-2024): {mlb_avg:.3f}")

    # Per-venue stats
    venue_stats = (
        df.groupby(["venue_id", "venue_name"])
        .agg(
            games_count=("total_runs", "count"),
            avg_runs=("total_runs", "mean"),
        )
        .reset_index()
    )

    # Minimum 100 games for reliable estimate
    venue_stats = venue_stats[venue_stats["games_count"] >= 100].copy()

    # Park factor: venue avg / MLB avg
    # Standard formula: PF = (runs at park / games at park) / (runs overall / games overall)
    venue_stats["park_factor"]     = venue_stats["avg_runs"] / mlb_avg
    venue_stats["park_factor_100"] = (venue_stats["park_factor"] * 100).round(1)

    # Also add current home-team venue mapping (most recent season per team)
    recent = df.sort_values("season").groupby("home_team_id").last().reset_index()
    tid_to_venue = dict(zip(recent["home_team_id"], recent["venue_id"]))
    venue_stats["team_id"] = venue_stats["venue_id"].map(
        {v: k for k, v in tid_to_venue.items()}
    )

    venue_stats = venue_stats.sort_values("park_factor", ascending=False)
    logger.info(f"  Park factors computed for {len(venue_stats)} venues")
    logger.info("  Top 5 hitter-friendly parks:")
    for _, row in venue_stats.head(5).iterrows():
        logger.info(f"    {row['venue_name']:<35} PF={row['park_factor']:.3f}  ({row['games_count']} games)")
    logger.info("  Top 5 pitcher-friendly parks:")
    for _, row in venue_stats.tail(5).iterrows():
        logger.info(f"    {row['venue_name']:<35} PF={row['park_factor']:.3f}  ({row['games_count']} games)")

    venue_stats.to_parquet(out_path, index=False)
    venue_stats.to_parquet(v9_path, index=False)
    logger.info(f"  Saved to {out_path}")
    logger.info(f"  Copied to {v9_path} (for V9 feature builder)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Validate Statcast pitcher data coverage
# ─────────────────────────────────────────────────────────────────────────────

def validate_statcast_coverage(years: list) -> None:
    """
    Quick coverage check: what % of game starters have Statcast percentile ranks?
    This tells us how much data we'll actually be able to use.
    """
    logger.info("\n  Statcast SP coverage check...")

    for year in years[-3:]:  # spot-check last 3 years
        starters_path = DATA_DIR / f"game_starters_{year}.parquet"
        pct_path      = V9_RAW_DIR / f"statcast_sp_pct_{year}.parquet"

        if not starters_path.exists() or not pct_path.exists():
            continue

        starters = pd.read_parquet(starters_path)
        pct      = pd.read_parquet(pct_path)
        pct_ids  = set(pct["player_id"].dropna().astype(int))

        home_covered = starters["home_sp_id"].dropna().astype(int).isin(pct_ids).mean()
        away_covered = starters["away_sp_id"].dropna().astype(int).isin(pct_ids).mean()
        both_covered = (
            starters["home_sp_id"].dropna().astype(int).isin(pct_ids) &
            starters["away_sp_id"].dropna().astype(int).isin(pct_ids)
        ).mean()

        logger.info(f"  {year}: home SP covered={home_covered:.1%}, "
                    f"away SP covered={away_covered:.1%}, "
                    f"both covered={both_covered:.1%}  "
                    f"({len(pct_ids)} pitchers in Statcast)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V10 data fetcher")
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(2018, 2027)),
                        help="Years to fetch starters for")
    parser.add_argument("--skip-starters", action="store_true")
    parser.add_argument("--skip-park-factors", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("V10 DATA FETCHER")
    logger.info(f"Output dir: {DATA_DIR}")
    logger.info(f"Years: {args.years}")
    logger.info("=" * 60)

    if not args.skip_starters:
        logger.info("\n[1/2] Per-game probable pitcher assignments...")
        fetch_game_starters(args.years)

    if not args.skip_park_factors:
        logger.info("\n[2/2] Park factors from historical run-scoring...")
        build_park_factors()

    validate_statcast_coverage(args.years)

    logger.info("\n" + "=" * 60)
    logger.info("V10 FETCH COMPLETE")
    logger.info(f"  Files in {DATA_DIR}:")
    for f in sorted(DATA_DIR.iterdir()):
        logger.info(f"    {f.name}  ({f.stat().st_size // 1024} KB)")
    logger.info("\nNext: python 02_build_v10_dataset.py")


if __name__ == "__main__":
    main()
