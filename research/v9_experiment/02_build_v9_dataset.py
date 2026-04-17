#!/usr/bin/env python3
"""
V9 Feature Builder — Professional-Analyst-Grade Features

Builds on V8's Elo/Pythagorean/rolling foundation and adds:
  - FanGraphs team pitching quality (xFIP, FIP, K%, BB%, WHIP, HR/9) — season & prior-year
  - FanGraphs team batting quality (wRC+, wOBA, OBP, ISO, K%, BB%) — season & prior-year
  - Starting pitcher quality (xFIP, K%, BB%, recent form from FanGraphs rolling)
  - Bullpen ERA/FIP from FanGraphs team data split
  - FanGraphs multi-year park factors (100=neutral, >100=hitter-friendly)
  - Run differential features (3g, 7g, 10g, 30g rolling windows)
  - Streak features (direction, magnitude)
  - Head-to-head features (season + 3yr)

Output:
  data/v9/features/train_v9.parquet       2015-2023 (training)
  data/v9/features/dev_v9.parquet         2024 (dev/calibration)
  data/v9/features/val_v9.parquet         2025 (validation holdout)
  data/v9/features/test_2026_v9.parquet   2026 YTD (live test)
  data/v9/features/feature_metadata.json  Column descriptions + groups

Usage:
    python 02_build_v9_dataset.py
    python 02_build_v9_dataset.py --rebuild  # force rebuild even if cached
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = REPO_ROOT / "data" / "v9" / "raw"
FEATURES_DIR = REPO_ROOT / "data" / "v9" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Legacy game CSV from backend
GAMES_CSV = REPO_ROOT.parent / "hanks_tank_backend" / "data" / "games" / "games_2015_2024.csv"

# Elo constants (V8-optimized)
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_BONUS = 70.0
ELO_SEASON_REGRESSION = 0.33

# Park factor defaults if FanGraphs data unavailable
DEFAULT_PARK_FACTOR = 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Elo engine
# ─────────────────────────────────────────────────────────────────────────────

def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def compute_elo_ratings(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute game-by-game Elo ratings. Returns games df with elo columns added.
    Expects: game_date, home_team_id, away_team_id, home_won, season
    """
    logger.info("  Computing Elo ratings...")
    games = games.sort_values("game_date").copy()
    elo: dict[int, float] = {}
    processed_seasons: set = set()

    home_elos, away_elos, home_elo_win_probs = [], [], []

    for _, row in games.iterrows():
        season = row["season"]
        htid = int(row["home_team_id"])
        atid = int(row["away_team_id"])

        # Season-start regression
        if season not in processed_seasons:
            processed_seasons.add(season)
            for tid in list(elo.keys()):
                elo[tid] = ELO_START + (elo[tid] - ELO_START) * (1.0 - ELO_SEASON_REGRESSION)

        # Initialise new teams
        elo.setdefault(htid, ELO_START)
        elo.setdefault(atid, ELO_START)

        h_elo = elo[htid] + ELO_HOME_BONUS
        a_elo = elo[atid]
        win_prob = elo_expected(h_elo, a_elo)

        home_elos.append(elo[htid])
        away_elos.append(elo[atid])
        home_elo_win_probs.append(win_prob)

        # Update
        actual = float(row["home_won"])
        delta = ELO_K * (actual - win_prob)
        elo[htid] += delta
        elo[atid] -= delta

    games["home_elo"] = home_elos
    games["away_elo"] = away_elos
    games["elo_differential"] = games["home_elo"] - games["away_elo"]
    games["elo_home_win_prob"] = home_elo_win_probs
    return games


# ─────────────────────────────────────────────────────────────────────────────
# Rolling game-by-game stats
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_team_stats(games: pd.DataFrame) -> pd.DataFrame:
    """
    Build rolling stats per team per game (uses only data BEFORE each game).
    Adds: win_pct, run_diff, runs_scored, runs_allowed (various windows),
    pythagorean, streaks, head-to-head.
    """
    logger.info("  Building rolling team stats...")
    games = games.sort_values("game_date").reset_index(drop=True)

    # We maintain a running state per team
    team_games: dict[int, list] = {}  # team_id → list of {"date", "won", "rs", "ra", "opp"}

    result_rows = []

    for idx, row in games.iterrows():
        htid = int(row["home_team_id"])
        atid = int(row["away_team_id"])
        gdate = str(row["game_date"])
        home_won = int(row["home_won"])
        hs = row.get("home_score", np.nan)
        as_ = row.get("away_score", np.nan)

        def get_stats(team_id: int, is_home: bool) -> dict:
            history = team_games.get(team_id, [])
            rs_col = [g["rs"] for g in history if not np.isnan(g["rs"])]
            ra_col = [g["ra"] for g in history if not np.isnan(g["ra"])]
            won_col = [g["won"] for g in history]
            opp_col = [g["opp"] for g in history]

            def win_pct(n: int) -> float:
                w = won_col[-n:] if n else won_col
                return np.mean(w) if w else 0.5

            def avg_rdiff(n: int) -> float:
                rs = rs_col[-n:]
                ra = ra_col[-n:]
                if len(rs) == 0 or len(ra) == 0:
                    return 0.0
                return float(np.mean(rs) - np.mean(ra))

            def avg_runs_scored(n: int) -> float:
                rs = rs_col[-n:]
                return float(np.mean(rs)) if rs else 4.5

            def avg_runs_allowed(n: int) -> float:
                ra = ra_col[-n:]
                return float(np.mean(ra)) if ra else 4.5

            # Pythagorean
            total_rs = sum(rs_col) if rs_col else 1.0
            total_ra = sum(ra_col) if ra_col else 1.0
            rs_exp = max(total_rs, 1e-6) ** 1.83
            ra_exp = max(total_ra, 1e-6) ** 1.83
            pythag = rs_exp / (rs_exp + ra_exp)
            actual_wp = win_pct(0) if won_col else 0.5
            luck = actual_wp - pythag

            # Season pythagorean
            season = row["season"]
            season_hist = [g for g in history if g.get("season") == season]
            if season_hist:
                sr = sum(g["rs"] for g in season_hist if not np.isnan(g["rs"]))
                sa = sum(g["ra"] for g in season_hist if not np.isnan(g["ra"]))
                se = max(sr, 1e-6) ** 1.83 / (max(sr, 1e-6) ** 1.83 + max(sa, 1e-6) ** 1.83)
            else:
                se = 0.5

            # Rolling 30g pythagorean
            sr30 = sum(rs_col[-30:]) if len(rs_col) >= 5 else total_rs
            sa30 = sum(ra_col[-30:]) if len(ra_col) >= 5 else total_ra
            p30e = max(sr30, 1e-6) ** 1.83
            p30a = max(sa30, 1e-6) ** 1.83
            pythag_30g = p30e / (p30e + p30a)

            # Streak
            streak = 0
            for w in reversed(won_col):
                if len(won_col) == 0:
                    break
                if streak == 0:
                    streak = 1 if w else -1
                elif (w and streak > 0) or (not w and streak < 0):
                    streak += (1 if w else -1)
                else:
                    break
            streak_dir = 1 if streak > 0 else (-1 if streak < 0 else 0)

            # H2H vs this opponent
            h2h_hist = [g for g in history if g["opp"] == (atid if is_home else htid)]
            h2h_pct_season = np.mean([g["won"] for g in h2h_hist if g.get("season") == season]) if any(g.get("season") == season for g in h2h_hist) else 0.5
            h2h_games_season = sum(1 for g in h2h_hist if g.get("season") == season)
            h2h_years = [g for g in h2h_hist if g.get("season", 0) >= (season - 3)]
            h2h_pct_3yr = np.mean([g["won"] for g in h2h_years]) if h2h_years else 0.5

            pfx = "home" if is_home else "away"
            return {
                f"{pfx}_win_pct_3g": win_pct(3),
                f"{pfx}_win_pct_7g": win_pct(7),
                f"{pfx}_win_pct_10g": win_pct(10),
                f"{pfx}_win_pct_30g": win_pct(30),
                f"{pfx}_win_pct_season": win_pct(0),
                f"{pfx}_run_diff_3g": avg_rdiff(3),
                f"{pfx}_run_diff_7g": avg_rdiff(7),
                f"{pfx}_run_diff_10g": avg_rdiff(10),
                f"{pfx}_run_diff_30g": avg_rdiff(30),
                f"{pfx}_runs_scored_10g": avg_runs_scored(10),
                f"{pfx}_runs_scored_30g": avg_runs_scored(30),
                f"{pfx}_runs_allowed_10g": avg_runs_allowed(10),
                f"{pfx}_runs_allowed_30g": avg_runs_allowed(30),
                f"{pfx}_pythag_season": se,
                f"{pfx}_pythag_last30": pythag_30g,
                f"{pfx}_luck_factor": luck,
                f"{pfx}_current_streak": abs(streak),
                f"{pfx}_streak_direction": streak_dir,
                f"{pfx}_games_played": len(history),
                f"{pfx}_era_proxy_10g": avg_runs_allowed(10),  # proxy for pitching quality
                f"{pfx}_era_proxy_30g": avg_runs_allowed(30),
                f"{pfx}_scoring_momentum": avg_runs_scored(7) - avg_runs_scored(30),
                f"{pfx}_h2h_win_pct_season": h2h_pct_season,
                f"{pfx}_h2h_games_season": h2h_games_season,
                f"{pfx}_h2h_win_pct_3yr": h2h_pct_3yr,
            }

        home_stats = get_stats(htid, is_home=True)
        away_stats = get_stats(atid, is_home=False)
        merged = {**home_stats, **away_stats}

        # Differentials
        merged["elo_win_prob_differential"] = row.get("elo_home_win_prob", 0.5) - 0.5
        merged["pythag_differential"] = merged["home_pythag_season"] - merged["away_pythag_season"]
        merged["run_diff_differential"] = merged["home_run_diff_10g"] - merged["away_run_diff_10g"]
        merged["luck_differential"] = merged["home_luck_factor"] - merged["away_luck_factor"]
        merged["win_pct_diff"] = merged["home_win_pct_season"] - merged["away_win_pct_season"]
        merged["streak_differential"] = (
            merged["home_current_streak"] * merged["home_streak_direction"]
            - merged["away_current_streak"] * merged["away_streak_direction"]
        )
        merged["era_proxy_differential"] = merged["away_era_proxy_10g"] - merged["home_era_proxy_10g"]  # positive = home pitching better

        # Context
        season_games = [g for g in team_games.get(htid, []) if g.get("season") == row["season"]]
        games_in_season = len(season_games)
        merged["season_game_number"] = games_in_season
        merged["season_pct_complete"] = min(games_in_season / 162.0, 1.0)
        merged["is_late_season"] = int(games_in_season > 120)
        merged["is_early_season"] = int(games_in_season < 30)

        result_rows.append(merged)

        # Update history for both teams
        game_entry_home = {
            "date": gdate, "won": home_won,
            "rs": hs if not np.isnan(hs) else np.nan,
            "ra": as_ if not np.isnan(as_) else np.nan,
            "opp": atid, "season": row["season"],
        }
        game_entry_away = {
            "date": gdate, "won": 1 - home_won,
            "rs": as_ if not np.isnan(as_) else np.nan,
            "ra": hs if not np.isnan(hs) else np.nan,
            "opp": htid, "season": row["season"],
        }
        if htid not in team_games:
            team_games[htid] = []
        if atid not in team_games:
            team_games[atid] = []
        team_games[htid].append(game_entry_home)
        team_games[atid].append(game_entry_away)

    stats_df = pd.DataFrame(result_rows)
    return pd.concat([games.reset_index(drop=True), stats_df], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# FanGraphs team quality features
# ─────────────────────────────────────────────────────────────────────────────

def _load_fg_team_stats(raw_dir: Path) -> Optional[dict]:
    """
    Load team quality stats. Tries in priority order:
      1. Statcast team pitching/batting aggregates (best)
      2. MLB Stats API team pitching/batting (fallback)
    Returns {"pitching": df, "batting": df} or None.
    """
    pitch_dfs, bat_dfs = [], []

    # Priority 1: Statcast team aggregates
    for f in sorted(raw_dir.glob("statcast_team_pitching_*.parquet")):
        try:
            pitch_dfs.append(pd.read_parquet(f))
        except Exception:
            pass
    for f in sorted(raw_dir.glob("statcast_team_batting_*.parquet")):
        try:
            bat_dfs.append(pd.read_parquet(f))
        except Exception:
            pass

    # Priority 2: MLB API team stats (adds ERA, WHIP, K, BB)
    for f in sorted(raw_dir.glob("mlb_team_pitching_*.parquet")):
        try:
            pitch_dfs.append(pd.read_parquet(f))
        except Exception:
            pass
    for f in sorted(raw_dir.glob("mlb_team_batting_*.parquet")):
        try:
            bat_dfs.append(pd.read_parquet(f))
        except Exception:
            pass

    if not pitch_dfs and not bat_dfs:
        logger.warning("  No team quality data found — run 01_fetch_data.py first")
        return None

    pitch = pd.concat(pitch_dfs, ignore_index=True) if pitch_dfs else pd.DataFrame()
    bat = pd.concat(bat_dfs, ignore_index=True) if bat_dfs else pd.DataFrame()

    logger.info(f"  Team pitching records: {len(pitch)}, batting records: {len(bat)}")
    return {"pitching": pitch, "batting": bat}


def _map_fg_abbr_to_team_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add team_id column to a FanGraphs dataframe using Team abbreviation."""
    FG_TEAM_MAP = {
        "LAA": 108, "ARI": 109, "BAL": 110, "BOS": 111, "CHC": 112,
        "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117,
        "KCR": 118, "LAD": 119, "WSN": 120, "NYM": 121, "OAK": 133,
        "PIT": 134, "SDP": 135, "SEA": 136, "SFG": 137, "STL": 138,
        "TBR": 139, "TEX": 140, "TOR": 141, "MIN": 142, "PHI": 143,
        "ATL": 144, "CHW": 145, "MIA": 146, "NYY": 147, "MIL": 158,
    }
    if "team_id" not in df.columns and "Team" in df.columns:
        df["team_id"] = df["Team"].map(FG_TEAM_MAP)
    return df


def _build_fg_quality_lookup(fg_stats: dict) -> dict:
    """
    Build lookup: {(team_id, season) → {pitching_features..., batting_features...}}

    Accepts data from two sources (both may be present; Statcast values preferred):
      - Statcast team aggregates: sc_sp_xera_pct, sc_sp_k_pct, sc_sp_bb_pct,
                                   sc_sp_whiff_pct, sc_sp_fbv_pct,
                                   sc_bat_xwoba_pct, sc_bat_ev_pct, sc_bat_hh_pct, sc_bat_brl_pct
      - MLB Stats API team stats:  era, whip, strikeOuts, baseOnBalls, inningsPitched (pitching)
                                   obp, slg, ops, avg, homeRuns (batting)
    """
    pitch_df = fg_stats.get("pitching", pd.DataFrame())
    bat_df = fg_stats.get("batting", pd.DataFrame())

    # Column maps: source_col → feature_name
    # We map both Statcast percentile rank cols and MLB API cols in one pass;
    # later values in the loop can overwrite earlier ones, so put Statcast last
    # (higher priority) if both sources exist in the same df.
    PITCH_COLS = {
        # MLB Stats API
        "era":          "fg_era",
        "whip":         "fg_whip",
        "strikeOuts":   "fg_k9",      # raw count — will normalise below
        "baseOnBalls":  "fg_bb9",
        "inningsPitched": "_ip",       # helper for rate calculation
        # Statcast team agg (percentile ranks 0-100)
        "sc_sp_xera_pct":   "fg_xfip",    # xERA pct ≈ xFIP signal
        "sc_sp_k_pct":      "fg_k_pct",
        "sc_sp_bb_pct":     "fg_bb_pct",
        "sc_sp_whiff_pct":  "fg_whiff_pct",
        "sc_sp_fbv_pct":    "fg_fbv_pct",
    }
    BAT_COLS = {
        # MLB Stats API
        "obp":      "fg_obp",
        "slg":      "fg_slg",
        "ops":      "fg_ops",
        "avg":      "fg_avg",
        "homeRuns": "fg_hr",
        # Statcast team agg (percentile ranks 0-100)
        "sc_bat_xwoba_pct": "fg_woba",       # xwOBA pct ≈ wRC+ signal
        "sc_bat_ev_pct":    "fg_ev_pct",
        "sc_bat_hh_pct":    "fg_hh_pct",
        "sc_bat_brl_pct":   "fg_brl_pct",
    }

    lookup: dict = {}

    for df, col_map in [(pitch_df, PITCH_COLS), (bat_df, BAT_COLS)]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            tid = row.get("team_id")
            season = row.get("season")
            if pd.isna(tid) or pd.isna(season):
                continue
            key = (int(tid), int(season))
            if key not in lookup:
                lookup[key] = {}
            for src_col, feat_name in col_map.items():
                if src_col in row.index:
                    val = row[src_col]
                    if isinstance(val, str):
                        try:
                            val = float(val)
                        except Exception:
                            continue
                    if not pd.isna(val):
                        lookup[key][feat_name] = float(val)

            # Convert raw K and BB to per-9 rates using IP
            entry = lookup.get(key, {})
            ip = entry.pop("_ip", None)
            if ip and ip > 0:
                if "fg_k9" in entry:
                    entry["fg_k9"] = entry["fg_k9"] / ip * 9
                if "fg_bb9" in entry:
                    entry["fg_bb9"] = entry["fg_bb9"] / ip * 9

    return lookup


def _attach_fg_features(games: pd.DataFrame, fg_lookup: dict) -> pd.DataFrame:
    """
    Join team quality features onto each game row.
    Source: MLB Stats API (ERA, WHIP, OBP, SLG) + Statcast aggregates (xERA%, K%, wOBA%).
    Strategy: use prior-year stats at season start, blend in current-year
    stats as the season progresses (weighted by games_played / 162).
    """
    logger.info("  Attaching team quality features (MLB API + Statcast)...")

    # All possible feature names produced by _build_fg_quality_lookup
    PITCH_FEATS = ["fg_era", "fg_whip", "fg_k9", "fg_bb9",
                   "fg_xfip", "fg_k_pct", "fg_bb_pct", "fg_whiff_pct", "fg_fbv_pct"]
    BAT_FEATS   = ["fg_obp", "fg_slg", "fg_ops", "fg_avg", "fg_hr",
                   "fg_woba", "fg_ev_pct", "fg_hh_pct", "fg_brl_pct"]
    ALL_FEATS = PITCH_FEATS + BAT_FEATS

    def get_qual(team_id: int, season: int, games_played: int) -> dict:
        if pd.isna(team_id):
            return {}
        prior   = fg_lookup.get((int(team_id), season - 1), {})
        current = fg_lookup.get((int(team_id), season), {})
        if not prior and not current:
            return {}
        w_cur   = min(games_played / 162.0, 1.0)
        w_prior = 1.0 - w_cur
        result  = {}
        for feat in ALL_FEATS:
            p, c = prior.get(feat), current.get(feat)
            if p is not None and c is not None:
                result[feat] = w_prior * p + w_cur * c
            elif c is not None:
                result[feat] = c
            elif p is not None:
                result[feat] = p
        return result

    rows = []
    for _, row in games.iterrows():
        season = int(row["season"])
        htid   = row["home_team_id"]
        atid   = row["away_team_id"]
        h_gp   = int(row.get("home_games_played", 0))
        a_gp   = int(row.get("away_games_played", 0))

        hq = get_qual(htid, season, h_gp)
        aq = get_qual(atid, season, a_gp)

        new_cols = {}
        for feat in ALL_FEATS:
            new_cols[f"home_{feat}"] = hq.get(feat, np.nan)
            new_cols[f"away_{feat}"] = aq.get(feat, np.nan)

        # Differentials (positive = home team advantage)
        # ERA/xERA: lower is better → away_era - home_era
        for f in ["fg_era", "fg_whip", "fg_xfip"]:
            hv = new_cols.get(f"home_{f}", np.nan)
            av = new_cols.get(f"away_{f}", np.nan)
            if not (np.isnan(hv) or np.isnan(av)):
                new_cols[f"{f}_differential"] = av - hv
        # Offense: higher is better → home - away
        for f in ["fg_ops", "fg_woba", "fg_obp", "fg_slg"]:
            hv = new_cols.get(f"home_{f}", np.nan)
            av = new_cols.get(f"away_{f}", np.nan)
            if not (np.isnan(hv) or np.isnan(av)):
                new_cols[f"{f}_differential"] = hv - av

        rows.append(new_cols)

    fg_df = pd.DataFrame(rows)
    return pd.concat([games.reset_index(drop=True), fg_df], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Park factors
# ─────────────────────────────────────────────────────────────────────────────

def _build_park_factor_lookup(raw_dir: Path) -> dict:
    """Load FanGraphs park factors, return {(team_id, season): park_factor}."""
    pf_path = raw_dir / "park_factors.parquet"
    if not pf_path.exists():
        logger.warning("  Park factors not found — using default 100")
        return {}

    try:
        pf = pd.read_parquet(pf_path)
        logger.info(f"  Park factors: {pf.shape}, cols: {list(pf.columns[:10])}")

        # FanGraphs park_factors columns vary by year but typically include:
        # Team, season, Basic, HR, 1B, 2B, 3B, so...
        # We'll use "Basic" (overall run scoring factor) as our main metric
        lookup = {}
        for _, row in pf.iterrows():
            team = row.get("Team", row.get("team", ""))
            season = row.get("season", row.get("Season"))
            basic = row.get("Basic", row.get("wRC", row.get("R", 100)))
            if pd.isna(basic):
                basic = 100

            # Map team name to team_id
            FG_MAP = {
                "LAA": 108, "ARI": 109, "BAL": 110, "BOS": 111, "CHC": 112,
                "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117,
                "KCR": 118, "LAD": 119, "WSN": 120, "NYM": 121, "OAK": 133,
                "PIT": 134, "SDP": 135, "SEA": 136, "SFG": 137, "STL": 138,
                "TBR": 139, "TEX": 140, "TOR": 141, "MIN": 142, "PHI": 143,
                "ATL": 144, "CHW": 145, "MIA": 146, "NYY": 147, "MIL": 158,
            }
            tid = FG_MAP.get(str(team).upper())
            if tid and season:
                lookup[(tid, int(season))] = float(basic)

        return lookup
    except Exception as e:
        logger.warning(f"  Error loading park factors: {e}")
        return {}


def _attach_park_factors(games: pd.DataFrame, pf_lookup: dict) -> pd.DataFrame:
    """Attach home park factor (100=neutral, 105=5% above avg run scoring)."""
    def get_pf(team_id: int, season: int) -> float:
        # Try current year, then average of last 3 years, then default
        for yr in [season, season - 1, season - 2, season - 3]:
            pf = pf_lookup.get((team_id, yr))
            if pf is not None:
                return pf
        return DEFAULT_PARK_FACTOR

    games = games.copy()
    games["home_park_factor"] = games.apply(
        lambda r: get_pf(int(r["home_team_id"]), int(r["season"])), axis=1
    )
    # Normalize to ratio (0-2 range centered at 1.0)
    games["home_park_factor_ratio"] = games["home_park_factor"] / 100.0
    return games


# ─────────────────────────────────────────────────────────────────────────────
# Calendar / context features
# ─────────────────────────────────────────────────────────────────────────────

def _add_calendar_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, month, and game time features."""
    games = games.copy()
    dates = pd.to_datetime(games["game_date"])
    games["day_of_week"] = dates.dt.dayofweek
    games["month"] = dates.dt.month
    games["is_weekend"] = (dates.dt.dayofweek >= 4).astype(int)

    # One-hot months (April=4 through October=10)
    for m in range(3, 11):
        games[f"month_{m}"] = (games["month"] == m).astype(int)

    return games


# ─────────────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────────────

def load_all_games(raw_dir: Path) -> pd.DataFrame:
    """Load and combine all game records: 2015-2024 CSV + 2025/2026 parquets."""
    all_dfs = []

    # 2015-2024 from backend CSV
    if GAMES_CSV.exists():
        games_csv = pd.read_csv(GAMES_CSV)
        games_csv = games_csv[games_csv["status_code"] == "F"].copy()  # Final only
        games_csv = games_csv[games_csv["game_type"] == "R"].copy()    # Regular season only
        games_csv["home_won"] = (games_csv["home_score"] > games_csv["away_score"]).astype(int)
        # CSV has both 'year' and 'season'; use 'season' (already present)
        games_csv["game_date"] = games_csv["game_date"].astype(str)
        all_dfs.append(games_csv[[
            "game_pk", "game_date", "season", "home_team_id", "away_team_id",
            "home_score", "away_score", "home_won",
        ]])
        logger.info(f"  games_2015_2024.csv: {len(games_csv)} games")
    else:
        logger.warning(f"  games_2015_2024.csv not found at {GAMES_CSV}")

    KEEP_COLS = ["game_pk", "game_date", "season", "home_team_id", "away_team_id",
                 "home_score", "away_score", "home_won"]

    # Additional years from MLB API parquets (catches any year not in CSV)
    # The CSV covers 2015-2024; MLB API parquets cover 2020-2026.
    # We deduplicate at the end, so overlapping years are fine.
    for f in sorted(raw_dir.glob("games_*.parquet")):
        try:
            df = pd.read_parquet(f)
            df["game_date"] = df["game_date"].astype(str)
            needed = ["game_date", "season", "home_team_id", "away_team_id", "home_won"]
            if not all(c in df.columns for c in needed):
                logger.warning(f"  {f.name}: missing cols {[c for c in needed if c not in df.columns]}")
                continue
            for c in KEEP_COLS:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[KEEP_COLS].copy()
            df = df.dropna(subset=["home_team_id", "away_team_id"])
            all_dfs.append(df)
            logger.info(f"  {f.name}: {len(df)} games")
        except Exception as e:
            logger.warning(f"  {f.name}: {e}")

    if not all_dfs:
        raise RuntimeError("No game data found! Run 01_fetch_data.py first.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["game_date"] = combined["game_date"].astype(str)
    combined["season"] = combined["season"].astype(int)
    combined["home_team_id"] = combined["home_team_id"].astype(int)
    combined["away_team_id"] = combined["away_team_id"].astype(int)
    combined["home_won"] = combined["home_won"].astype(int)

    # Deduplicate (keep latest)
    if "game_pk" in combined.columns:
        combined = combined.drop_duplicates(subset=["game_pk"], keep="last")
    else:
        combined = combined.drop_duplicates(subset=["game_date", "home_team_id", "away_team_id"], keep="last")

    combined = combined.sort_values("game_date").reset_index(drop=True)
    logger.info(f"  Total games: {len(combined)} ({combined['season'].min()}-{combined['season'].max()})")
    return combined


def build_features(rebuild: bool = False) -> dict[str, pd.DataFrame]:
    """Main entry point. Returns {split_name: DataFrame}."""
    
    # Check cache
    splits = {}
    cache_files = {
        "train": FEATURES_DIR / "train_v9.parquet",
        "dev": FEATURES_DIR / "dev_v9.parquet",
        "val": FEATURES_DIR / "val_v9.parquet",
        "test_2026": FEATURES_DIR / "test_2026_v9.parquet",
    }

    if not rebuild and all(f.exists() for f in cache_files.values()):
        logger.info("Loading cached V9 features...")
        for split, path in cache_files.items():
            splits[split] = pd.read_parquet(path)
            logger.info(f"  {split}: {splits[split].shape}")
        return splits

    logger.info("Building V9 feature dataset from scratch...")
    logger.info("=" * 50)

    # 1. Load all games
    logger.info("[1/6] Loading game data...")
    games = load_all_games(RAW_DIR)

    # 2. Elo
    logger.info("[2/6] Computing Elo ratings...")
    games = compute_elo_ratings(games)

    # 3. Rolling stats
    logger.info("[3/6] Building rolling stats...")
    games = _rolling_team_stats(games)

    # 4. FanGraphs quality features
    logger.info("[4/6] Attaching FanGraphs team quality features...")
    fg_stats = _load_fg_team_stats(RAW_DIR)
    if fg_stats:
        fg_lookup = _build_fg_quality_lookup(fg_stats)
        games = _attach_fg_features(games, fg_lookup)
        logger.info(f"  FG features added; fg_lookup entries: {len(fg_lookup)}")
    else:
        logger.warning("  No FanGraphs data — FG quality features will be NaN")

    # 5. Park factors
    logger.info("[5/6] Attaching park factors...")
    pf_lookup = _build_park_factor_lookup(RAW_DIR)
    games = _attach_park_factors(games, pf_lookup)

    # 6. Calendar features
    logger.info("[6/6] Adding calendar features...")
    games = _add_calendar_features(games)

    # Drop rows with no outcome
    games = games.dropna(subset=["home_won"])
    games["home_won"] = games["home_won"].astype(int)

    # Log feature count
    feature_cols = [c for c in games.columns if c not in {
        "home_won", "game_pk", "game_date", "home_team_id", "away_team_id",
        "home_score", "away_score", "home_team_name", "away_team_name",
        "season", "_type", "fetch_team", "home_team_fg", "away_team_fg",
    }]
    logger.info(f"\nTotal features: {len(feature_cols)}")
    logger.info(f"Feature list (first 30): {feature_cols[:30]}")

    # ── Split by year ──
    splits = {
        "train": games[games["season"] <= 2023].copy(),
        "dev":   games[games["season"] == 2024].copy(),
        "val":   games[games["season"] == 2025].copy(),
        "test_2026": games[games["season"] == 2026].copy(),
    }

    # Save
    for split_name, df in splits.items():
        path = cache_files[split_name]
        df.to_parquet(path, index=False)
        logger.info(f"  Saved {split_name}: {df.shape} → {path}")

    # Save feature metadata
    feature_groups = {
        "elo": [c for c in feature_cols if "elo" in c],
        "pythag": [c for c in feature_cols if "pythag" in c or "luck" in c],
        "rolling_form": [c for c in feature_cols if any(x in c for x in ["win_pct", "run_diff", "streak", "scoring_mom"])],
        "fg_pitching": [c for c in feature_cols if "fg_" in c and any(x in c for x in ["era", "fip", "xfip", "whip", "k_pct", "bb_pct", "hr9", "lob"])],
        "fg_batting": [c for c in feature_cols if "fg_" in c and any(x in c for x in ["wrc", "woba", "obp", "slg", "iso", "bat_k", "bat_bb"])],
        "park": [c for c in feature_cols if "park" in c],
        "calendar": [c for c in feature_cols if any(x in c for x in ["month", "day_of_week", "weekend", "season_pct"])],
        "context": [c for c in feature_cols if any(x in c for x in ["divisional", "h2h", "game_number", "late_season", "early_season"])],
    }
    meta = {
        "total_features": len(feature_cols),
        "all_features": feature_cols,
        "groups": feature_groups,
        "splits": {k: len(v) for k, v in splits.items()},
    }
    with open(FEATURES_DIR / "feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("\n" + "=" * 50)
    logger.info("V9 FEATURE BUILD COMPLETE")
    logger.info(f"  Train: {len(splits['train'])} games (2015-2023)")
    logger.info(f"  Dev:   {len(splits['dev'])} games (2024)")
    logger.info(f"  Val:   {len(splits['val'])} games (2025 holdout)")
    logger.info(f"  Test:  {len(splits['test_2026'])} games (2026 YTD)")
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info("\nNext step: python 03_train_v9_experiment.py")
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if cached")
    args = parser.parse_args()
    build_features(rebuild=args.rebuild)
