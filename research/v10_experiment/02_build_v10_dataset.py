#!/usr/bin/env python3
"""
V10 Feature Builder — Game-Level SP Quality + Park Factors + Rest/Travel

Builds on the full V9 feature set and adds three new feature groups:

  1. GAME-LEVEL SP QUALITY (primary new signal)
     For each game, looks up the specific starter's Statcast percentile
     ranks (xERA%, K%, BB%, whiff%, FB velocity%) rather than team averages.
     Features: home_sp_xera, away_sp_xera, sp_xera_diff,
               home_sp_k_pct, away_sp_k_pct, sp_k_diff,
               home_sp_bb_pct, away_sp_bb_pct, sp_bb_diff,
               home_sp_whiff, away_sp_whiff, sp_whiff_diff,
               home_sp_fbv, away_sp_fbv, sp_fbv_diff,
               sp_quality_composite_diff,
               home_sp_known, away_sp_known  (missingness flags)

  2. VENUE-BASED PARK FACTORS (replaces default-100 proxy)
     Uses actual run-scoring per venue computed from 2015-2024 game data.
     Features: home_park_factor (ratio), home_park_factor_100 (0-200 scale)

  3. REST & TRAVEL FEATURES (no new API calls)
     Computed from game dates per team.
     Features: home_days_rest, away_days_rest, rest_differential,
               home_road_trip_length, away_road_trip_length,
               series_game_number, games_in_series

Output:
  data/v10/features/train_v10.parquet       2015-2023 (training)
  data/v10/features/dev_v10.parquet         2024 (dev/calibration)
  data/v10/features/val_v10.parquet         2025 (validation holdout)
  data/v10/features/test_2026_v10.parquet   2026 YTD (live test)
  data/v10/features/feature_metadata.json

Usage:
    python 02_build_v10_dataset.py
    python 02_build_v10_dataset.py --rebuild
"""

import argparse
import json
import logging
import sys
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

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
V9_RAW_DIR  = REPO_ROOT / "data" / "v9" / "raw"
V10_RAW_DIR = REPO_ROOT / "data" / "v10" / "raw"
FEATURES_DIR = REPO_ROOT / "data" / "v10" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

GAMES_CSV = REPO_ROOT.parent / "hanks_tank_backend" / "data" / "games" / "games_2015_2024.csv"

# Add V9 builder to path so we can import shared helpers
V9_DIR = REPO_ROOT / "research" / "v9_experiment"
sys.path.insert(0, str(V9_DIR))

# ── Elo constants (same as V9) ─────────────────────────────────────────────
ELO_START           = 1500.0
ELO_K               = 20.0
ELO_HOME_BONUS      = 70.0
ELO_SEASON_REGRESSION = 0.33

# ── SP quality league-average defaults (used when starter unknown) ─────────
# These are 50th-percentile values for the Statcast percentile rank scale (0-100)
SP_DEFAULT_XERA   = 50.0
SP_DEFAULT_K_PCT  = 50.0
SP_DEFAULT_BB_PCT = 50.0
SP_DEFAULT_WHIFF  = 50.0
SP_DEFAULT_FBV    = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Elo engine (identical to V9)
# ─────────────────────────────────────────────────────────────────────────────

def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def compute_elo_ratings(games: pd.DataFrame) -> pd.DataFrame:
    logger.info("  Computing Elo ratings...")
    games = games.sort_values("game_date").copy()
    elo: dict[int, float] = {}
    processed_seasons: set = set()
    home_elos, away_elos, home_elo_win_probs = [], [], []

    for _, row in games.iterrows():
        season = row["season"]
        htid = int(row["home_team_id"])
        atid = int(row["away_team_id"])

        if season not in processed_seasons:
            processed_seasons.add(season)
            for tid in list(elo.keys()):
                elo[tid] = ELO_START + (elo[tid] - ELO_START) * (1.0 - ELO_SEASON_REGRESSION)

        elo.setdefault(htid, ELO_START)
        elo.setdefault(atid, ELO_START)

        h_elo = elo[htid] + ELO_HOME_BONUS
        a_elo = elo[atid]
        win_prob = elo_expected(h_elo, a_elo)

        home_elos.append(elo[htid])
        away_elos.append(elo[atid])
        home_elo_win_probs.append(win_prob)

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
# Rolling game-by-game stats (identical to V9)
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_team_stats(games: pd.DataFrame) -> pd.DataFrame:
    logger.info("  Building rolling team stats...")
    games = games.sort_values("game_date").reset_index(drop=True)
    team_games: dict[int, list] = {}
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

            total_rs = sum(rs_col) if rs_col else 1.0
            total_ra = sum(ra_col) if ra_col else 1.0
            rs_exp = max(total_rs, 1e-6) ** 1.83
            ra_exp = max(total_ra, 1e-6) ** 1.83
            pythag = rs_exp / (rs_exp + ra_exp)
            actual_wp = win_pct(0) if won_col else 0.5
            luck = actual_wp - pythag

            season = row["season"]
            season_hist = [g for g in history if g.get("season") == season]
            if season_hist:
                sr = sum(g["rs"] for g in season_hist if not np.isnan(g["rs"]))
                sa = sum(g["ra"] for g in season_hist if not np.isnan(g["ra"]))
                se = max(sr, 1e-6) ** 1.83 / (max(sr, 1e-6) ** 1.83 + max(sa, 1e-6) ** 1.83)
            else:
                se = 0.5

            sr30 = sum(rs_col[-30:]) if len(rs_col) >= 5 else total_rs
            sa30 = sum(ra_col[-30:]) if len(ra_col) >= 5 else total_ra
            p30e = max(sr30, 1e-6) ** 1.83
            p30a = max(sa30, 1e-6) ** 1.83
            pythag_30g = p30e / (p30e + p30a)

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
                f"{pfx}_era_proxy_10g": avg_runs_allowed(10),
                f"{pfx}_era_proxy_30g": avg_runs_allowed(30),
                f"{pfx}_scoring_momentum": avg_runs_scored(7) - avg_runs_scored(30),
                f"{pfx}_h2h_win_pct_season": h2h_pct_season,
                f"{pfx}_h2h_games_season": h2h_games_season,
                f"{pfx}_h2h_win_pct_3yr": h2h_pct_3yr,
            }

        home_stats = get_stats(htid, is_home=True)
        away_stats = get_stats(atid, is_home=False)
        merged = {**home_stats, **away_stats}

        merged["elo_win_prob_differential"] = row.get("elo_home_win_prob", 0.5) - 0.5
        merged["pythag_differential"] = merged["home_pythag_season"] - merged["away_pythag_season"]
        merged["run_diff_differential"] = merged["home_run_diff_10g"] - merged["away_run_diff_10g"]
        merged["luck_differential"] = merged["home_luck_factor"] - merged["away_luck_factor"]
        merged["win_pct_diff"] = merged["home_win_pct_season"] - merged["away_win_pct_season"]
        merged["streak_differential"] = (
            merged["home_current_streak"] * merged["home_streak_direction"]
            - merged["away_current_streak"] * merged["away_streak_direction"]
        )
        merged["era_proxy_differential"] = merged["away_era_proxy_10g"] - merged["home_era_proxy_10g"]

        season_games = [g for g in team_games.get(htid, []) if g.get("season") == row["season"]]
        games_in_season = len(season_games)
        merged["season_game_number"] = games_in_season
        merged["season_pct_complete"] = min(games_in_season / 162.0, 1.0)
        merged["is_late_season"] = int(games_in_season > 120)
        merged["is_early_season"] = int(games_in_season < 30)

        result_rows.append(merged)

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
# FanGraphs / MLB API team quality (identical to V9)
# ─────────────────────────────────────────────────────────────────────────────

def _load_fg_team_stats(raw_dir: Path) -> Optional[dict]:
    pitch_dfs, bat_dfs = [], []
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
        return None
    pitch = pd.concat(pitch_dfs, ignore_index=True) if pitch_dfs else pd.DataFrame()
    bat = pd.concat(bat_dfs, ignore_index=True) if bat_dfs else pd.DataFrame()
    logger.info(f"  Team pitching records: {len(pitch)}, batting records: {len(bat)}")
    return {"pitching": pitch, "batting": bat}


def _build_fg_quality_lookup(fg_stats: dict) -> dict:
    pitch_df = fg_stats.get("pitching", pd.DataFrame())
    bat_df   = fg_stats.get("batting",  pd.DataFrame())

    PITCH_COLS = {
        "era": "fg_era", "whip": "fg_whip",
        "strikeOuts": "fg_k9", "baseOnBalls": "fg_bb9", "inningsPitched": "_ip",
        "sc_sp_xera_pct": "fg_xfip", "sc_sp_k_pct": "fg_k_pct",
        "sc_sp_bb_pct": "fg_bb_pct", "sc_sp_whiff_pct": "fg_whiff_pct",
        "sc_sp_fbv_pct": "fg_fbv_pct",
    }
    BAT_COLS = {
        "obp": "fg_obp", "slg": "fg_slg", "ops": "fg_ops", "avg": "fg_avg", "homeRuns": "fg_hr",
        "sc_bat_xwoba_pct": "fg_woba", "sc_bat_ev_pct": "fg_ev_pct",
        "sc_bat_hh_pct": "fg_hh_pct", "sc_bat_brl_pct": "fg_brl_pct",
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

            entry = lookup.get(key, {})
            ip = entry.pop("_ip", None)
            if ip and ip > 0:
                if "fg_k9" in entry:
                    entry["fg_k9"] = entry["fg_k9"] / ip * 9
                if "fg_bb9" in entry:
                    entry["fg_bb9"] = entry["fg_bb9"] / ip * 9

    return lookup


def _attach_fg_features(games: pd.DataFrame, fg_lookup: dict) -> pd.DataFrame:
    logger.info("  Attaching team quality features (MLB API + Statcast)...")
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
        for f in ["fg_era", "fg_whip", "fg_xfip"]:
            hv = new_cols.get(f"home_{f}", np.nan)
            av = new_cols.get(f"away_{f}", np.nan)
            if not (np.isnan(hv) or np.isnan(av)):
                new_cols[f"{f}_differential"] = av - hv
        for f in ["fg_ops", "fg_woba", "fg_obp", "fg_slg"]:
            hv = new_cols.get(f"home_{f}", np.nan)
            av = new_cols.get(f"away_{f}", np.nan)
            if not (np.isnan(hv) or np.isnan(av)):
                new_cols[f"{f}_differential"] = hv - av
        rows.append(new_cols)

    fg_df = pd.DataFrame(rows)
    return pd.concat([games.reset_index(drop=True), fg_df], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# NEW V10: Game-level SP quality (Statcast percentile ranks)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sp_quality_lookup(raw_dir: Path) -> dict:
    """
    Build lookup: {(player_id, season): {xera, k_pct, bb_pct, whiff, fbv}}
    from statcast_sp_pct_{year}.parquet files (V9 raw dir).

    Statcast percentile ranks are 0-100 where:
      - xera: 99 = pitcher allows very little expected runs (BEST = HIGH)
      - k_percent: 99 = pitcher strikes out most hitters (BEST = HIGH)
      - bb_percent: 99 = pitcher walks fewest hitters (BEST = HIGH — stat flipped by Statcast)
      - whiff_percent: 99 = pitcher generates most whiffs (BEST = HIGH)
      - fb_velocity: 99 = hardest fastball (BEST = HIGH)
    All are oriented so higher = better pitcher. Differentials are home - away.
    """
    lookup: dict = {}
    years = list(range(2015, 2027))

    for year in years:
        # Try V9 raw dir first, then V10 raw dir
        for raw_d in [raw_dir, V10_RAW_DIR]:
            f = raw_d / f"statcast_sp_pct_{year}.parquet"
            if not f.exists():
                continue
            try:
                df = pd.read_parquet(f)
                # Columns: player_id, year (or season), xera, k_percent, bb_percent,
                #          whiff_percent, fb_velocity
                season_col = "season" if "season" in df.columns else "year"
                for _, row in df.iterrows():
                    pid = row.get("player_id")
                    s   = row.get(season_col)
                    if pd.isna(pid) or pd.isna(s):
                        continue
                    key = (int(pid), int(s))
                    lookup[key] = {
                        "xera":   float(row["xera"])          if not pd.isna(row.get("xera"))          else np.nan,
                        "k_pct":  float(row["k_percent"])     if not pd.isna(row.get("k_percent"))     else np.nan,
                        "bb_pct": float(row["bb_percent"])    if not pd.isna(row.get("bb_percent"))    else np.nan,
                        "whiff":  float(row["whiff_percent"]) if not pd.isna(row.get("whiff_percent")) else np.nan,
                        "fbv":    float(row["fb_velocity"])   if not pd.isna(row.get("fb_velocity"))   else np.nan,
                    }
            except Exception as e:
                logger.warning(f"  Error reading {f.name}: {e}")
            break  # Found file in this dir, don't try next dir

    logger.info(f"  SP quality lookup: {len(lookup)} pitcher-season entries")
    return lookup


def _build_game_starters_lookup(raw_dir: Path) -> dict:
    """
    Build lookup: {game_pk: {home_sp_id, away_sp_id, venue_id,
                              series_game_number, games_in_series}}
    from game_starters_{year}.parquet files (V10 raw dir).
    """
    lookup: dict = {}
    for f in sorted(raw_dir.glob("game_starters_*.parquet")):
        try:
            df = pd.read_parquet(f)
            for _, row in df.iterrows():
                gpk = row.get("game_pk")
                if pd.isna(gpk):
                    continue
                lookup[int(gpk)] = {
                    "home_sp_id":         int(row["home_sp_id"]) if not pd.isna(row.get("home_sp_id")) else None,
                    "away_sp_id":         int(row["away_sp_id"]) if not pd.isna(row.get("away_sp_id")) else None,
                    "venue_id":           int(row["venue_id"])   if not pd.isna(row.get("venue_id"))   else None,
                    "series_game_number": int(row["series_game_number"]) if not pd.isna(row.get("series_game_number")) else 1,
                    "games_in_series":    int(row["games_in_series"])    if not pd.isna(row.get("games_in_series"))    else 3,
                }
        except Exception as e:
            logger.warning(f"  Error reading {f.name}: {e}")

    logger.info(f"  Game starters lookup: {len(lookup)} games")
    return lookup


def _attach_sp_quality(
    games: pd.DataFrame,
    sp_lookup: dict,
    gs_lookup: dict,
) -> pd.DataFrame:
    """
    For each game, look up the specific starter via gs_lookup[game_pk],
    then look up their Statcast percentile rank via sp_lookup[(player_id, season)].

    Falls back to prior-year data when current-year data not yet available.
    Imputes league average (50th pct) for unknown starters.

    New columns:
        home_sp_xera, away_sp_xera, sp_xera_diff
        home_sp_k_pct, away_sp_k_pct, sp_k_diff
        home_sp_bb_pct, away_sp_bb_pct, sp_bb_diff
        home_sp_whiff, away_sp_whiff, sp_whiff_diff
        home_sp_fbv, away_sp_fbv, sp_fbv_diff
        sp_quality_composite_diff  (weighted average of all 5 differentials)
        home_sp_known, away_sp_known  (1 = starter ID found + has Statcast data)
        series_game_number, games_in_series  (from game_starters)
    """
    logger.info("  Attaching game-level SP quality features...")

    METRICS = ["xera", "k_pct", "bb_pct", "whiff", "fbv"]
    DEFAULTS = {
        "xera": SP_DEFAULT_XERA, "k_pct": SP_DEFAULT_K_PCT,
        "bb_pct": SP_DEFAULT_BB_PCT, "whiff": SP_DEFAULT_WHIFF,
        "fbv": SP_DEFAULT_FBV,
    }
    # Composite weights: xERA is most predictive, velocity least
    COMPOSITE_WEIGHTS = {"xera": 0.35, "k_pct": 0.25, "bb_pct": 0.20, "whiff": 0.15, "fbv": 0.05}

    def get_sp_stats(player_id: Optional[int], season: int) -> tuple[dict, bool]:
        """Return (stats_dict, is_known). Tries current year then prior year."""
        if player_id is None:
            return {m: DEFAULTS[m] for m in METRICS}, False

        for yr in [season, season - 1]:
            entry = sp_lookup.get((player_id, yr))
            if entry is not None:
                # Fill any NaN metrics with defaults
                result = {}
                any_real = False
                for m in METRICS:
                    v = entry.get(m, np.nan)
                    if not np.isnan(v):
                        result[m] = v
                        any_real = True
                    else:
                        result[m] = DEFAULTS[m]
                return result, any_real

        return {m: DEFAULTS[m] for m in METRICS}, False

    rows = []
    missing_game_pk = 0
    missing_sp = {"home": 0, "away": 0}

    for _, row in games.iterrows():
        season = int(row["season"])
        game_pk = int(row["game_pk"]) if not pd.isna(row.get("game_pk")) else None

        gs = gs_lookup.get(game_pk, {}) if game_pk else {}
        home_sp_id = gs.get("home_sp_id")
        away_sp_id = gs.get("away_sp_id")

        if not gs:
            missing_game_pk += 1

        home_stats, home_known = get_sp_stats(home_sp_id, season)
        away_stats, away_known = get_sp_stats(away_sp_id, season)

        if not home_known:
            missing_sp["home"] += 1
        if not away_known:
            missing_sp["away"] += 1

        new_cols = {}

        # Raw percentile rank features
        for m in METRICS:
            new_cols[f"home_sp_{m}"] = home_stats[m]
            new_cols[f"away_sp_{m}"] = away_stats[m]
            new_cols[f"sp_{m}_diff"] = home_stats[m] - away_stats[m]  # positive = home SP better

        # Composite quality differential (higher = home SP significantly better)
        new_cols["sp_quality_composite_diff"] = sum(
            COMPOSITE_WEIGHTS[m] * new_cols[f"sp_{m}_diff"] for m in METRICS
        )

        # Missingness flags (the model can learn to discount these rows)
        new_cols["home_sp_known"] = int(home_known)
        new_cols["away_sp_known"] = int(away_known)

        # Series context
        new_cols["series_game_number"] = gs.get("series_game_number", 1)
        new_cols["games_in_series"] = gs.get("games_in_series", 3)
        new_cols["is_series_opener"] = int(gs.get("series_game_number", 1) == 1)

        # Venue ID (for park factor join)
        new_cols["venue_id"] = gs.get("venue_id")

        rows.append(new_cols)

    logger.info(f"  SP quality: {missing_game_pk} games missing game_pk in starters lookup")
    logger.info(f"  SP quality: home SP unknown={missing_sp['home']}, away SP unknown={missing_sp['away']}")

    sp_df = pd.DataFrame(rows)
    return pd.concat([games.reset_index(drop=True), sp_df], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# NEW V10: Venue-based park factors
# ─────────────────────────────────────────────────────────────────────────────

def _build_venue_park_factor_lookup(raw_dir: Path) -> dict:
    """
    Build lookup: {venue_id: park_factor_ratio}
    park_factor_ratio: 1.0 = neutral, 1.10 = 10% above average run scoring.
    """
    for d in [raw_dir, V10_RAW_DIR, V9_RAW_DIR]:
        pf_path = d / "park_factors.parquet"
        if pf_path.exists():
            try:
                pf = pd.read_parquet(pf_path)
                lookup = {}
                for _, row in pf.iterrows():
                    vid = row.get("venue_id")
                    pf_ratio = row.get("park_factor")  # already a ratio (1.273 = 27% above avg)
                    pf_100   = row.get("park_factor_100")  # 100-scale
                    if not pd.isna(vid) and not pd.isna(pf_ratio):
                        lookup[int(vid)] = {
                            "ratio": float(pf_ratio),
                            "pf100": float(pf_100) if not pd.isna(pf_100) else float(pf_ratio) * 100,
                        }
                logger.info(f"  Park factors loaded: {len(lookup)} venues from {pf_path}")
                return lookup
            except Exception as e:
                logger.warning(f"  Error loading park factors from {pf_path}: {e}")
    logger.warning("  Park factors not found — using neutral 1.0")
    return {}


def _attach_venue_park_factors(games: pd.DataFrame, pf_lookup: dict) -> pd.DataFrame:
    """
    Join venue park factors. Uses venue_id column (added by _attach_sp_quality).
    Falls back to 1.0 (neutral) for unknown venues.
    """
    logger.info("  Attaching venue-based park factors...")
    pf_ratios, pf_100s, pf_known = [], [], []

    for _, row in games.iterrows():
        vid = row.get("venue_id")
        entry = pf_lookup.get(int(vid), {}) if (vid is not None and not pd.isna(vid)) else {}
        pf_ratios.append(entry.get("ratio", 1.0))
        pf_100s.append(entry.get("pf100", 100.0))
        pf_known.append(int(bool(entry)))

    games = games.copy()
    games["home_park_factor"]     = pf_ratios       # 1.273 for Coors, 0.887 for AT&T Park
    games["home_park_factor_100"] = pf_100s         # 127 for Coors
    games["park_factor_known"]    = pf_known        # 1 if venue matched

    # Hitter-friendliness indicator
    games["is_hitter_park"] = (games["home_park_factor"] > 1.05).astype(int)
    games["is_pitcher_park"] = (games["home_park_factor"] < 0.95).astype(int)

    known = sum(pf_known)
    logger.info(f"  Park factor assigned: {known}/{len(games)} games ({100*known/len(games):.1f}%)")
    return games


# ─────────────────────────────────────────────────────────────────────────────
# NEW V10: Rest & travel features
# ─────────────────────────────────────────────────────────────────────────────

def _add_rest_travel_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest and travel features from game dates.
    For each game, looks back at the team's previous game to compute:
      - days_rest: calendar days since last game (capped at 7)
      - road_trip_length: consecutive away games (0 if team is at home)

    Also uses venue_id to track if team changed location (proxy for travel).

    New columns:
        home_days_rest, away_days_rest, rest_differential
        home_road_trip_length, away_road_trip_length
    """
    logger.info("  Computing rest & travel features...")
    games = games.sort_values("game_date").copy()
    games["game_date_dt"] = pd.to_datetime(games["game_date"])

    # State per team
    last_game_date: dict[int, pd.Timestamp] = {}
    road_trip_len: dict[int, int] = {}   # consecutive away games

    home_rest_vals, away_rest_vals = [], []
    home_road_vals, away_road_vals = [], []

    for _, row in games.iterrows():
        htid = int(row["home_team_id"])
        atid = int(row["away_team_id"])
        gdate = row["game_date_dt"]

        def get_days_rest(team_id: int) -> float:
            last = last_game_date.get(team_id)
            if last is None:
                return 3.0  # assume moderate rest at season start
            delta = (gdate - last).days
            return float(min(delta, 7))  # cap at 7 (longer rest uncommon/special)

        def get_road_trip(team_id: int, is_home: bool) -> int:
            if is_home:
                return 0  # home team always at home
            return road_trip_len.get(team_id, 0)

        home_rest = get_days_rest(htid)
        away_rest = get_days_rest(atid)
        home_road = get_road_trip(htid, is_home=True)
        away_road = get_road_trip(atid, is_home=False)

        home_rest_vals.append(home_rest)
        away_rest_vals.append(away_rest)
        home_road_vals.append(home_road)
        away_road_vals.append(away_road)

        # Update state
        last_game_date[htid] = gdate
        last_game_date[atid] = gdate

        # Home team resets road trip counter; away team increments
        road_trip_len[htid] = 0
        road_trip_len[atid] = road_trip_len.get(atid, 0) + 1

    games["home_days_rest"]         = home_rest_vals
    games["away_days_rest"]         = away_rest_vals
    games["rest_differential"]      = games["home_days_rest"] - games["away_days_rest"]
    games["home_road_trip_length"]  = home_road_vals   # always 0 (home team is home)
    games["away_road_trip_length"]  = away_road_vals

    # Binary flags
    games["home_rested"]  = (games["home_days_rest"] >= 2).astype(int)
    games["away_tired"]   = (games["away_days_rest"] <= 1).astype(int)
    games["long_road_trip"] = (games["away_road_trip_length"] >= 5).astype(int)

    games = games.drop(columns=["game_date_dt"])
    logger.info(f"  Rest/travel features added. Avg away road trip: {games['away_road_trip_length'].mean():.1f} games")
    return games


# ─────────────────────────────────────────────────────────────────────────────
# Calendar features (identical to V9)
# ─────────────────────────────────────────────────────────────────────────────

def _add_calendar_features(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    dates = pd.to_datetime(games["game_date"])
    games["day_of_week"] = dates.dt.dayofweek
    games["month"] = dates.dt.month
    games["is_weekend"] = (dates.dt.dayofweek >= 4).astype(int)
    for m in range(3, 11):
        games[f"month_{m}"] = (games["month"] == m).astype(int)
    return games


# ─────────────────────────────────────────────────────────────────────────────
# Game loader (identical to V9)
# ─────────────────────────────────────────────────────────────────────────────

def load_all_games() -> pd.DataFrame:
    all_dfs = []

    if GAMES_CSV.exists():
        games_csv = pd.read_csv(GAMES_CSV)
        games_csv = games_csv[games_csv["status_code"] == "F"].copy()
        games_csv = games_csv[games_csv["game_type"] == "R"].copy()
        games_csv["home_won"] = (games_csv["home_score"] > games_csv["away_score"]).astype(int)
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

    for f in sorted(V9_RAW_DIR.glob("games_*.parquet")):
        try:
            df = pd.read_parquet(f)
            df["game_date"] = df["game_date"].astype(str)
            needed = ["game_date", "season", "home_team_id", "away_team_id", "home_won"]
            if not all(c in df.columns for c in needed):
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
    combined["game_date"]    = combined["game_date"].astype(str)
    combined["season"]       = combined["season"].astype(int)
    combined["home_team_id"] = combined["home_team_id"].astype(int)
    combined["away_team_id"] = combined["away_team_id"].astype(int)
    combined["home_won"]     = combined["home_won"].astype(int)

    if "game_pk" in combined.columns:
        combined = combined.drop_duplicates(subset=["game_pk"], keep="last")
    else:
        combined = combined.drop_duplicates(subset=["game_date", "home_team_id", "away_team_id"], keep="last")

    combined = combined.sort_values("game_date").reset_index(drop=True)
    logger.info(f"  Total games: {len(combined)} ({combined['season'].min()}-{combined['season'].max()})")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────────────

def build_features(rebuild: bool = False) -> dict[str, pd.DataFrame]:
    cache_files = {
        "train":     FEATURES_DIR / "train_v10.parquet",
        "dev":       FEATURES_DIR / "dev_v10.parquet",
        "val":       FEATURES_DIR / "val_v10.parquet",
        "test_2026": FEATURES_DIR / "test_2026_v10.parquet",
    }

    if not rebuild and all(f.exists() for f in cache_files.values()):
        logger.info("Loading cached V10 features...")
        splits = {}
        for split, path in cache_files.items():
            splits[split] = pd.read_parquet(path)
            logger.info(f"  {split}: {splits[split].shape}")
        return splits

    logger.info("Building V10 feature dataset from scratch...")
    logger.info("=" * 60)

    # ── 1. Load games ──────────────────────────────────────────────────────
    logger.info("[1/9] Loading game data...")
    games = load_all_games()

    # ── 2. Elo ─────────────────────────────────────────────────────────────
    logger.info("[2/9] Computing Elo ratings...")
    games = compute_elo_ratings(games)

    # ── 3. Rolling stats ───────────────────────────────────────────────────
    logger.info("[3/9] Building rolling team stats...")
    games = _rolling_team_stats(games)

    # ── 4. Team quality (MLB API + Statcast aggregates) ────────────────────
    logger.info("[4/9] Attaching team quality features...")
    fg_stats = _load_fg_team_stats(V9_RAW_DIR)
    if fg_stats:
        fg_lookup = _build_fg_quality_lookup(fg_stats)
        games = _attach_fg_features(games, fg_lookup)
        logger.info(f"  FG lookup entries: {len(fg_lookup)}")
    else:
        logger.warning("  No team quality data — FG features will be NaN")

    # ── 5. Game-level SP quality (NEW in V10) ──────────────────────────────
    logger.info("[5/9] Attaching game-level SP quality features (V10 NEW)...")
    sp_lookup = _build_sp_quality_lookup(V9_RAW_DIR)
    gs_lookup = _build_game_starters_lookup(V10_RAW_DIR)
    games = _attach_sp_quality(games, sp_lookup, gs_lookup)

    # ── 6. Venue-based park factors (NEW in V10) ───────────────────────────
    logger.info("[6/9] Attaching venue-based park factors (V10 NEW)...")
    pf_lookup = _build_venue_park_factor_lookup(V10_RAW_DIR)
    games = _attach_venue_park_factors(games, pf_lookup)

    # ── 7. Rest & travel features (NEW in V10) ────────────────────────────
    logger.info("[7/9] Computing rest & travel features (V10 NEW)...")
    games = _add_rest_travel_features(games)

    # ── 8. Calendar ────────────────────────────────────────────────────────
    logger.info("[8/9] Adding calendar features...")
    games = _add_calendar_features(games)

    # ── 9. Finalize ────────────────────────────────────────────────────────
    logger.info("[9/9] Finalizing splits...")
    games = games.dropna(subset=["home_won"])
    games["home_won"] = games["home_won"].astype(int)

    EXCLUDE_COLS = {
        "home_won", "game_pk", "game_date", "home_team_id", "away_team_id",
        "home_score", "away_score", "home_team_name", "away_team_name",
        "season", "_type", "fetch_team", "home_team_fg", "away_team_fg",
        "venue_id",  # keep as metadata but not a feature
    }
    feature_cols = [c for c in games.columns if c not in EXCLUDE_COLS]

    logger.info(f"\nTotal features: {len(feature_cols)} (V9 had ~121)")

    # Count new V10 features
    v10_new = [c for c in feature_cols if any(x in c for x in [
        "sp_xera", "sp_k_pct", "sp_bb_pct", "sp_whiff", "sp_fbv",
        "sp_quality", "sp_known", "series_game", "games_in_series",
        "is_series_opener", "days_rest", "road_trip", "rested", "tired",
        "long_road", "park_factor", "hitter_park", "pitcher_park",
        "park_factor_known",
    ])]
    logger.info(f"V10 new features: {len(v10_new)}")
    logger.info(f"V10 new feature names: {v10_new}")

    # Splits
    splits = {
        "train":     games[games["season"] <= 2023].copy(),
        "dev":       games[games["season"] == 2024].copy(),
        "val":       games[games["season"] == 2025].copy(),
        "test_2026": games[games["season"] == 2026].copy(),
    }

    for split_name, df in splits.items():
        path = cache_files[split_name]
        df.to_parquet(path, index=False)
        logger.info(f"  Saved {split_name}: {df.shape} -> {path}")

    # Feature metadata
    feature_groups = {
        "elo":              [c for c in feature_cols if "elo" in c],
        "pythag":           [c for c in feature_cols if "pythag" in c or "luck" in c],
        "rolling_form":     [c for c in feature_cols if any(x in c for x in ["win_pct", "run_diff", "streak", "scoring_mom"])],
        "team_pitching":    [c for c in feature_cols if "fg_" in c and any(x in c for x in ["era", "xfip", "whip", "k_pct", "bb_pct", "fbv"])],
        "team_batting":     [c for c in feature_cols if "fg_" in c and any(x in c for x in ["obp", "slg", "ops", "woba", "ev", "hh", "brl"])],
        "sp_quality":       [c for c in feature_cols if "sp_" in c],
        "park_factors":     [c for c in feature_cols if "park_factor" in c or "hitter_park" in c or "pitcher_park" in c],
        "rest_travel":      [c for c in feature_cols if any(x in c for x in ["days_rest", "road_trip", "rested", "tired", "long_road"])],
        "series_context":   [c for c in feature_cols if any(x in c for x in ["series_game", "games_in_series", "is_series"])],
        "calendar":         [c for c in feature_cols if any(x in c for x in ["month", "day_of_week", "weekend", "season_pct"])],
        "context":          [c for c in feature_cols if any(x in c for x in ["h2h", "game_number", "late_season", "early_season"])],
    }

    meta = {
        "total_features": len(feature_cols),
        "all_features": feature_cols,
        "v10_new_features": v10_new,
        "groups": feature_groups,
        "splits": {k: len(v) for k, v in splits.items()},
        "description": {
            "sp_quality": "Game-level starter Statcast percentile ranks (xERA, K%, BB%, Whiff%, FBV). Higher = better pitcher.",
            "park_factors": "Venue-based park factors from 2015-2024 run-scoring. 1.0=neutral, 1.27=Coors Field.",
            "rest_travel": "Days rest between games, road trip length in consecutive away games.",
        }
    }
    with open(FEATURES_DIR / "feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("V10 FEATURE BUILD COMPLETE")
    logger.info(f"  Train: {len(splits['train'])} games (2015-2023)")
    logger.info(f"  Dev:   {len(splits['dev'])} games (2024)")
    logger.info(f"  Val:   {len(splits['val'])} games (2025 holdout)")
    logger.info(f"  Test:  {len(splits['test_2026'])} games (2026 YTD)")
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  New V10 features: {len(v10_new)}")
    logger.info("\nNext step: python 03_train_v10_experiment.py")
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if cached")
    args = parser.parse_args()
    build_features(rebuild=args.rebuild)
