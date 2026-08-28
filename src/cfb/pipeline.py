"""College football feature build, training and prediction.

Reuses the NFL feature builder and model harness rather than duplicating them — the
Elo/Pythagorean/form/streak/H2H machinery is genuinely sport-agnostic, and the pieces
that differ (Elo constants, MOV damping, season regression) are injected as EloParams.
This is the reuse the second sport was supposed to prove out.

FBS and FCS are modelled SEPARATELY but share one Elo pool. Cross-division games are
the only edges connecting the two populations, so splitting the pool would leave them
mutually uncalibrated and make those games unpredictable.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the NFL modules; they are the shared football core in practice.
NFL_DIR = Path(__file__).resolve().parents[1] / "nfl"
sys.path.insert(0, str(NFL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import EloParams, build_features, feature_columns  # noqa: E402
from models import build_xgb, evaluate  # noqa: E402

import cfb_config  # noqa: E402
from espn_data import load_games  # noqa: E402

logger = logging.getLogger(__name__)

CFB_ELO = EloParams(
    start=cfb_config.ELO_START,
    k=cfb_config.ELO_K,
    home_bonus=cfb_config.ELO_HOME_BONUS,
    season_regression=cfb_config.ELO_SEASON_REGRESSION,
    pythag_exponent=cfb_config.PYTHAG_EXPONENT,
    mov_damping=cfb_config.ELO_MOV_DAMPING,
)

MODEL_VERSION = "cfb_v1"
CONFIDENCE_TIERS = {"high": 0.80, "medium": 0.65}  # wider than NFL: bigger mismatches


def confidence_tier(p: float) -> str:
    edge = abs(p - 0.5) + 0.5
    if edge >= CONFIDENCE_TIERS["high"]:
        return "high"
    if edge >= CONFIDENCE_TIERS["medium"]:
        return "medium"
    return "low"


def normalize_games(df: pd.DataFrame) -> pd.DataFrame:
    """Map CFB fields onto the column names the shared feature builder expects."""
    g = df.copy()
    g["game_type"] = np.where(g["is_postseason"] == 1, "POST", "REG")
    g["div_game"] = g["conference_game"]
    g["location"] = np.where(g["neutral_site"] == 1, "Neutral", "Home")
    g["spread_line"] = np.nan   # ESPN's free endpoint carries no betting lines
    g["total_line"] = np.nan

    # Rest days: not supplied by ESPN, so derive from each team's previous game.
    g = g.sort_values(["game_date"]).reset_index(drop=True)
    last_seen: dict[str, pd.Timestamp] = {}
    home_rest, away_rest = [], []
    for r in g.itertuples(index=False):
        for team, bucket in ((r.home_team, home_rest), (r.away_team, away_rest)):
            prev = last_seen.get(team)
            bucket.append((r.game_date - prev).days if prev is not None else np.nan)
        last_seen[r.home_team] = r.game_date
        last_seen[r.away_team] = r.game_date
    g["home_rest"] = home_rest
    g["away_rest"] = away_rest
    return g


def build(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature table over BOTH divisions — one shared Elo pool."""
    games = normalize_games(df if df is not None else load_games())
    feats = build_features(games, epa=None, elo=CFB_ELO)

    # Carry division and cross-division context onto the feature rows.
    meta = games[["game_id", "division", "neutral_site", "is_postseason",
                  "home_team_name", "away_team_name"]]
    feats = feats.merge(meta, on="game_id", how="left")

    div_of = {}
    for r in games.itertuples(index=False):
        div_of[r.home_team] = r.division
        div_of[r.away_team] = r.division
    feats["home_division"] = feats["home_team"].map(div_of)
    feats["away_division"] = feats["away_team"].map(div_of)
    feats["cross_division"] = (feats["home_division"] != feats["away_division"]).astype(int)
    return feats


CFB_NON_FEATURE = {
    "division", "home_division", "away_division",
    "home_team_name", "away_team_name",
}


def cfb_feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(feats, include_market=False)
            if c not in CFB_NON_FEATURE]


def backfill_division(feats: pd.DataFrame, division: str, season: int) -> pd.DataFrame:
    """Honest week-by-week out-of-sample predictions for one division and season."""
    cols = cfb_feature_columns(feats)
    subset = feats[feats["division"] == division]

    out = []
    for wk in sorted(subset[subset["season"] == season]["week"].unique()):
        # Train on every prior game in this division, across all seasons.
        train = subset[(subset["season"] < season)
                       | ((subset["season"] == season) & (subset["week"] < wk))]
        test = subset[(subset["season"] == season) & (subset["week"] == wk)]
        if len(train) < 300 or test.empty:
            continue

        model = build_xgb()
        model.fit(train[cols].fillna(0.0).values, train["home_won"].astype(int).values)
        proba = model.predict_proba(test[cols].fillna(0.0).values)[:, 1]

        rows = pd.DataFrame({
            "game_id": test["game_id"].values,
            "season": test["season"].values,
            "week": test["week"].values,
            "division": division,
            "game_date": pd.to_datetime(test["game_date"].values),
            "home_team_id": test["home_team"].values,
            "away_team_id": test["away_team"].values,
            "home_team_name": test["home_team_name"].values,
            "away_team_name": test["away_team_name"].values,
            "home_win_probability": proba,
            "away_win_probability": 1 - proba,
            "predicted_winner": np.where(proba > 0.5,
                                         test["home_team_name"].values,
                                         test["away_team_name"].values),
            "confidence_tier": [confidence_tier(p) for p in proba],
            "model_version": MODEL_VERSION,
            "predicted_at": datetime.now(timezone.utc),
            "elo_differential": test["elo_differential"].values,
            "elo_home_win_prob": test["elo_home_win_prob"].values,
            "pythag_differential": test["pythag_differential"].values,
            "home_point_diff_3g": test["home_point_diff_3g"].values,
            "away_point_diff_3g": test["away_point_diff_3g"].values,
            "home_current_streak": test["home_current_streak"].values,
            "away_current_streak": test["away_current_streak"].values,
            "is_divisional": test["is_divisional"].values,
            "cross_division": test["cross_division"].values,
            "neutral_site": test["neutral_site"].values,
            "home_won": test["home_won"].values,
        })
        rows["actual_winner"] = np.where(rows["home_won"] == 1,
                                         rows["home_team_name"], rows["away_team_name"])
        rows["prediction_correct"] = (
            (rows["home_win_probability"] > 0.5).astype(int) == rows["home_won"]
        ).astype(int)
        out.append(rows)

    if not out:
        return pd.DataFrame()
    result = pd.concat(out, ignore_index=True)
    logger.info("%s %d: %d predictions, %.2f%% accurate",
                division, season, len(result),
                100 * result["prediction_correct"].mean())
    return result


def baselines(feats: pd.DataFrame, division: str, season: int) -> dict:
    s = feats[(feats["division"] == division) & (feats["season"] == season)]
    if s.empty:
        return {}
    elo_pred = (s["elo_home_win_prob"] > 0.5).astype(int)
    return {
        "games": len(s),
        "always_home": float(s["home_won"].mean()),
        "elo_only": float((elo_pred == s["home_won"]).mean()),
    }
