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
from espn_data import load_games, resolve_team_divisions  # noqa: E402

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

    # Team divisions come from resolve_team_divisions, not from each game's own
    # `division`: that column names the ESPN feed the game was fetched from, so both
    # sides of a cross-division game carry the same tag and comparing them always
    # yields 0. The cross_division flag is derived at ingest from the fact that ESPN
    # lists such a game in both feeds.
    div_of = resolve_team_divisions(games)
    feats["home_division"] = feats["home_team"].map(div_of)
    feats["away_division"] = feats["away_team"].map(div_of)

    if "cross_division" in games.columns:
        feats = feats.merge(
            games[["game_id", "cross_division"]], on="game_id", how="left"
        )
        feats["cross_division"] = feats["cross_division"].fillna(0).astype(int)
    else:
        # Pre-v2 caches have no ingest-time flag; the resolved divisions still give a
        # correct answer, which the old same-column comparison did not.
        feats["cross_division"] = (
            feats["home_division"] != feats["away_division"]
        ).astype(int)
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


def predict_week(season: int, week: int) -> pd.DataFrame:
    """Predict a scheduled (not yet played) week for both divisions.

    Mirrors the NFL path: team state is built from every completed game, the unplayed
    slate is appended so it inherits that state without contributing to it, and each
    division is predicted by a model trained only on its own division's history.
    """
    from espn_data import fetch_scheduled

    played = load_games()
    upcoming = fetch_scheduled(season, week)
    if upcoming.empty:
        logger.warning("no scheduled games for %d week %d", season, week)
        return pd.DataFrame()

    # Drop any that already finished — those belong to the backfill path.
    upcoming = upcoming[upcoming["home_won"].isna()].copy()
    if upcoming.empty:
        logger.info("every %d wk%d game already final", season, week)
        return pd.DataFrame()

    upcoming["game_date"] = pd.to_datetime(upcoming["game_date"])
    combined = pd.concat([played, upcoming], ignore_index=True)
    feats = build(combined)

    cols = cfb_feature_columns(feats)
    out = []
    for division in ("fbs", "fcs"):
        train = feats[(feats["division"] == division) & feats["home_won"].notna()]
        target = feats[(feats["division"] == division) & feats["home_won"].isna()
                       & (feats["season"] == season) & (feats["week"] == week)]
        if len(train) < 300 or target.empty:
            continue

        model = build_xgb()
        model.fit(train[cols].fillna(0.0).values, train["home_won"].astype(int).values)
        proba = model.predict_proba(target[cols].fillna(0.0).values)[:, 1]

        out.append(pd.DataFrame({
            "game_id": target["game_id"].values,
            "season": target["season"].values,
            "week": target["week"].values,
            "division": division,
            "game_date": pd.to_datetime(target["game_date"].values),
            "home_team_id": target["home_team"].values,
            "away_team_id": target["away_team"].values,
            "home_team_name": target["home_team_name"].values,
            "away_team_name": target["away_team_name"].values,
            "home_win_probability": proba,
            "away_win_probability": 1 - proba,
            "predicted_winner": np.where(proba > 0.5, target["home_team_name"].values,
                                         target["away_team_name"].values),
            "confidence_tier": [confidence_tier(p) for p in proba],
            "model_version": MODEL_VERSION,
            "predicted_at": datetime.now(timezone.utc),
            "elo_differential": target["elo_differential"].values,
            "elo_home_win_prob": target["elo_home_win_prob"].values,
            "pythag_differential": target["pythag_differential"].values,
            "home_point_diff_3g": target["home_point_diff_3g"].values,
            "away_point_diff_3g": target["away_point_diff_3g"].values,
            "home_current_streak": target["home_current_streak"].values,
            "away_current_streak": target["away_current_streak"].values,
            "is_divisional": target["is_divisional"].values,
            "cross_division": target["cross_division"].values,
            "neutral_site": target["neutral_site"].values,
            "home_won": None,
            "actual_winner": None,
            "prediction_correct": None,
        }))
        logger.info("%s %d wk%d: %d predictions", division, season, week, len(out[-1]))

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def next_unplayed_week(season: int, max_week: int = 21) -> int | None:
    """Earliest week in the season that still has games without a result.

    Lets the weekly scheduler fire a bare {"mode": "predict_next"} instead of hardcoding
    a week number that would go stale after seven days.
    """
    from espn_data import fetch_scheduled

    for week in range(1, max_week + 1):
        slate = fetch_scheduled(season, week)
        if slate.empty:
            continue
        if slate["home_won"].isna().any():
            return week
    return None
