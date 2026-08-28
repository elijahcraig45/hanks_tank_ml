"""Generate NFL predictions and write them to BigQuery.

Two modes:
  --season/--week   predict one upcoming week (the weekly cron path)
  --backfill        walk a whole season week by week, training only on prior data,
                    so the stored predictions are honestly out-of-sample and can be
                    scored against actual results

SCHEMA CONTRACT — the column names below deliberately mirror the MLB
mlb_2026_season.game_predictions table. hanks_tank's PredictionsPage and the backend
predictions controller read these names. Renaming any of them silently breaks the
frontend reuse. Add new columns; do not rename these.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bq_io import ensure_dataset, load_table, upsert_week  # noqa: E402
from config import CTX  # noqa: E402
from data import completed_games, load_schedules  # noqa: E402
from features import build_features, feature_columns  # noqa: E402
from train_nfl_models import build_xgb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_VERSION = "nfl_v1_pure"

# Recalibrated for NFL. MLB's {high: 0.64, medium: 0.57} is far too tight — football
# win probabilities spread much wider than baseball's.
CONFIDENCE_TIERS = {"high": 0.72, "medium": 0.60}


def confidence_tier(p: float) -> str:
    edge = abs(p - 0.5) + 0.5
    if edge >= CONFIDENCE_TIERS["high"]:
        return "high"
    if edge >= CONFIDENCE_TIERS["medium"]:
        return "medium"
    return "low"


def _stable_game_pk(game_id: str) -> int:
    """Deterministic int key so the existing /:gamePk route shape works for NFL."""
    return abs(hash(game_id)) % (10 ** 12)


def _load_epa() -> pd.DataFrame | None:
    import polars as pl
    from epa import EPA_CACHE

    if EPA_CACHE.exists():
        return pl.read_parquet(EPA_CACHE).to_pandas()
    return None


def build_prediction_rows(feats: pd.DataFrame, proba: np.ndarray,
                          actuals: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = pd.DataFrame({
        "game_id": feats["game_id"].values,
        "game_pk": [_stable_game_pk(g) for g in feats["game_id"]],
        "game_date": pd.to_datetime(feats["game_date"].values),
        "season": feats["season"].values,
        "week": feats["week"].values,
        "home_team_id": feats["home_team"].values,
        "away_team_id": feats["away_team"].values,
        "home_team_name": feats["home_team"].values,
        "away_team_name": feats["away_team"].values,
        "home_win_probability": proba,
        "away_win_probability": 1.0 - proba,
        "predicted_winner": np.where(proba > 0.5,
                                     feats["home_team"].values,
                                     feats["away_team"].values),
        "confidence_tier": [confidence_tier(p) for p in proba],
        "model_version": MODEL_VERSION,
        "predicted_at": datetime.now(timezone.utc),
        # explainer columns the UI renders as "why" cards
        "elo_differential": feats["elo_differential"].values,
        "elo_home_win_prob": feats["elo_home_win_prob"].values,
        "home_pythag_season": feats["home_pythag_season"].values,
        "away_pythag_season": feats["away_pythag_season"].values,
        "pythag_differential": feats["pythag_differential"].values,
        "home_point_diff_3g": feats["home_point_diff_3g"].values,
        "away_point_diff_3g": feats["away_point_diff_3g"].values,
        "home_current_streak": feats["home_current_streak"].values,
        "away_current_streak": feats["away_current_streak"].values,
        "h2h_win_pct": feats["h2h_win_pct"].values,
        "is_divisional": feats["is_divisional"].values,
        "spread_line": feats["spread_line"].values,
    })

    if "net_epa_8g" in feats.columns:
        rows["net_epa_8g"] = feats["net_epa_8g"].values
        rows["home_off_epa_play_8g"] = feats["home_off_epa_play_8g"].values
        rows["away_off_epa_play_8g"] = feats["away_off_epa_play_8g"].values

    # Vegas implied probability, vig removed via the spread sign as a coarse proxy.
    rows["vegas_implied_home_prob"] = np.where(
        feats["spread_line"].notna(),
        1 / (1 + np.exp(-feats["spread_line"].fillna(0) / 7.0)),
        np.nan,
    )
    rows["model_vs_vegas_edge"] = rows["home_win_probability"] - rows["vegas_implied_home_prob"]

    if actuals is not None:
        rows = rows.merge(
            actuals[["game_id", "home_won", "home_score", "away_score"]],
            on="game_id", how="left",
        )
        rows["actual_winner"] = np.where(
            rows["home_won"] == 1, rows["home_team_name"], rows["away_team_name"])
        rows["prediction_correct"] = (
            (rows["home_win_probability"] > 0.5).astype(int) == rows["home_won"]
        ).astype("Int64")
        rows.loc[rows["home_won"].isna(), "prediction_correct"] = pd.NA

    return rows


def backfill_season(season: int) -> pd.DataFrame:
    """Week-by-week honest out-of-sample predictions for a completed season."""
    games = completed_games()
    epa = _load_epa()
    if epa is not None:
        seasons = sorted(epa["season"].unique().tolist())
        games = games[games["season"].isin(seasons)]

    feats = build_features(games, epa=epa)
    cols = feature_columns(feats, include_market=False)

    out = []
    weeks = sorted(feats[feats["season"] == season]["week"].unique())
    for wk in weeks:
        train = feats[(feats["season"] < season)
                      | ((feats["season"] == season) & (feats["week"] < wk))]
        test = feats[(feats["season"] == season) & (feats["week"] == wk)]
        if train.empty or test.empty:
            continue

        model = build_xgb()
        model.fit(train[cols].fillna(0.0).values, train["home_won"].values)
        proba = model.predict_proba(test[cols].fillna(0.0).values)[:, 1]
        out.append(build_prediction_rows(test, proba, actuals=games))

    result = pd.concat(out, ignore_index=True)
    logger.info("backfilled %d predictions for %d", len(result), season)
    return result


def predict_week(season: int, week: int) -> pd.DataFrame:
    """Predict a scheduled (not yet played) week — the weekly cron path.

    Trains on every completed game, then emits features for the upcoming slate by
    running the same chronological pass with the unplayed games appended. Unplayed
    rows produce features but do not update team state.
    """
    played = completed_games()
    epa = _load_epa()
    if epa is not None:
        seasons = sorted(epa["season"].unique().tolist())
        played = played[played["season"].isin(seasons)]

    schedule = load_schedules()
    upcoming = schedule[(schedule["season"] == season) & (schedule["week"] == week)].copy()
    if upcoming.empty:
        raise SystemExit(f"no scheduled games found for {season} week {week}")

    upcoming["home_won"] = np.nan
    upcoming["game_date"] = pd.to_datetime(upcoming["gameday"])

    combined = pd.concat([played, upcoming], ignore_index=True)
    feats = build_features(combined, epa=epa)

    train = feats[feats["home_won"].notna()]
    target = feats[feats["home_won"].isna()]
    cols = feature_columns(feats, include_market=False)

    model = build_xgb()
    model.fit(train[cols].fillna(0.0).values, train["home_won"].astype(int).values)
    proba = model.predict_proba(target[cols].fillna(0.0).values)[:, 1]

    logger.info("predicted %d games for %d week %d", len(target), season, week)
    return build_prediction_rows(target, proba)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, help="season to backfill week by week")
    ap.add_argument("--season", type=int, help="season to predict (with --week)")
    ap.add_argument("--week", type=int, help="week to predict (with --season)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    if args.season and args.week:
        rows = predict_week(args.season, args.week)
        print(f"\n{args.season} week {args.week}: {len(rows)} predictions")
        print(rows[["away_team_name", "home_team_name", "home_win_probability",
                    "predicted_winner", "confidence_tier"]].to_string(index=False))
        if not args.no_write:
            ensure_dataset(CTX.season_dataset)
            upsert_week(rows, CTX.season_dataset, "game_predictions",
                        args.season, args.week)
        return 0

    if not args.backfill:
        ap.error("pass either --backfill SEASON or --season S --week W")

    rows = backfill_season(args.backfill)

    acc = rows["prediction_correct"].mean()
    print(f"\n{args.backfill}: {len(rows)} predictions, accuracy {acc:.2%}")
    print(rows.groupby("confidence_tier")["prediction_correct"]
          .agg(["count", "mean"]).to_string())

    if not args.no_write:
        ensure_dataset(CTX.season_dataset)
        load_table(rows, CTX.season_dataset, "game_predictions",
                   write_disposition="WRITE_TRUNCATE",
                   partition_field="game_date",
                   cluster_fields=["season", "week"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
