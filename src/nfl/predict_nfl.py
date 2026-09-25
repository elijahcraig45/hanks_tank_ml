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

from bq_io import ensure_dataset, load_table, replace_seasons, upsert_week  # noqa: E402
from config import CTX  # noqa: E402
from data import completed_games, load_schedules  # noqa: E402
from features import build_features, feature_columns  # noqa: E402
from train_nfl_models import build_xgb  # noqa: E402
import margin_ridge as mr  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Bumped from nfl_v1_pure: same XGBoost, but now actually fed the EPA block it was
# validated with. Rows written before the fix carry the old name and NULL EPA.
MODEL_VERSION = "nfl_v1_pure_epa"
RIDGE_MODEL_VERSION = "nfl_v2_margin_ridge"

# The ridge runs as a SHADOW model into its own table, read by nothing, until it has
# earned a place in game_predictions.
SHADOW_TABLE = "game_predictions_ridge_shadow"

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


def load_epa(source: str = "auto") -> pd.DataFrame | None:
    """Per-team, per-week EPA aggregates, read from where the ingest writes them.

    The weekly ingest persists team_week_epa to BigQuery (nfl_historical, 2006 onward,
    refreshed one season at a time). This used to read only the local parquet cache,
    which in Cloud Functions lives in /tmp and is empty on every cold container — so
    predict_week silently trained and predicted WITHOUT EPA, and every stored 2024-2026
    NFL prediction has net_epa_8g NULL. The backtests that justified the EPA features
    had them; production never did. Classic train/serve skew.

    source: "auto" (BigQuery, then local cache), "bq", or "cache". Reading the
    aggregate is ~11k rows; nothing here rebuilds play-by-play.
    """
    if source in ("auto", "bq"):
        try:
            from bq_io import query

            df = query(
                f"SELECT * FROM `{CTX.project}.{CTX.hist_dataset}.team_week_epa`")
            if not df.empty:
                logger.info("EPA from BigQuery: %d rows, seasons %d-%d", len(df),
                            df["season"].min(), df["season"].max())
                return df
        except Exception as exc:
            if source == "bq":
                raise
            logger.warning("EPA BigQuery read failed (%s); trying local cache", exc)
    from epa import EPA_CACHE

    if EPA_CACHE.exists():
        return pd.read_parquet(EPA_CACHE)
    return None


def _require_epa(epa: pd.DataFrame | None, season: int) -> pd.DataFrame:
    """Refuse to predict without EPA rather than quietly dropping the feature block.

    Seasons before the first EPA season are dropped by the caller (nflverse pbp starts
    in 2006). A season that should have EPA but does not is an ingest failure, and a
    model silently trained on a different feature set than it was validated on is
    exactly the bug this replaces.
    """
    if epa is None or epa.empty:
        raise RuntimeError("no team_week_epa available (BigQuery and local cache both "
                           "empty) — refusing to predict without the EPA features")
    have = set(int(s) for s in epa["season"].unique())
    if season - 1 not in have:
        raise RuntimeError(f"team_week_epa has no {season - 1} rows; run the ingest "
                           f"for that season before predicting {season}")
    return epa


def _epa_era(games: pd.DataFrame, epa: pd.DataFrame) -> pd.DataFrame:
    """Keep games from the first EPA season on. Not `isin(epa seasons)`: that dropped
    any season whose EPA had not been ingested yet, current season included."""
    return games[games["season"] >= int(epa["season"].min())]


def ridge_frame(games: pd.DataFrame, feats: pd.DataFrame | None = None,
                epa: pd.DataFrame | None = None,
                epa_cfg: "mr.RidgeConfig | None" = None) -> pd.DataFrame:
    """Games in the shape margin_ridge expects, plus the candidate adjustments.

      rest_diff        home rest minus away rest, days, clipped to +-10
      net_epa_8g       rolling 8-game net EPA/play differential (from `feats`)
      epa_rating_diff  an opponent-adjusted EPA rating: a margin ridge fit to each
                       game's EPA margin (home minus away offensive EPA/play, x65
                       plays so it is in points), walk-forward, so every row carries
                       only its own pre-game estimate
    """
    rg = games[["game_id", "season", "week", "home_team", "away_team"]].copy()
    rg["margin"] = pd.to_numeric(games["result"], errors="coerce")
    rg["neutral"] = (games["location"].fillna("Home") != "Home").astype(int)
    rest = (pd.to_numeric(games["home_rest"], errors="coerce")
            - pd.to_numeric(games["away_rest"], errors="coerce"))
    rg["rest_diff"] = rest.clip(-10, 10).fillna(0.0)
    rg = rg.reset_index(drop=True)

    if feats is not None and "net_epa_8g" in feats.columns:
        rg = rg.merge(feats[["game_id", "net_epa_8g"]], on="game_id", how="left")

    if epa is not None and len(epa):
        off = epa.set_index(["season", "week", "team"])["off_epa_play"]
        h = off.reindex(pd.MultiIndex.from_frame(
            rg[["season", "week", "home_team"]])).to_numpy()
        a = off.reindex(pd.MultiIndex.from_frame(
            rg[["season", "week", "away_team"]])).to_numpy()
        em = rg[["game_id", "season", "week", "home_team", "away_team", "neutral"]].copy()
        em["margin"] = (h - a) * 65.0
        cfg = epa_cfg or mr.NFL_RIDGE.with_(covariates=())
        rg["epa_rating_diff"] = mr.walk_forward(em, np.ones(len(em), bool), cfg)
    return rg


def build_prediction_rows(feats: pd.DataFrame, proba: np.ndarray,
                          actuals: pd.DataFrame | None = None,
                          model_version: str = MODEL_VERSION,
                          margin: np.ndarray | None = None,
                          ridge: "mr.MarginRidge | None" = None) -> pd.DataFrame:
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
        "model_version": model_version,
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

    # Ridge-only, additive. Positive = home favoured, the same sign as spread_line.
    if margin is not None:
        rows["predicted_home_margin"] = np.asarray(margin, dtype=float)
    if ridge is not None:
        rows["home_power_rating"] = feats["home_team"].map(ridge.ratings).fillna(0.0).values
        rows["away_power_rating"] = feats["away_team"].map(ridge.ratings).fillna(0.0).values
        rows["home_field_points"] = ridge.hfa * (
            feats["is_neutral_site"].fillna(0).values == 0)

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


def backfill_season(season: int, model: str = "xgb",
                    cfg: "mr.RidgeConfig" = mr.NFL_RIDGE) -> pd.DataFrame:
    """Week-by-week honest out-of-sample predictions for a completed season.

    model="xgb" is production. model="ridge" is the shadow margin ridge, which needs
    no EPA and trains on the decaying two-season window before each week.
    """
    games = completed_games()
    if model == "ridge":
        feats = build_features(games, epa=None)
        rg = ridge_frame(games)
        target = (rg["season"] == season).to_numpy()
        margin = mr.walk_forward(rg, target, cfg)
        pred = rg.loc[target, ["game_id"]].assign(margin=margin[target]).dropna()
        test = feats.merge(pred, on="game_id")
        result = build_prediction_rows(
            test, mr.win_prob(test["margin"].values, cfg.sigma), actuals=games,
            model_version=RIDGE_MODEL_VERSION, margin=test["margin"].values)
        logger.info("ridge backfilled %d predictions for %d", len(result), season)
        return result

    epa = _require_epa(load_epa(), season)
    games = _epa_era(games, epa)

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

        xgb = build_xgb()
        xgb.fit(train[cols].fillna(0.0).values, train["home_won"].values)
        proba = xgb.predict_proba(test[cols].fillna(0.0).values)[:, 1]
        out.append(build_prediction_rows(test, proba, actuals=games))

    result = pd.concat(out, ignore_index=True)
    logger.info("backfilled %d predictions for %d", len(result), season)
    return result


def _kickoff_utc(schedule: pd.DataFrame) -> pd.Series:
    """nflverse gameday + gametime are US/Eastern wall-clock; a missing time is read
    as midnight, so an undated game is never treated as not yet started."""
    local = pd.to_datetime(schedule["gameday"].astype(str) + " "
                           + schedule.get("gametime", pd.Series("00:00", index=schedule.index))
                           .fillna("00:00").astype(str), errors="coerce")
    return local.dt.tz_localize("America/New_York", ambiguous="NaT",
                                nonexistent="NaT").dt.tz_convert("UTC")


def _upcoming(schedule: pd.DataFrame, season: int, week: int,
              now: pd.Timestamp | None = None) -> pd.DataFrame:
    upcoming = schedule[(schedule["season"] == season) & (schedule["week"] == week)].copy()
    if upcoming.empty:
        raise SystemExit(f"no scheduled games found for {season} week {week}")
    # Only games that have not kicked off. A rerun after Thursday used to re-predict
    # the week's finished games, writing a post-kickoff row over the real pregame one
    # (and duplicating that game in the ridge frame).
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    started = upcoming["result"].notna() if "result" in upcoming else False
    started = started | (_kickoff_utc(upcoming) <= now).fillna(False)
    upcoming = upcoming[~started].copy()
    if upcoming.empty:
        raise SystemExit(f"every {season} week {week} game has already kicked off")
    upcoming["home_won"] = np.nan
    upcoming["game_date"] = pd.to_datetime(upcoming["gameday"])
    return upcoming


def predict_week(season: int, week: int, model: str = "xgb",
                 cfg: "mr.RidgeConfig" = mr.NFL_RIDGE,
                 played: pd.DataFrame | None = None,
                 schedule: pd.DataFrame | None = None,
                 epa: pd.DataFrame | None = None) -> pd.DataFrame:
    """Predict a scheduled (not yet played) week — the weekly cron path.

    Trains on every completed game, then emits features for the upcoming slate by
    running the same chronological pass with the unplayed games appended. Unplayed
    rows produce features but do not update team state.

    model="xgb" (production) needs EPA and refuses to run without it. model="ridge"
    (shadow) needs only scores.
    """
    played = completed_games() if played is None else played
    schedule = load_schedules() if schedule is None else schedule
    upcoming = _upcoming(schedule, season, week)

    if model == "ridge":
        combined = pd.concat([played, upcoming], ignore_index=True)
        feats = build_features(combined, epa=None)
        rg = ridge_frame(combined)
        ridge = mr.fit(rg, mr.time_index(season, week), cfg)
        if ridge is None:
            raise RuntimeError(f"ridge: too few games before {season} week {week}")
        target = feats[feats["home_won"].isna()]
        tgt = target[["game_id"]].merge(rg, on="game_id", how="left")
        margin = ridge.predict_margin(tgt)
        logger.info("ridge predicted %d games for %d week %d", len(target), season, week)
        return build_prediction_rows(target, mr.win_prob(margin, cfg.sigma),
                                     model_version=RIDGE_MODEL_VERSION,
                                     margin=margin, ridge=ridge)

    epa = _require_epa(load_epa() if epa is None else epa, season)
    played = _epa_era(played, epa)
    lag = _epa_lag_weeks(played, epa, season)
    if lag:
        logger.warning("team_week_epa is %d week(s) behind the %d results — the rolling "
                       "EPA features are stale for this prediction", lag, season)

    combined = pd.concat([played, upcoming], ignore_index=True)
    feats = build_features(combined, epa=epa)

    train = feats[feats["home_won"].notna()]
    target = feats[feats["home_won"].isna()]
    cols = feature_columns(feats, include_market=False)

    xgb = build_xgb()
    xgb.fit(train[cols].fillna(0.0).values, train["home_won"].astype(int).values)
    proba = xgb.predict_proba(target[cols].fillna(0.0).values)[:, 1]

    logger.info("predicted %d games for %d week %d", len(target), season, week)
    return build_prediction_rows(target, proba)


def _epa_lag_weeks(played: pd.DataFrame, epa: pd.DataFrame, season: int) -> int:
    """How many played weeks of `season` have no EPA rows yet."""
    done = set(played.loc[played["season"] == season, "week"].astype(int))
    have = set(epa.loc[epa["season"] == season, "week"].astype(int))
    return len(done - have)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, help="season to backfill week by week")
    ap.add_argument("--season", type=int, help="season to predict (with --week)")
    ap.add_argument("--week", type=int, help="week to predict (with --season)")
    ap.add_argument("--model", choices=["xgb", "ridge"], default="xgb",
                    help="ridge is the shadow model and writes only to "
                         f"{SHADOW_TABLE}")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    table = SHADOW_TABLE if args.model == "ridge" else "game_predictions"

    if args.season and args.week:
        rows = predict_week(args.season, args.week, model=args.model)
        print(f"\n{args.season} week {args.week}: {len(rows)} predictions")
        print(rows[["away_team_name", "home_team_name", "home_win_probability",
                    "predicted_winner", "confidence_tier"]].to_string(index=False))
        if not args.no_write:
            ensure_dataset(CTX.season_dataset)
            upsert_week(rows, CTX.season_dataset, table, args.season, args.week)
        return 0

    if not args.backfill:
        ap.error("pass either --backfill SEASON or --season S --week W")

    rows = backfill_season(args.backfill, model=args.model)

    acc = rows["prediction_correct"].mean()
    print(f"\n{args.backfill}: {len(rows)} predictions, accuracy {acc:.2%}")
    print(rows.groupby("confidence_tier")["prediction_correct"]
          .agg(["count", "mean"]).to_string())

    if not args.no_write:
        ensure_dataset(CTX.season_dataset)
        if args.model == "ridge":
            replace_seasons(rows, CTX.season_dataset, table,
                            partition_field="game_date", cluster_fields=["season", "week"])
        else:
            load_table(rows, CTX.season_dataset, table,
                       write_disposition="WRITE_TRUNCATE",
                       partition_field="game_date",
                       cluster_fields=["season", "week"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
