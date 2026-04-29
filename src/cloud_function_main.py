#!/usr/bin/env python3
"""
Cloud Function entry point for the 2026 daily pipeline.

Triggered by Cloud Scheduler via HTTP. Runs:
  1. Data collection (yesterday's games, stats, standings, statcast)
  2. Validation
  3. Feature rebuild (V3/V4 rolling features)
  4. Elo update (V8 Elo ratings from yesterday's outcomes)
  5. V8 feature rebuild (Pythagorean, run diff, streaks, H2H → game_v8_features)
  6. Weekly: rosters refresh, batch predictions

Supported modes (passed in request body as JSON):
  daily               Full daily pipeline (steps 1-5 + conditionals)
  backfill            Historical data collection for a date range
  features            Rebuild V3/V4 game_features only
  v8_features         Build V8 features for today's games only
  update_elo          Update Elo ratings from yesterday's game outcomes
  pregame_v8          Per-game: lineups → matchup → V7 → V8 → predict → report (RECOMMENDED V8)
  pregame_v10         Per-game: lineups → matchup → V7 → V8 → V10 → predict → report (RECOMMENDED V10)
  pregame_v7          Per-game: lineups → matchup → V7 → predict → report
  pregame             Per-game: lineups → matchup → predict → report
  predict_today       Per-game prediction only (features must already exist)
  train_weekly        Run V8 (or v7/v6) weekly retraining — Sundays
  backfill_v7         Rebuild V7 features for a historical date range
  backfill_v8         Build V8 features for a historical date range
  schedule_pregame_tasks  Enqueue Cloud Tasks for today's games
  scouting_reports        Build/refresh scouting reports for a date

Environment variables:
  GCP_PROJECT  – defaults to hankstank
"""

import json
import logging
import os
import traceback
from datetime import date, timedelta

import functions_framework

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@functions_framework.http
def daily_pipeline(request):
    """HTTP Cloud Function entry point."""
    try:
        req_json = request.get_json(silent=True) or {}
    except Exception:
        req_json = {}

    # Allow overriding target date and mode via request body
    target_date_str = req_json.get("date")
    mode = req_json.get("mode", "daily")  # daily | backfill | features | predict | validate
    dry_run = req_json.get("dry_run", False)

    yesterday = date.today() - timedelta(days=1)
    target = date.fromisoformat(target_date_str) if target_date_str else yesterday

    results = {"status": "ok", "date": target.isoformat(), "mode": mode, "steps": []}

    # Game PKs for per-game triggered modes (lineups, matchup_features, predict_today)
    game_pks_raw = req_json.get("game_pks", [])
    game_pks = [int(pk) for pk in game_pks_raw] if game_pks_raw else []

    try:
        if mode in ("daily", "backfill"):
            results["steps"].append(_run_collection(target, mode, dry_run, req_json))

        if mode in ("daily", "validate"):
            results["steps"].append(_run_validation())

        if mode in ("daily", "features"):
            results["steps"].append(_run_features(dry_run))

        # V8 Elo update: runs immediately after game collection so Elo is
        # current for today's predictions. Cheap — processes ~15 games max.
        if mode in ("daily", "update_elo"):
            results["steps"].append(_run_v8_elo_update(target, dry_run))

        # V8 features: Pythagorean, run diff, streaks, H2H from BQ game history.
        # Runs daily to ensure game_v8_features table is populated for the
        # pre-game prediction step. Also runs in pregame_v8 mode per-game.
        if mode in ("daily", "v8_features"):
            results["steps"].append(_run_v8_features(target, game_pks, dry_run))

        # V10 features: SP quality, park factors, rest/travel, team quality from
        # MLB API. Runs daily AFTER V8 (reads from game_v8_features). Also runs
        # per-game in pregame_v10 mode.
        if mode in ("daily", "v10_features"):
            results["steps"].append(_run_v10_features(target, game_pks, dry_run))

        # Weekly prediction run (Friday)
        if mode == "predict" or (mode == "daily" and target.weekday() == 4):
            results["steps"].append(_run_weekly_predictions(dry_run))

        # Per-game pre-game modes (triggered by Cloud Tasks ~90 min before first pitch)
        if mode == "lineups":
            results["steps"].append(_run_lineup_fetch(target, game_pks, dry_run))

        if mode == "matchup_features":
            results["steps"].append(_run_matchup_features(target, game_pks, dry_run))

        # V7 matchup features (bullpen health, moon phase, pitcher venue splits)
        if mode in ("matchup_v7_features", "pregame_v7", "daily"):
            results["steps"].append(_run_v7_features(target, game_pks, dry_run))

        if mode == "predict_today":
            results["steps"].append(_run_daily_prediction(target, game_pks, dry_run, req_json))

        # Combined pre-game pipeline:
        #   pregame:    lineups → V5/V6 matchup → V7 features → prediction → scouting report
        #   pregame_v8: lineups → V5/V6 matchup → V7 features → V8 features → prediction → scouting report
        #   pregame_v10:lineups → matchup → V7 → V8 → V10 → prediction → scouting report
        #               V10 is the recommended production mode for best accuracy.
        if mode in ("pregame", "pregame_v7", "pregame_v8", "pregame_v10"):
            results["steps"].append(_run_lineup_fetch(target, game_pks, dry_run))
            results["steps"].append(_run_matchup_features(target, game_pks, dry_run))
            if mode in ("pregame_v7", "pregame_v8", "pregame_v10"):
                results["steps"].append(_run_v7_features(target, game_pks, dry_run))
            if mode in ("pregame_v8", "pregame_v10"):
                results["steps"].append(_run_v8_features(target, game_pks, dry_run))
            if mode == "pregame_v10":
                results["steps"].append(_run_v10_features(target, game_pks, dry_run))
            results["steps"].append(_run_daily_prediction(target, game_pks, dry_run, req_json))
            results["steps"].append(_run_scouting_reports(target, dry_run))

        # Weekly model training (Sundays only) — keeps training cost-efficient
        # model_version options: v10 (recommended), v8, v7, v6 (legacy)
        if mode == "train_weekly" or (mode == "daily" and target.weekday() == 6):
            model_version = req_json.get("model_version", "v10")
            if model_version in ("v10", "v8"):
                results["steps"].append(_run_weekly_training_v8(dry_run))
            elif model_version == "v7":
                results["steps"].append(_run_weekly_training_v7(dry_run))
            else:
                results["steps"].append(_run_weekly_training(dry_run))
            # On Sundays also refresh the in-season SP percentile data in GCS
            # so the next week's predictions have up-to-date Statcast xERA ranks.
            results["steps"].append(_refresh_sp_gcs(target.year, dry_run))

        # V7 backfill: recompute V7 features for a historical date range
        if mode == "backfill_v7":
            results["steps"].append(_run_v7_backfill(
                date.fromisoformat(req_json.get("start", "2026-03-01")),
                date.fromisoformat(req_json.get("end", target.isoformat())),
                dry_run,
            ))

        # V8 backfill: build V8 features for all games since season start
        if mode == "backfill_v8":
            results["steps"].append(_run_v8_backfill(
                date.fromisoformat(req_json.get("start", "2026-03-27")),
                date.fromisoformat(req_json.get("end", target.isoformat())),
                dry_run,
            ))

        # V10 backfill: build V10 features for all games since season start
        if mode == "backfill_v10":
            results["steps"].append(_run_v10_backfill(
                date.fromisoformat(req_json.get("start", "2026-03-27")),
                date.fromisoformat(req_json.get("end", target.isoformat())),
                dry_run,
            ))

        # Roster refresh on Mondays
        if mode == "daily" and target.weekday() == 0:
            results["steps"].append(_run_rosters(target, dry_run))

        # Daily scouting reports: one JSON blob per game written to BQ.
        # Runs in daily mode (after predictions) and on-demand.
        if mode in ("daily", "scouting_reports"):
            report_date = date.fromisoformat(req_json.get("date", target.isoformat()))
            results["steps"].append(_run_scouting_reports(report_date, dry_run))

        # Morning schedule check: enqueue per-game Cloud Tasks for today
        if mode == "schedule_pregame_tasks":
            results["steps"].append(_schedule_pregame_tasks(target, dry_run))

    except Exception as e:
        logger.error("Pipeline error: %s\n%s", e, traceback.format_exc())
        results["status"] = "error"
        results["error"] = str(e)
        return (json.dumps(results), 500, {"Content-Type": "application/json"})

    return (json.dumps(results), 200, {"Content-Type": "application/json"})


def _run_lineup_fetch(target: date, game_pks: list, dry_run: bool) -> dict:
    from fetch_game_lineups import LineupFetcher

    fetcher = LineupFetcher(dry_run=dry_run)
    if game_pks:
        result = fetcher.run_for_game_pks(game_pks, target)
    else:
        result = fetcher.run_for_date(target)
    return {"step": "lineups", **result}


def _run_matchup_features(target: date, game_pks: list, dry_run: bool) -> dict:
    from build_matchup_features import MatchupFeatureBuilder

    builder = MatchupFeatureBuilder(dry_run=dry_run)
    if game_pks:
        result = builder.run_for_game_pks(game_pks, target)
    else:
        result = builder.run_for_date(target)
    return {"step": "matchup_features", **result}


def _run_scouting_reports(target: date, dry_run: bool) -> dict:
    from build_scouting_reports import run as build_reports
    result = build_reports(target, dry_run=dry_run)
    return {"step": "scouting_reports", **result}


def _run_daily_prediction(
    target: date, game_pks: list, dry_run: bool, req_json: dict
) -> dict:
    from predict_today_games import DailyPredictor

    fallback_v4 = req_json.get("fallback_v4", False)
    predictor = DailyPredictor(dry_run=dry_run, fallback_v4=fallback_v4)
    if game_pks:
        result = predictor.run_for_game_pks(game_pks, target)
    else:
        result = predictor.run_for_date(target)
    return {"step": "predict_today", **result}


def _run_weekly_training(dry_run: bool) -> dict:
    """Run V6 model training (Sundays only, weekly cadence — production default)."""
    from train_v6_models import V6Trainer

    trainer = V6Trainer(use_v6_join=True, use_v5_join=True)
    trainer.run(dry_run=dry_run, upload=True)
    return {
        "step": "weekly_training",
        "model": "v6_pitcher_venue_stacked_ensemble",
    }


def _run_weekly_training_v7(dry_run: bool) -> dict:
    """Run V7 model training. Invoked via {mode: train_weekly, model_version: v7}."""
    from train_v7_models import V7Trainer

    trainer = V7Trainer(use_v5_join=True, use_v6_join=True, use_v7_join=True)
    trainer.run(dry_run=dry_run, upload=True)
    return {
        "step": "weekly_training_v7",
        "model": "v7_bullpen_moon_venue_stacked_ensemble",
    }


def _run_v7_features(target: date, game_pks: list, dry_run: bool) -> dict:
    """Build V7 matchup features (bullpen health, moon phase, pitcher venue splits)."""
    from build_v7_features import V7FeatureBuilder

    builder = V7FeatureBuilder(dry_run=dry_run)
    if game_pks:
        result = builder.run_for_game_pks(game_pks, target)
    else:
        result = builder.run_for_date(target)
    return {"step": "v7_features", **result}


def _run_v7_backfill(start: date, end: date, dry_run: bool) -> dict:
    """Backfill V7 features day-by-day for a historical date range."""
    from build_v7_features import V7FeatureBuilder
    import time

    builder = V7FeatureBuilder(dry_run=dry_run)
    total = 0
    errors = []
    current = start
    while current <= end:
        try:
            r = builder.run_for_date(current)
            total += r.get("games_processed", 0)
        except Exception as e:
            errors.append({"date": current.isoformat(), "error": str(e)})
        current += timedelta(days=1)
        time.sleep(0.25)   # gentle rate-limit against BQ
    return {
        "step": "v7_backfill",
        "dates_processed": (end - start).days + 1,
        "games_processed": total,
        "errors": errors,
    }


def _schedule_pregame_tasks(target: date, dry_run: bool) -> dict:
    """
    Morning job: delegate lineup scheduling to the backend so a single
    source of truth controls multi-checkpoint lineup refresh cadence.
    """
    import os
    import requests
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://hankstank.uc.r.appspot.com")
    schedule_url = f"{BACKEND_URL}/api/lineup/schedule-today"

    if dry_run:
        return {
            "step": "schedule_pregame_tasks",
            "date": target.isoformat(),
            "dry_run": True,
            "delegated_to": schedule_url,
        }

    resp = requests.get(
        schedule_url,
        params={"date": target.isoformat()},
        headers={"User-Agent": "HanksTank/2.0"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    logger.info("Delegated lineup scheduling for %s", target.isoformat())
    return {
        "step": "schedule_pregame_tasks",
        "date": target.isoformat(),
        "scheduled": data,
    }


def _run_collection(target: date, mode: str, dry_run: bool, req_json: dict) -> dict:
    from season_2026_pipeline import SeasonPipeline

    pipeline = SeasonPipeline(dry_run=dry_run)
    if mode == "backfill":
        start = date.fromisoformat(req_json.get("start", "2026-02-20"))
        end = date.fromisoformat(req_json.get("end", target.isoformat()))
        pipeline.run_backfill(start, end)
    else:
        pipeline.run_daily(target)

    pipeline.print_summary()
    return {"step": "collection", "stats": {
        k: v for k, v in pipeline.stats.items() if k != "errors"
    }, "errors": pipeline.stats["errors"]}


def _run_validation() -> dict:
    from data_validation import DataValidator

    v = DataValidator(fix_duplicates=True)
    code = v.run()
    return {"step": "validation", "exit_code": code,
            "errors": v.errors, "warnings": v.warnings}


def _run_features(dry_run: bool) -> dict:
    from build_2026_features import FeatureBuilder

    builder = FeatureBuilder(dry_run=dry_run)
    df = builder.build_features()
    builder.save_features(df)
    return {"step": "features", "rows": len(df)}


def _run_weekly_predictions(dry_run: bool) -> dict:
    from predict_2026_weekly import WeeklyPredictor

    predictor = WeeklyPredictor(dry_run=dry_run)
    df = predictor.run()
    return {"step": "predictions", "games": len(df)}


def _run_rosters(target: date, dry_run: bool) -> dict:
    from season_2026_pipeline import SeasonPipeline

    pipeline = SeasonPipeline(dry_run=dry_run)
    pipeline.run_rosters(target)
    return {"step": "rosters", "rows": pipeline.stats["rosters"]}


# ---------------------------------------------------------------------------
# V8 pipeline steps
# ---------------------------------------------------------------------------

def _run_v8_elo_update(target: date, dry_run: bool) -> dict:
    """
    Update team Elo ratings in BQ after yesterday's game outcomes.

    Reads completed games for target date from mlb_2026_season.games,
    applies the Elo K=15 update rule, and writes back to team_elo_ratings.
    This is a 30-row UPSERT — extremely cheap (~$0.00/run).
    """
    from build_v8_features_live import V8LiveFeatureBuilder
    from google.cloud import bigquery

    builder = V8LiveFeatureBuilder(dry_run=dry_run)
    bq = bigquery.Client(project="hankstank")

    try:
        sql = f"""
        SELECT game_pk, game_date, home_team_id, away_team_id,
               CAST(home_score AS INT64) AS home_score,
               CAST(away_score AS INT64) AS away_score
        FROM `hankstank.mlb_2026_season.games`
        WHERE game_date = '{target.isoformat()}'
          AND status IN ('Final', 'Completed Early')
          AND home_score IS NOT NULL AND away_score IS NOT NULL
        """
        completed = bq.query(sql).to_dataframe()
        n = builder.update_elo_after_games(completed)
        return {"step": "v8_elo_update", "games_processed": n, "date": target.isoformat()}
    except Exception as e:
        logger.warning("V8 Elo update failed (non-fatal): %s", e)
        return {"step": "v8_elo_update", "status": "skipped", "reason": str(e)}


def _run_v8_features(target: date, game_pks: list, dry_run: bool) -> dict:
    """
    Build V8 features (Elo, Pythagorean, run differential, streaks, H2H)
    for upcoming games on target date. Writes to game_v8_features table.

    Called once daily (covering all scheduled games) and again per-game in
    pregame_v8 mode to ensure the freshest data before each prediction.
    """
    from build_v8_features_live import V8LiveFeatureBuilder

    builder = V8LiveFeatureBuilder(dry_run=dry_run)
    if game_pks:
        result = builder.run_for_game_pks(game_pks, target)
    else:
        result = builder.run_for_date(target)
    return {"step": "v8_features", **result}


def _run_v8_backfill(start: date, end: date, dry_run: bool) -> dict:
    """Rebuild V8 features for a historical date range (e.g., full 2026 season)."""
    from build_v8_features_live import V8LiveFeatureBuilder

    builder = V8LiveFeatureBuilder(dry_run=dry_run)
    result = builder.run_backfill(start, end)
    return {"step": "v8_backfill", **result}


def _run_v10_features(target: date, game_pks: list, dry_run: bool) -> dict:
    """
    Build V10 features (SP quality, park factors, rest/travel, team quality,
    extra rolling stats) for upcoming games on target date.
    Writes to game_v10_features table. Reads from game_v8_features + MLB API.

    Called once daily (covering all scheduled games) and again per-game in
    pregame_v10 mode to ensure the freshest SP quality data before each prediction.
    """
    from build_v10_features_live import V10LiveFeatureBuilder

    builder = V10LiveFeatureBuilder(dry_run=dry_run)
    if game_pks:
        result = builder.run_for_game_pks(game_pks, target)
    else:
        result = builder.run_for_date(target)
    return {"step": "v10_features", **result}


def _run_v10_backfill(start: date, end: date, dry_run: bool) -> dict:
    """Rebuild V10 features for a historical date range (backfill after deployment)."""
    from build_v10_features_live import V10LiveFeatureBuilder

    builder = V10LiveFeatureBuilder(dry_run=dry_run)
    result = builder.run_backfill(start, end)
    return {"step": "v10_backfill", **result}


def _refresh_sp_gcs(year: int, dry_run: bool) -> dict:
    """
    Refresh the current-season Statcast SP percentile parquet in GCS.

    Called every Sunday so xERA/K%/BB%/whiff/FBV percentile ranks stay current
    as pitchers accumulate plate appearances mid-season. Historical years (< current
    season) are stable and skipped.

    Downloads fresh data directly from Baseball Savant's public CSV endpoint,
    computes percentile ranks, and writes the parquet to GCS at:
        gs://hanks_tank_data/sp_quality/statcast_sp_{year}.parquet
    """
    import io
    import urllib.request
    from datetime import date as _date

    import pandas as pd
    from google.cloud import storage

    current_year = _date.today().year
    if year < current_year:
        return {"step": "refresh_sp_gcs", "status": "skipped (historical year)", "year": year}
    if dry_run:
        return {"step": "refresh_sp_gcs", "status": "skipped (dry_run)", "year": year}

    try:
        # Baseball Savant pitcher leaderboard CSV (current season, min 10 PA)
        url = (
            f"https://baseballsavant.mlb.com/leaderboard/expected_statistics"
            f"?type=pitcher&year={year}&position=&team=&min=10&csv=true"
        )
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read()

        df = pd.read_csv(io.BytesIO(raw))

        # Compute percentile ranks for key SP quality columns
        pct_cols = ["xera", "k_percent", "bb_percent", "whiff_percent", "fastball_avg_speed"]
        existing = [c for c in pct_cols if c in df.columns]
        for col in existing:
            df[f"{col}_pct"] = df[col].rank(pct=True, ascending=(col == "bb_percent")) * 100

        # Upload to GCS
        bucket_name = "hanks_tank_data"
        blob_path = f"sp_quality/statcast_sp_{year}.parquet"
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_file(buf, content_type="application/octet-stream")

        return {
            "step": "refresh_sp_gcs",
            "status": "ok",
            "year": year,
            "rows": len(df),
            "gcs_path": f"gs://{bucket_name}/{blob_path}",
        }
    except Exception as e:
        return {"step": "refresh_sp_gcs", "status": "error", "error": str(e)}


def _run_weekly_training_v8(dry_run: bool) -> dict:
    """
    Re-train V8 model weekly (Sundays) with updated 2026 season data.

    V8 uses CatBoost with team ID embeddings and the full 85-feature set.
    Weekly retraining ensures the model stays calibrated as the season progresses
    and team quality distributions shift (injuries, trades, callups).

    Training is done in the Cloud Function (memory: 2GB, timeout: 540s).
    The updated model is uploaded to GCS at models/vertex/game_outcome_2026_v8/model.pkl
    """
    import subprocess
    import sys

    if dry_run:
        logger.info("[DRY RUN] Would run V8 weekly training")
        return {"step": "weekly_training_v8", "status": "dry_run"}

    try:
        result = subprocess.run(
            [sys.executable, "train_v8_models.py", "--weekly-update"],
            capture_output=True, text=True, timeout=480,
        )
        if result.returncode != 0:
            logger.error("V8 training error: %s", result.stderr[-2000:])
            return {
                "step": "weekly_training_v8",
                "status": "error",
                "stderr": result.stderr[-500:],
            }
        return {
            "step": "weekly_training_v8",
            "model": "v8_catboost_team_embeddings",
            "status": "ok",
        }
    except Exception as e:
        logger.error("V8 training exception: %s", e)
        return {"step": "weekly_training_v8", "status": "error", "error": str(e)}
