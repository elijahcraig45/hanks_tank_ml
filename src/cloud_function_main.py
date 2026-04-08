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
  pregame_v8          Per-game: lineups → matchup → V7 → V8 → predict (RECOMMENDED)
  pregame_v7          Per-game: lineups → matchup → V7 → predict
  pregame             Per-game: lineups → matchup → predict
  predict_today       Per-game prediction only (features must already exist)
  train_weekly        Run V8 (or v7/v6) weekly retraining — Sundays
  backfill_v7         Rebuild V7 features for a historical date range
  backfill_v8         Build V8 features for a historical date range
  schedule_pregame_tasks  Enqueue Cloud Tasks for today's games

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
        #   pregame:    lineups → V5/V6 matchup → V7 features → prediction
        #   pregame_v8: lineups → V5/V6 matchup → V7 features → V8 features → prediction
        #                                                         ^^^^^^^^^^^^
        #               V8 features build from the game_pks in this window
        #               ensuring Elo + rolling stats are current for this game
        if mode in ("pregame", "pregame_v7", "pregame_v8"):
            results["steps"].append(_run_lineup_fetch(target, game_pks, dry_run))
            results["steps"].append(_run_matchup_features(target, game_pks, dry_run))
            if mode in ("pregame_v7", "pregame_v8"):
                results["steps"].append(_run_v7_features(target, game_pks, dry_run))
            if mode == "pregame_v8":
                results["steps"].append(_run_v8_features(target, game_pks, dry_run))
            results["steps"].append(_run_daily_prediction(target, game_pks, dry_run, req_json))

        # Weekly model training (Sundays only) — keeps training cost-efficient
        # model_version options: v8 (recommended), v7, v6 (legacy)
        if mode == "train_weekly" or (mode == "daily" and target.weekday() == 6):
            model_version = req_json.get("model_version", "v8")
            if model_version == "v8":
                results["steps"].append(_run_weekly_training_v8(dry_run))
            elif model_version == "v7":
                results["steps"].append(_run_weekly_training_v7(dry_run))
            else:
                results["steps"].append(_run_weekly_training(dry_run))

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

        # Roster refresh on Mondays
        if mode == "daily" and target.weekday() == 0:
            results["steps"].append(_run_rosters(target, dry_run))

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
    Morning job: inspect today's game schedule and enqueue Cloud Tasks
    for each game's pre-game pipeline (~90 minutes before first pitch).
    Called by cron at ~10 AM ET / 14:00 UTC each day.
    """
    import os
    import requests
    import urllib3
    from datetime import datetime, timezone, timedelta

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    MLB_API = "https://statsapi.mlb.com/api/v1"
    BACKEND_URL = os.environ.get(
        "BACKEND_URL",
        "https://hankstank.uc.r.appspot.com"
    )

    # Fetch today's schedule
    resp = requests.get(
        f"{MLB_API}/schedule",
        params={"date": target.isoformat(), "sportId": 1,
                "hydrate": "team", "gameType": "R,F,D,L,W"},
        headers={"User-Agent": "HanksTank/2.0"},
        timeout=30,
        verify=False,
    )
    resp.raise_for_status()
    data = resp.json()

    tasks_scheduled = []
    now_utc = datetime.now(tz=timezone.utc)

    for day in data.get("dates", []):
        for g in day.get("games", []):
            game_pk = g["gamePk"]
            game_datetime_str = g.get("gameDate", "")
            try:
                game_time = datetime.fromisoformat(game_datetime_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            # Schedule pre-game task for 90 minutes before first pitch
            trigger_time = game_time - timedelta(minutes=90)

            if trigger_time <= now_utc:
                # Game is too soon or already started — run immediately if < 30 min past
                if now_utc - trigger_time < timedelta(minutes=30):
                    trigger_time = now_utc + timedelta(seconds=30)
                else:
                    logger.info("Skipping task for game %s — trigger time already passed", game_pk)
                    continue

            delay_seconds = max(0, int((trigger_time - now_utc).total_seconds()))

            if not dry_run:
                # POST to backend scheduler endpoint which creates the Cloud Task
                try:
                    task_resp = requests.post(
                        f"{BACKEND_URL}/api/lineup/schedule-task",
                        json={
                            "game_pks": [game_pk],
                            "game_date": target.isoformat(),
                            "delay_seconds": delay_seconds,
                        },
                        headers={"Content-Type": "application/json"},
                        timeout=15,
                    )
                    if task_resp.ok:
                        tasks_scheduled.append({
                            "game_pk": game_pk,
                            "trigger_time": trigger_time.isoformat(),
                            "delay_seconds": delay_seconds,
                        })
                    else:
                        logger.warning(
                            "Failed to schedule task for game %s: %s",
                            game_pk, task_resp.text
                        )
                except Exception as task_err:
                    logger.error("Error scheduling task for game %s: %s", game_pk, task_err)
            else:
                tasks_scheduled.append({
                    "game_pk": game_pk,
                    "trigger_time": trigger_time.isoformat(),
                    "delay_seconds": delay_seconds,
                    "dry_run": True,
                })

    logger.info("Scheduled %d pre-game tasks for %s", len(tasks_scheduled), target)
    return {
        "step": "schedule_pregame_tasks",
        "tasks_scheduled": len(tasks_scheduled),
        "tasks": tasks_scheduled,
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
