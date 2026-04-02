#!/usr/bin/env python3
"""
Cloud Function entry point for the 2026 daily pipeline.

Triggered by Cloud Scheduler via HTTP. Runs:
  1. Data collection (yesterday's games, stats, standings, statcast)
  2. Validation
  3. Feature rebuild
  4. Weekly: rosters refresh, batch predictions

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

        # Weekly prediction run (Friday)
        if mode == "predict" or (mode == "daily" and target.weekday() == 4):
            results["steps"].append(_run_weekly_predictions(dry_run))

        # Per-game pre-game modes (triggered by Cloud Tasks ~90 min before first pitch)
        if mode == "lineups":
            results["steps"].append(_run_lineup_fetch(target, game_pks, dry_run))

        if mode == "matchup_features":
            results["steps"].append(_run_matchup_features(target, game_pks, dry_run))

        if mode == "predict_today":
            results["steps"].append(_run_daily_prediction(target, game_pks, dry_run, req_json))

        # Combined pre-game pipeline: lineups → matchup features → today prediction
        if mode == "pregame":
            results["steps"].append(_run_lineup_fetch(target, game_pks, dry_run))
            results["steps"].append(_run_matchup_features(target, game_pks, dry_run))
            results["steps"].append(_run_daily_prediction(target, game_pks, dry_run, req_json))

        # Weekly model training (Sundays only) — keeps training cost-efficient
        if mode == "train_weekly" or (mode == "daily" and target.weekday() == 6):
            results["steps"].append(_run_weekly_training(dry_run))

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
    """Run V5 model training (Sundays only, weekly cadence)."""
    from train_v5_models import V5Trainer

    trainer = V5Trainer(use_matchup_join=True)
    trainer.run(dry_run=dry_run)
    return {
        "step": "weekly_training",
        "model": "v5_matchup_stacked_ensemble",
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
