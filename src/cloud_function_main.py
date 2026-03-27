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

        # Roster refresh on Mondays
        if mode == "daily" and target.weekday() == 0:
            results["steps"].append(_run_rosters(target, dry_run))

    except Exception as e:
        logger.error("Pipeline error: %s\n%s", e, traceback.format_exc())
        results["status"] = "error"
        results["error"] = str(e)
        return (json.dumps(results), 500, {"Content-Type": "application/json"})

    return (json.dumps(results), 200, {"Content-Type": "application/json"})


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
