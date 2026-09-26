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
  daily               Full daily pipeline (steps 1-5, power rankings, Monday rosters).
                      V7 features and scouting reports are NOT in it: for target =
                      yesterday they only rebuilt already-final games (~300s).
  rosters             Roster snapshot only (for mlb-2026-roster-refresh)
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
  power_rankings          Rebuild the Bradley-Terry MLB power-ranking board

Environment variables:
  GCP_PROJECT  – defaults to hankstank
"""

import json
import logging
import os
import sys
import time
import traceback
from datetime import date, timedelta

import functions_framework

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class _TimedSteps(list):
    """The results["steps"] list, stamping each step with how long it took.

    Every step is appended right after it returns, so the time since the previous
    append is that step's duration. Each stamp is also printed as a structured log
    line, because INFO from `logger` does not reach Cloud Logging in the deployed
    function -- only WARNING and above show up, without our basicConfig format,
    so the root logger is evidently configured before that call and it is a
    no-op. That is why a 504 used to leave no trace of which step was still
    running when the 540s limit hit.
    """

    def __init__(self, mode: str):
        super().__init__()
        self._mode = mode
        self._t0 = self._last = time.monotonic()

    def append(self, step):
        now = time.monotonic()
        if isinstance(step, dict):
            step.setdefault("seconds", round(now - self._last, 1))
        self._last = now
        super().append(step)
        name = step.get("step", "?") if isinstance(step, dict) else "?"
        secs = step.get("seconds") if isinstance(step, dict) else None
        elapsed = round(now - self._t0, 1)
        print(json.dumps({
            "severity": "NOTICE",
            "message": f"[{self._mode}] step {name} took {secs}s ({elapsed}s elapsed)",
            "mode": self._mode,
            "step": name,
            "step_seconds": secs,
            "elapsed_seconds": elapsed,
        }), file=sys.stdout, flush=True)


def _run_power_rankings(dry_run: bool) -> dict:
    """Rebuild the MLB power-ranking board.

    Same Bradley-Terry engine the football boards use. Runs daily off the back of the
    collection step, and is deliberately non-fatal: this board is a reading surface, and
    losing it must never take down the predictions the site is actually built on.

    Note the ratings barely separate baseball teams (measured walk-forward, team
    strength moves log loss from 0.6931 to only 0.6808), which is why the published
    board leads with bootstrap rank ranges rather than a confident 1-30 order.
    """
    step = {"step": "power_rankings"}
    if dry_run:
        step["status"] = "skipped (dry run)"
        return step

    try:
        from rankings.build import build_board, write_bq

        season = int(os.environ.get("MLB_SEASON", date.today().year))
        table, meta = build_board("mlb", season, n_boot=200)
        step.update(write_bq(table, meta))
        step["status"] = "ok"
        step["teams"] = len(table)
    except Exception as exc:
        logger.error("power rankings refresh failed: %s", exc)
        step["status"] = "error"
        step["error"] = str(exc)[:200]
    return step


@functions_framework.http
def daily_pipeline(request):
    """HTTP Cloud Function entry point."""
    try:
        req_json = request.get_json(silent=True) or {}
    except Exception:
        req_json = {}

    # Allow overriding target date and mode via request body
    target_date_str = req_json.get("date")
    mode = req_json.get("mode", "daily")  # daily | backfill | features | predict | validate | pa_sim
    dry_run = req_json.get("dry_run", False)

    yesterday = date.today() - timedelta(days=1)
    target = date.fromisoformat(target_date_str) if target_date_str else yesterday

    results = {"status": "ok", "date": target.isoformat(), "mode": mode,
               "steps": _TimedSteps(mode)}

    # Game PKs for per-game triggered modes (lineups, matchup_features, predict_today)
    game_pks_raw = req_json.get("game_pks", [])
    game_pks = [int(pk) for pk in game_pks_raw] if game_pks_raw else []

    try:
        if mode in ("daily", "backfill"):
            results["steps"].append(_run_collection(target, mode, dry_run, req_json))

        if mode in ("daily", "validate"):
            results["steps"].append(_run_validation(dry_run))

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

        # Power rankings: refit from the games just collected. Cheap (30 teams) and
        # non-fatal, so it rides along with the daily run instead of taking its own
        # Scheduler job.
        if mode in ("daily", "power_rankings"):
            results["steps"].append(_run_power_rankings(dry_run))

        # Weekly prediction run — mlb-2026-weekly-predict, Friday 5 AM ET.
        #
        # This used to also fire from `daily` when target.weekday() == 4. Since
        # `target` is yesterday that actually ran on Saturdays, and it was the
        # only path that ran at all while mlb-2026-weekly-predict was POSTing a
        # malformed body and falling through to `daily`. With that job fixed the
        # piggyback is pure duplication — _run_weekly_predictions takes no date
        # and always predicts the upcoming week from now, so both invocations
        # write the same slate a day apart, and a repeat write inside the
        # streaming-buffer window degrades to INSERT-only and duplicates rows.
        if mode == "predict":
            results["steps"].append(_run_weekly_predictions(dry_run))

        # Per-game pre-game modes (triggered by Cloud Tasks ~90 min before first pitch)
        if mode == "lineups":
            results["steps"].append(_run_lineup_fetch(target, game_pks, dry_run))

        if mode == "matchup_features":
            results["steps"].append(_run_matchup_features(target, game_pks, dry_run))

        # V7 matchup features (bullpen health, moon phase, pitcher venue splits).
        #
        # Not part of `daily` any more. There `target` is yesterday, so this rebuilt
        # V7 rows for games that were already final -- ~150s of BigQuery round
        # trips (measured 2026-09-25, 12 games) producing post-hoc rows nothing
        # forward-looking reads. The pregame_v7/v8/v10 tasks already build V7 per
        # game before first pitch, which is the only copy predictions use.
        if mode in ("matchup_v7_features", "pregame_v7"):
            results["steps"].append(_run_v7_features(target, game_pks, dry_run))

        if mode == "predict_today":
            results["steps"].append(_run_daily_prediction(target, game_pks, dry_run, req_json))

        # PA simulator, shadow-only (writes game_predictions_sim, never game_predictions)
        if mode in ("pa_sim", "pregame_sim"):
            results["steps"].append(_run_pa_sim(target, dry_run, req_json))

        # More shadows, same rule: they write their own tables, never game_predictions.
        # Target defaults to yesterday, so a forward-looking run must pass "date".
        if mode == "logit3":
            results["steps"].append(_run_logit3(target, game_pks, dry_run))
        if mode == "sim_blend":
            results["steps"].append(_run_sim_blend(target, game_pks, dry_run, req_json))

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
            # shadow run so both models are scored on the same games; enable with
            # {"run_pa_sim": true} in the task body once the shadow table exists
            if mode == "pregame_v10" and req_json.get("run_pa_sim"):
                results["steps"].append(_run_pa_sim(target, dry_run, req_json))
            # opt-in shadows, off by default: {"run_logit3": true} / {"run_sim_blend": true}
            if mode == "pregame_v10" and req_json.get("run_logit3"):
                results["steps"].append(_run_logit3(target, game_pks, dry_run))
            if mode == "pregame_v10" and req_json.get("run_sim_blend"):
                results["steps"].append(_run_sim_blend(target, game_pks, dry_run, req_json))
            results["steps"].append(_run_scouting_reports(target, dry_run))

        # Weekly model training — mlb-2026-weekly-train-v10, Sunday 2 AM ET.
        # model_version options: v10 (recommended), v8, v7, v6 (legacy)
        #
        # This used to also fire from `daily` whenever target (yesterday) was a
        # Sunday, i.e. on Monday's 4 AM run and on the Monday 3 AM roster-refresh
        # job (whose body is also {"mode":"daily"}). It launches a subprocess with
        # a 480s timeout inside a 540s request, so both Monday runs 504'd, and the
        # dedicated Sunday job already does this work.
        if mode == "train_weekly":
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

        # Roster refresh on Mondays, or on its own via {"mode":"rosters"} -- which is
        # what mlb-2026-roster-refresh should send. Its body is {"mode":"daily"}, so
        # at Monday 3 AM it re-runs the whole daily chain with target = Sunday and
        # never reaches this branch: the job named roster-refresh refreshes no rosters.
        if mode == "rosters":
            roster_date = date.fromisoformat(target_date_str) if target_date_str else date.today()
            results["steps"].append(_run_rosters(roster_date, dry_run))
        elif mode == "daily" and target.weekday() == 0:
            results["steps"].append(_run_rosters(target, dry_run))

        # Scouting reports: one JSON blob per game written to BQ.
        #
        # On demand only. `daily` used to run this for `target` = yesterday, i.e.
        # it rebuilt reports for games already played (~150s, 5 BigQuery queries a
        # game run serially). It was the last step of the chain and the one still
        # running when mlb-2026-daily hit the 540s limit. Pregame tasks write each
        # report before first pitch.
        if mode == "scouting_reports":
            report_date = date.fromisoformat(req_json.get("date", target.isoformat()))
            results["steps"].append(_run_scouting_reports(report_date, dry_run))

        # Morning schedule check: enqueue per-game Cloud Tasks for today.
        # Must target today, not the pipeline-wide default of yesterday — the
        # backend skips any game whose first pitch has already passed, so
        # yesterday's slate enqueues nothing at all. An explicit "date" in the
        # request body still wins, for manual re-runs.
        if mode == "schedule_pregame_tasks":
            pregame_target = (
                date.fromisoformat(target_date_str) if target_date_str else date.today()
            )
            results["steps"].append(_schedule_pregame_tasks(pregame_target, dry_run))

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


def _run_validation(dry_run: bool = False) -> dict:
    from data_validation import DataValidator

    # The duplicate fix DELETEs rows from games, so a dry run only reports duplicates.
    v = DataValidator(fix_duplicates=not dry_run)
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


def _run_pa_sim(target: date, dry_run: bool, req_json: dict) -> dict:
    """Plate-appearance Monte Carlo simulation for the target date's slate.

    Plays each game `n_episodes` times PA by PA off ~1.9M historical plate appearances
    and writes the resulting win probabilities to game_predictions_sim -- a SHADOW
    table. It deliberately does NOT write game_predictions, so running this in
    production cannot change what the live site serves; promoting it is a separate,
    explicit change to PA_SIM_TABLE.
    """
    from pa_sim.pipeline import run_slate

    return run_slate(
        target,
        dry_run=dry_run,
        n_episodes=int(req_json.get("n_episodes", 1000)),
        alpha=req_json.get("alpha"),
    )


def _shadow(step: str, fn) -> dict:
    """A shadow model must never fail the production run it rides along with."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - logged and reported, deliberately non-fatal
        logger.exception("%s shadow failed (non-fatal)", step)
        return {"step": step, "status": "error", "error": str(e)[:300]}


def _run_logit3(target: date, game_pks: list, dry_run: bool) -> dict:
    """3-feature L1 logistic, refit in-season; writes game_predictions_logit3 only."""
    def go():
        from logit3_shadow import run_slate
        return run_slate(target, dry_run=dry_run, game_pks=game_pks or None)
    return _shadow("logit3", go)


def _run_sim_blend(target: date, game_pks: list, dry_run: bool, req_json: dict) -> dict:
    """v2 PA sim + team-strength blend; writes game_predictions_sim_blend, game_props_sim,
    game_sim_distributions and player_sim_projections only (append, this run's pregame
    games only; nothing under dry_run). Refuses (insufficient_memory) below
    SIM_BLEND_MIN_MEMORY_MB, which the current 1 GB function is."""
    def go():
        from pa_sim.blend import memory_ok, run_slate
        ok, have, need = memory_ok()
        if not ok:
            return {"step": "sim_blend", "status": "insufficient_memory",
                    "memory_mb": have, "required_mb": need}
        return run_slate(target, dry_run=dry_run, game_pks=game_pks or None,
                         experimental=bool(req_json.get("experimental_props")),
                         n_episodes=req_json.get("n_episodes"))
    return _shadow("sim_blend", go)


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
