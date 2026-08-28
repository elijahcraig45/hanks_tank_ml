"""Cloud Function entry point for the NFL weekly pipeline.

Deployed as its own function (`nfl-weekly-pipeline`) with --source=src/nfl, NOT bundled
with the MLB pipeline. That isolation is deliberate: an NFL deploy must never redeploy
the code that predicts MLB games daily, and the NFL image must not carry pybaseball,
MLB-StatsAPI, or catboost.

Modes (POST body {"mode": ...}):
  ingest        refresh schedules + EPA into nfl_historical
  predict_week  predict a scheduled week (defaults to the next unplayed week)
  score         recompute results for completed games and update predictions
  backfill      re-run a whole season week by week

Cadence note: NFL is week-shaped, not date-shaped. Results settle Thursday->Monday, so
ingest runs Tuesday and predictions run Wednesday.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import functions_framework

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _next_unplayed_week() -> tuple[int, int]:
    """The earliest scheduled week with no result yet."""
    import pandas as pd
    from data import load_schedules

    sched = load_schedules(refresh=True)
    pending = sched[sched["result"].isna()].copy()
    if pending.empty:
        raise RuntimeError("no unplayed games on the schedule")
    pending["gameday"] = pd.to_datetime(pending["gameday"])
    row = pending.sort_values("gameday").iloc[0]
    return int(row["season"]), int(row["week"])


@functions_framework.http
def nfl_pipeline(request):
    try:
        req = request.get_json(silent=True) or {}
    except Exception:
        req = {}

    mode = req.get("mode", "predict_week")
    result: dict = {"mode": mode, "steps": {}}

    try:
        if mode == "ingest":
            from backfill_nfl_history import backfill_epa, backfill_games, backfill_teams
            from bq_io import ensure_dataset
            from config import CTX

            ensure_dataset(CTX.hist_dataset)
            ensure_dataset(CTX.season_dataset)
            result["steps"]["games"] = backfill_games()
            result["steps"]["teams"] = backfill_teams()
            result["steps"]["epa"] = backfill_epa()

        elif mode == "predict_week":
            from bq_io import ensure_dataset, upsert_week
            from config import CTX
            from predict_nfl import predict_week

            season = req.get("season")
            week = req.get("week")
            if not (season and week):
                season, week = _next_unplayed_week()

            rows = predict_week(int(season), int(week))
            ensure_dataset(CTX.season_dataset)
            upsert_week(rows, CTX.season_dataset, "game_predictions",
                        int(season), int(week))
            result["steps"]["predicted"] = len(rows)
            result["season"] = season
            result["week"] = week

        elif mode == "score":
            from bq_io import query
            from config import CTX

            # Join stored predictions to final scores and refresh the result columns.
            sql = f"""
              UPDATE `{CTX.project}.{CTX.season_dataset}.game_predictions` p
              SET p.prediction_correct =
                    IF((p.home_win_probability > 0.5) = (g.home_won = 1), 1, 0),
                  p.actual_winner =
                    IF(g.home_won = 1, p.home_team_name, p.away_team_name)
              FROM `{CTX.project}.{CTX.hist_dataset}.games` g
              WHERE p.game_id = g.game_id AND g.home_won IS NOT NULL
            """
            query(sql)
            result["steps"]["scored"] = "ok"

        elif mode == "backfill":
            from bq_io import ensure_dataset, load_table
            from config import CTX
            from predict_nfl import backfill_season

            season = int(req.get("season", 2025))
            rows = backfill_season(season)
            ensure_dataset(CTX.season_dataset)
            load_table(rows, CTX.season_dataset, "game_predictions",
                       write_disposition="WRITE_APPEND")
            result["steps"]["backfilled"] = len(rows)

        else:
            return ({"error": f"unknown mode: {mode}"}, 400)

        result["status"] = "ok"
        return (result, 200)

    except Exception as exc:
        logger.error("pipeline failed: %s", exc)
        logger.error(traceback.format_exc())
        return ({"status": "error", "mode": mode, "error": str(exc)}, 500)
