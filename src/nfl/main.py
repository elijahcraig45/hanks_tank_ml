"""Cloud Function entry point for the NFL weekly pipeline.

Deployed as its own function (`nfl-weekly-pipeline`) with --source=src/nfl, NOT bundled
with the MLB pipeline. That isolation is deliberate: an NFL deploy must never redeploy
the code that predicts MLB games daily, and the NFL image must not carry pybaseball,
MLB-StatsAPI, or catboost.

Modes (POST body {"mode": ...}):
  ingest        refresh schedules + EPA, then refresh rankings and player stats
  rankings      rebuild the Bradley-Terry board only
  stats         rebuild player season stats and league leaders only
  predict_week  predict a scheduled week (defaults to the next unplayed week)
  score         recompute results for completed games and update predictions
  backfill      re-run a whole season week by week

The derived steps hang off `ingest` rather than taking Scheduler jobs of their own:
they only make sense after the week's games land, so chaining them in one invocation
makes that ordering structural rather than a race between cron entries. Both are
non-fatal — a failed leaderboard must not cost us the ingest everything else needs.

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



def _refresh_rankings(season: int, steps: dict) -> None:
    """Rebuild and publish the power-ranking board. Never fatal."""
    try:
        from rankings.build import build_board, write_bq

        table, meta = build_board("nfl", season, n_boot=200)
        steps["rankings"] = write_bq(table, meta)
    except Exception as exc:
        logger.error("rankings refresh failed: %s", exc)
        steps["rankings"] = {"error": str(exc)[:200]}


def _refresh_stats(season: int, steps: dict) -> None:
    """Rebuild player season stats and league leaders. Never fatal."""
    try:
        from stats.build import build as build_stats, write_bq as write_stats

        tables = build_stats("nfl", season)
        steps["stats"] = write_stats("nfl", season, tables)
    except Exception as exc:
        logger.error("stats refresh failed: %s", exc)
        steps["stats"] = {"error": str(exc)[:200]}


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
            season = int(req.get("season", CTX.season))

            # Games and teams come from a single upstream file that always carries the
            # whole history, so rebuilding them in full is correct and cheap.
            result["steps"]["games"] = backfill_games()
            result["steps"]["teams"] = backfill_teams()

            # EPA is different: it is derived from play-by-play, the heaviest load in
            # this repo. Scoped to the current season so the weekly run refreshes a
            # slice instead of rebuilding twenty seasons — which is what it was doing,
            # and why it died at the 2GB limit on every cold container without ever
            # writing the current season.
            result["steps"]["epa"] = backfill_epa(seasons=[season])

            # Derived from the games that just landed, so they belong in this call.
            _refresh_rankings(season, result["steps"])
            _refresh_stats(season, result["steps"])

            # The pick'em sheet, after the rankings it enriches each side with.
            # Non-fatal: a sheet without context is still a usable sheet, and losing
            # the ingest because a board was missing would be the wrong trade.
            try:
                from stats import pickem as pickem_games

                result["steps"]["pickem"] = pickem_games.refresh("nfl", season)
            except Exception as exc:
                logger.error("pickem refresh failed: %s", exc)
                result["steps"]["pickem"] = {"error": str(exc)[:200]}

        elif mode == "rankings":
            from config import CTX

            _refresh_rankings(int(req.get("season", CTX.season)), result["steps"])

        elif mode == "stats":
            from config import CTX

            _refresh_stats(int(req.get("season", CTX.season)), result["steps"])

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
            # home_won is set alongside the two result columns, not just implied by
            # them. The diagnostics endpoint derives Brier and log loss from it, so a
            # row scored without it reads as an away win and reports the wrong error
            # for every game — while still looking correctly scored.
            sql = f"""
              UPDATE `{CTX.project}.{CTX.season_dataset}.game_predictions` p
              SET p.home_won = g.home_won,
                  p.prediction_correct =
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

            import pandas as pd

            requested = req.get("seasons") or [req.get("season", 2025)]
            seasons = [int(s) for s in requested]

            frames = []
            for season in seasons:
                rows = backfill_season(season)
                if not rows.empty:
                    frames.append(rows)
                    result["steps"][str(season)] = len(rows)

            if frames:
                allrows = pd.concat(frames, ignore_index=True)
                ensure_dataset(CTX.season_dataset)

                # Replace exactly the games being rewritten so a re-run is idempotent
                # rather than duplicating every prediction it already made.
                from google.cloud import bigquery

                client = bigquery.Client(project=CTX.project)
                table_id = f"{CTX.project}.{CTX.season_dataset}.game_predictions"
                try:
                    client.query(
                        f"DELETE FROM `{table_id}` WHERE game_id IN UNNEST(@ids)",
                        job_config=bigquery.QueryJobConfig(
                            query_parameters=[
                                bigquery.ArrayQueryParameter(
                                    "ids", "STRING",
                                    allrows["game_id"].astype(str).tolist()
                                )
                            ]
                        ),
                    ).result()
                except Exception as exc:
                    logger.warning("pre-delete skipped (%s)", exc)

                load_table(allrows, CTX.season_dataset, "game_predictions",
                           write_disposition="WRITE_APPEND")
                result["steps"]["written"] = len(allrows)

        else:
            return ({"error": f"unknown mode: {mode}"}, 400)

        result["status"] = "ok"
        return (result, 200)

    except Exception as exc:
        logger.error("pipeline failed: %s", exc)
        logger.error(traceback.format_exc())
        return ({"status": "error", "mode": mode, "error": str(exc)}, 500)
