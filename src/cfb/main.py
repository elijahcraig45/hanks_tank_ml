"""Cloud Function entry point for the college football weekly pipeline.

Separate function from both the MLB daily pipeline and the NFL weekly one. Deployed by
scripts/gcp/cfb/deploy_cfb.sh, which stages src/cfb plus the two shared modules it
imports from src/nfl (features.py, train_nfl_models.py) into a temp source dir — so the
image carries no MLB code and no nflverse dependency.

Modes (POST body {"mode": ...}):
  ingest        refresh games from ESPN, score last week's picks, then rankings + stats
  score         fill in results for predictions whose games have now been played
  rankings      rebuild the Bradley-Terry board only
  stats         rebuild team season stats and league leaders only
  predict_week  predict a scheduled week for both divisions
  backfill      re-run a completed season week by week, per division

`ingest` runs the derived steps itself rather than each getting its own Scheduler job:
they must run after the games land, and chaining them in one invocation makes that
ordering structural instead of a race between two cron entries. They are also
non-fatal — a failed leaderboard must not cost us the game ingest that everything else
depends on — so each records its own status in `steps`.
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

DIVISIONS = ("fbs", "fcs")




def _score_predictions(steps: dict) -> None:
    """Fill in the outcome columns for predictions whose games have since finished.

    Without this the loop never closes: predict_week writes a row per upcoming game
    with prediction_correct NULL, the games get played, and nothing ever goes back to
    record whether the pick was right. The diagnostics page reads exactly those columns,
    so an unscored week is an invisible week.

    Written as an UPDATE-in-place rather than a rewrite so the original prediction — and
    the timestamp proving it was made before kickoff — survives untouched.
    """
    try:
        import cfb_config
        from google.cloud import bigquery

        client = bigquery.Client(project=cfb_config.CTX.project)
        preds = (f"{cfb_config.CTX.project}."
                 f"{cfb_config.CTX.season_dataset}.game_predictions")
        games = f"{cfb_config.CTX.project}.{cfb_config.CTX.hist_dataset}.games"

        job = client.query(f"""
            UPDATE `{preds}` p
            SET p.home_won = g.home_won,
                p.actual_winner = IF(g.home_won = 1,
                                     p.home_team_name, p.away_team_name),
                p.prediction_correct =
                    IF((p.home_win_probability > 0.5) = (g.home_won = 1), 1, 0)
            FROM `{games}` g
            WHERE p.game_id = g.game_id
              AND g.home_won IS NOT NULL
              AND p.prediction_correct IS NULL
        """)
        job.result()
        steps["scored"] = job.num_dml_affected_rows
    except Exception as exc:
        logger.error("scoring failed: %s", exc)
        steps["scored"] = {"error": str(exc)[:200]}


def _refresh_rankings(season: int, steps: dict) -> None:
    """Rebuild and publish the power-ranking board. Never fatal.

    The board needs BOTH this season and last: last season is the decaying prior that
    makes an early-season rating meaningful rather than noise. A fresh Cloud Function
    container starts with an empty games cache, and `ingest` only fetches the current
    season, so ask for the prior explicitly — fetch_history skips season/division pairs
    it already has, making the overlap cheap.
    """
    try:
        from espn_data import fetch_history
        from rankings.build import build_board, write_bq

        fetch_history(first_season=season - 1, last_season=season)
        table, meta = build_board("cfb", season, n_boot=200)
        steps["rankings"] = write_bq(table, meta)
    except Exception as exc:
        logger.error("rankings refresh failed: %s", exc)
        steps["rankings"] = {"error": str(exc)[:200]}


def _refresh_stats(season: int, steps: dict) -> None:
    """Rebuild team season stats and league leaders. Never fatal."""
    try:
        from stats.build import build as build_stats, write_bq as write_stats

        tables = build_stats("cfb", season)
        steps["stats"] = write_stats("cfb", season, tables)
    except Exception as exc:
        logger.error("stats refresh failed: %s", exc)
        steps["stats"] = {"error": str(exc)[:200]}


@functions_framework.http
def cfb_pipeline(request):
    try:
        req = request.get_json(silent=True) or {}
    except Exception:
        req = {}

    mode = req.get("mode", "ingest")
    result: dict = {"mode": mode, "steps": {}}

    try:
        if mode == "ingest":
            import cfb_config
            from backfill_cfb import ensure_datasets, replace_seasons
            from espn_data import fetch_history
            import pandas as pd

            season = int(req.get("season", cfb_config.CTX.season))
            games = fetch_history(first_season=season, last_season=season)
            ensure_datasets()
            g = games.copy()
            g["game_date"] = pd.to_datetime(g["game_date"])

            # Replace only this season, never the table. The previous WRITE_TRUNCATE
            # wiped 2021-2024: fetch_history is scoped to one season, and on a cold
            # container the /tmp parquet cache is empty, so the frame that overwrote
            # the whole table held nothing but the current season.
            result["steps"]["games"] = replace_seasons(
                g, cfb_config.CTX.hist_dataset, "games",
                partition_field="game_date", cluster_fields=["season", "division"],
            )

            # All derived from the games that just landed, so they belong in this
            # call. Scoring runs before the rankings so a week's results are on the
            # record before anything is rated on them.
            _score_predictions(result["steps"])
            _refresh_rankings(season, result["steps"])
            _refresh_stats(season, result["steps"])

        elif mode == "score":
            _score_predictions(result["steps"])

        elif mode == "rankings":
            import cfb_config

            season = int(req.get("season", cfb_config.CTX.season))
            _refresh_rankings(season, result["steps"])

        elif mode == "stats":
            import cfb_config

            season = int(req.get("season", cfb_config.CTX.season))
            _refresh_stats(season, result["steps"])

        elif mode in ("predict_week", "predict_next"):
            import cfb_config
            import pandas as pd
            from backfill_cfb import ensure_datasets, load
            from pipeline import next_unplayed_week, predict_week

            season = int(req.get("season", cfb_config.CTX.season))
            week = req.get("week")
            if week is None:
                week = next_unplayed_week(season)
                if week is None:
                    result["steps"]["predicted"] = 0
                    result["status"] = "ok"
                    result["note"] = "no unplayed weeks remaining"
                    return (result, 200)

            rows = predict_week(season, int(week))
            if not rows.empty:
                ensure_datasets()
                # Replace this week's slice so re-runs are idempotent.
                from google.cloud import bigquery
                c = bigquery.Client(project=cfb_config.CTX.project)
                try:
                    c.query(
                        f"DELETE FROM `{cfb_config.CTX.project}."
                        f"{cfb_config.CTX.season_dataset}.game_predictions` "
                        f"WHERE season={season} AND week={int(week)} "
                        f"AND prediction_correct IS NULL"
                    ).result()
                except Exception:
                    pass
                load(rows, cfb_config.CTX.season_dataset, "game_predictions",
                     write_disposition="WRITE_APPEND")
                result["steps"]["predicted"] = len(rows)

        elif mode == "backfill":
            import cfb_config
            import pandas as pd
            from backfill_cfb import ensure_datasets, load
            from espn_data import load_games
            from pipeline import backfill_division, build

            # Accepts a list so several seasons can be rebuilt with one methodology.
            requested = req.get("seasons") or [req.get("season", cfb_config.CTX.season)]
            seasons = [int(s) for s in requested]
            feats = build(load_games())

            frames = []
            for season in seasons:
                for division in DIVISIONS:
                    rows = backfill_division(feats, division, season)
                    if not rows.empty:
                        frames.append(rows)
                        result["steps"][f"{season}_{division}"] = len(rows)

            if frames:
                ensure_datasets()
                allrows = pd.concat(frames, ignore_index=True)

                # Replace exactly the games being rewritten, never the table. The
                # previous WRITE_TRUNCATE wiped every row including upcoming weeks —
                # predictions for unplayed games that a backfill cannot regenerate,
                # because it only ever produces rows for completed games.
                from google.cloud import bigquery

                client = bigquery.Client(project=cfb_config.CTX.project)
                table_id = (f"{cfb_config.CTX.project}."
                            f"{cfb_config.CTX.season_dataset}.game_predictions")
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

                load(allrows, cfb_config.CTX.season_dataset, "game_predictions",
                     partition_field="game_date", cluster_fields=["season", "division"],
                     write_disposition="WRITE_APPEND")
                result["steps"]["written"] = len(allrows)

        else:
            return ({"error": f"unknown mode: {mode}"}, 400)

        result["status"] = "ok"
        return (result, 200)

    except Exception as exc:
        logger.error("cfb pipeline failed: %s", exc)
        logger.error(traceback.format_exc())
        return ({"status": "error", "mode": mode, "error": str(exc)}, 500)
