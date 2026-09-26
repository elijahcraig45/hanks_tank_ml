"""Cloud Function entry point for the college football weekly pipeline.

Separate function from both the MLB daily pipeline and the NFL weekly one. Deployed by
scripts/gcp/cfb/deploy_cfb.sh, which stages src/cfb plus the two shared modules it
imports from src/nfl (features.py, train_nfl_models.py) into a temp source dir — so the
image carries no MLB code and no nflverse dependency.

Modes (POST body {"mode": ...}):
  ingest        refresh games from ESPN, score last week's picks, then rankings + stats
  score         fill in results for predictions whose games have now been played
  rankings      rebuild the Bradley-Terry board only
  stats         rebuild ESPN team season stats and league leaders only
  cfbd          rebuild the CollegeFootballData tables: advanced stats, players, lines
  predict_week  predict a scheduled week for both divisions
  backfill      re-run a completed season week by week, per division
                ({"model": "ridge"} writes the shadow ridge to its own table)
  fpi_snapshot  record ESPN FPI's pregame win probability for the next unplayed week
  drives        fetch CFBD /drives for completed weeks not yet stored, into
                cfb_historical.drives (game_id-scoped; honours dry_run: fetches, writes nothing)

Shadow model: {"shadow_ridge": true} on predict_week/predict_next (or CFB_RIDGE_SHADOW=1)
also writes the margin ridge's predictions to cfb_season.game_predictions_ridge_shadow.
Nothing reads that table; it exists so the ridge can be scored live before adoption.

Drive-sim shadow: {"shadow_drive_sim": true} on predict_week/predict_next (or
CFB_DRIVE_SIM_SHADOW=1, which deploy_cfb.sh --shadow sets) also runs the college drive
simulator (cfb_drive_sim.py) and writes cfb_season.game_sim_distributions and
cfb_season.game_predictions_drive_sim: pregame games only, game_id-scoped replaces into
tables that must already exist (CREATE_NEVER; DDL in
scripts/gcp/football/create_cfb_drive_sim_tables.sql), nothing under dry_run. It trains
on cfb_historical.drives, which `drives` mode keeps current. Experimental: it loses to
the margin ridge on winners in the backtest; its validated use is the margin shape.

FPI snapshot: {"fpi_snapshot": true} on ingest/predict_week (or FPI_SNAPSHOT=1) also
appends ESPN FPI's pregame predictions to cfb_season.fpi_game_predictions, for the model
comparison page. Off by default; the table must be created first (it is loaded with
CREATE_NEVER) — see scripts/gcp/football/create_fpi_game_predictions.sql.

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




def _shadow_enabled(req: dict) -> bool:
    """The margin ridge is an experiment: it runs only when asked for, per request
    ({"shadow_ridge": true}) or per deployment (CFB_RIDGE_SHADOW=1), and writes only
    to its own table, which nothing reads. Off by default."""
    import os

    return bool(req.get("shadow_ridge")) or os.environ.get("CFB_RIDGE_SHADOW") == "1"


def _shadow_ridge_week(season: int, week: int, steps: dict) -> None:
    """Predict the week with the margin ridge into the shadow table. Never fatal:
    an experiment must not cost the production predictions anything."""
    try:
        import cfb_config
        from backfill_cfb import ensure_datasets, replace_game_ids
        from pipeline import SHADOW_TABLE, predict_week

        rows = predict_week(season, week, model="ridge")
        if not rows.empty:
            ensure_datasets()
            replace_game_ids(rows, cfb_config.CTX.season_dataset, SHADOW_TABLE,
                             partition_field="game_date",
                             cluster_fields=["season", "division"])
        steps["shadow_ridge"] = len(rows)
    except Exception as exc:
        logger.error("shadow ridge failed: %s", exc)
        steps["shadow_ridge"] = {"error": str(exc)[:200]}


def _drive_sim_enabled(req: dict) -> bool:
    """The drive simulator is a shadow: per request ({"shadow_drive_sim": true}) or per
    deployment (CFB_DRIVE_SIM_SHADOW=1). Off by default."""
    import os

    return bool(req.get("shadow_drive_sim")) or os.environ.get("CFB_DRIVE_SIM_SHADOW") == "1"


def _drive_sim_week(season: int, week: int, steps: dict, dry_run: bool = False) -> None:
    """Drive-sim shadow for one week. Never fatal; under dry_run computes, writes nothing."""
    try:
        import cfb_drive_sim

        dist, pred, info = cfb_drive_sim.predict_week_drive_sim(season, week)
        games = dist["game_id"].astype(str).tolist() if len(dist) else []
        if dry_run or not len(dist):
            steps["shadow_drive_sim"] = {**info, "written": 0, "games": games}
            return
        from backfill_cfb import ensure_datasets

        ensure_datasets()
        steps["shadow_drive_sim"] = {**info, "written": cfb_drive_sim.write(dist, pred)}
    except (Exception, SystemExit) as exc:
        logger.error("shadow drive_sim failed: %s", exc)
        steps["shadow_drive_sim"] = {"error": str(exc)[:200]}


def _fpi_snapshot(season: int, week: int | None, steps: dict) -> None:
    """Append FPI's pregame numbers for one week's games that have not kicked off. Never
    fatal: FPI is shown for comparison and must not cost the pipeline anything."""
    try:
        from espn_data import fetch_scheduled
        from pipeline import next_unplayed_week
        from rankings import fpi_games

        week = week if week is not None else next_unplayed_week(season)
        if week is None:
            steps["fpi_snapshot"] = 0
            return
        slate = fpi_games.cfb_slate(fetch_scheduled(season, int(week)))
        fpi_games.run_snapshot("cfb", slate, steps)
    except Exception as exc:
        logger.error("FPI snapshot failed: %s", exc)
        steps["fpi_snapshot"] = {"error": str(exc)[:200]}


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


def _refresh_stats(season: int, steps: dict,
                   providers: tuple[str, ...] = ("espn",),
                   label: str = "stats") -> None:
    """Rebuild stat tables for the given providers. Never fatal.

    Defaults to ESPN only, so the weekly ingest keeps the cost it always had. The
    CollegeFootballData feeds are a dozen HTTP calls, an 85-column flatten and a
    14,000-row pivot, and chaining that onto an invocation already spending its 540
    seconds on games, scoring and a bootstrap fit is how a timeout starts costing the
    games ingest everything else depends on. They get their own mode instead.
    """
    try:
        from stats.build import build as build_stats, write_bq as write_stats

        tables = build_stats("cfb", season, providers=providers)
        steps[label] = write_stats("cfb", season, tables)
    except Exception as exc:
        logger.error("%s refresh failed: %s", label, exc)
        steps[label] = {"error": str(exc)[:200]}


@functions_framework.http
def cfb_pipeline(request):
    try:
        req = request.get_json(silent=True) or {}
    except Exception:
        req = {}

    mode = req.get("mode", "ingest")
    result: dict = {"mode": mode, "steps": {}}

    # dry_run used to be ignored here, so {"dry_run": true} ran for real and replaced
    # a week of production predictions. Now only the predict modes support it (they
    # compute and return, writing nothing); any other mode refuses rather than run.
    # `drives` also honours it: it fetches and transforms, and skips every BigQuery write.
    dry_run = bool(req.get("dry_run"))
    if dry_run and mode not in ("predict_week", "predict_next", "drives"):
        return ({"mode": mode, "error": "dry_run is only supported for predict_week, "
                 "predict_next and drives"}, 400)

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

            from rankings import fpi_games

            if fpi_games.enabled(req):
                _fpi_snapshot(season, None, result["steps"])

        elif mode == "fpi_snapshot":
            import cfb_config

            season = int(req.get("season", cfb_config.CTX.season))
            week = req.get("week")
            _fpi_snapshot(season, int(week) if week is not None else None,
                          result["steps"])

        elif mode == "score":
            _score_predictions(result["steps"])

        elif mode == "drives":
            # CollegeFootballData /drives for the drive-sim shadow: one call per completed
            # week that has no drives stored yet, capped per run.
            import cfb_config
            from stats import cfbd

            season = int(req.get("season", cfb_config.CTX.season))
            cfbd.reset_call_counter()
            if not cfbd.has_api_key():
                result["steps"]["drives"] = {"skipped": "no CFBD_API_KEY configured"}
            else:
                from cfb_drives import ingest as ingest_drives

                weeks = req.get("weeks")
                result["steps"]["drives"] = ingest_drives(
                    season, dry_run=dry_run,
                    weeks=[int(w) for w in weeks] if weeks else None)
                result["steps"]["cfbd_calls"] = cfbd.calls_used()
            if dry_run:
                result["dry_run"] = True

        elif mode == "rankings":
            import cfb_config

            season = int(req.get("season", cfb_config.CTX.season))
            _refresh_rankings(season, result["steps"])

        elif mode == "stats":
            import cfb_config

            season = int(req.get("season", cfb_config.CTX.season))
            _refresh_stats(season, result["steps"])

        elif mode == "cfbd":
            # CollegeFootballData feeds: per-game and per-season advanced stats, the
            # per-player table, and betting lines. Separate from `stats` because these
            # need an API key and a paid tier, so they must be able to fail — or be
            # skipped entirely on an unkeyed deployment — without touching anything
            # ESPN supplies.
            import cfb_config
            from stats import cfbd

            season = int(req.get("season", cfb_config.CTX.season))
            cfbd.reset_call_counter()

            if not cfbd.has_api_key():
                result["steps"]["cfbd"] = {"skipped": "no CFBD_API_KEY configured"}
            else:
                _refresh_stats(season, result["steps"],
                               providers=("cfbd",), label="cfbd")

                # The pick'em sheet. Refreshed here rather than in `ingest` because it
                # needs the CFBD schedule AND the rankings the ingest produces, so it
                # has to run after both — and because a failure must cost the sheet
                # only, never the stats.
                try:
                    from stats import pickem as pickem_games

                    result["steps"]["pickem"] = pickem_games.refresh("cfb", season)
                except Exception as exc:
                    logger.error("pickem refresh failed: %s", exc)
                    result["steps"]["pickem"] = {"error": str(exc)[:200]}

                # Surfaced so monthly spend is visible in the logs rather than
                # discovered when the allowance runs out.
                result["steps"]["cfbd_calls"] = cfbd.calls_used()

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
            if dry_run:
                result["dry_run"] = True
                result["week"] = int(week)
                result["steps"]["predicted"] = len(rows)
                result["steps"]["games"] = (rows["game_id"].astype(str).tolist()
                                            if not rows.empty else [])
                if _shadow_enabled(req):
                    try:
                        result["steps"]["shadow_ridge"] = len(
                            predict_week(season, int(week), model="ridge"))
                    except Exception as exc:
                        result["steps"]["shadow_ridge"] = {"error": str(exc)[:200]}
                if _drive_sim_enabled(req):
                    _drive_sim_week(season, int(week), result["steps"], dry_run=True)
                result["status"] = "ok"
                return (result, 200)
            if not rows.empty:
                from backfill_cfb import replace_game_ids

                ensure_datasets()
                # Replace exactly the games predicted, so re-runs are idempotent. This
                # used to clear every unscored row of the week, which deleted the
                # pregame rows of games already under way or final-but-unscored,
                # and predict_week never rewrites those.
                replace_game_ids(rows, cfb_config.CTX.season_dataset, "game_predictions",
                                 partition_field="game_date",
                                 cluster_fields=["season", "division"])
                result["steps"]["predicted"] = len(rows)

            if _shadow_enabled(req):
                _shadow_ridge_week(season, int(week), result["steps"])
            if _drive_sim_enabled(req):
                _drive_sim_week(season, int(week), result["steps"])

            from rankings import fpi_games

            if fpi_games.enabled(req):
                _fpi_snapshot(season, int(week), result["steps"])

        elif mode == "backfill":
            import cfb_config
            import pandas as pd
            from backfill_cfb import ensure_datasets, load
            from pipeline import SHADOW_TABLE, backfill_division, build, load_played_games

            # Accepts a list so several seasons can be rebuilt with one methodology.
            requested = req.get("seasons") or [req.get("season", cfb_config.CTX.season)]
            seasons = [int(s) for s in requested]
            # {"model": "ridge"} backfills the shadow model into its own table only.
            model = "ridge" if req.get("model") == "ridge" else "xgb"
            games = load_played_games()
            feats = build(games)

            frames = []
            for season in seasons:
                for division in DIVISIONS:
                    rows = backfill_division(feats, division, season, model=model,
                                             games=games)
                    if not rows.empty:
                        frames.append(rows)
                        result["steps"][f"{season}_{division}"] = len(rows)

            if frames and model == "ridge":
                from backfill_cfb import replace_game_ids

                ensure_datasets()
                result["steps"]["written_shadow"] = replace_game_ids(
                    pd.concat(frames, ignore_index=True),
                    cfb_config.CTX.season_dataset, SHADOW_TABLE,
                    partition_field="game_date", cluster_fields=["season", "division"])
                frames = []

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
