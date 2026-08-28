"""Cloud Function entry point for the college football weekly pipeline.

Separate function from both the MLB daily pipeline and the NFL weekly one. Deployed by
scripts/gcp/cfb/deploy_cfb.sh, which stages src/cfb plus the two shared modules it
imports from src/nfl (features.py, train_nfl_models.py) into a temp source dir — so the
image carries no MLB code and no nflverse dependency.

Modes (POST body {"mode": ...}):
  ingest        refresh FBS + FCS games from ESPN into cfb_historical
  predict_week  predict a scheduled week for both divisions
  backfill      re-run a completed season week by week, per division
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
            from backfill_cfb import ensure_datasets, load
            from espn_data import fetch_history
            import pandas as pd

            season = int(req.get("season", cfb_config.CTX.season))
            games = fetch_history(first_season=season, last_season=season)
            ensure_datasets()
            g = games.copy()
            g["game_date"] = pd.to_datetime(g["game_date"])
            result["steps"]["games"] = load(
                g, cfb_config.CTX.hist_dataset, "games",
                partition_field="game_date", cluster_fields=["season", "division"],
            )

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

            season = int(req.get("season", cfb_config.CTX.season))
            feats = build(load_games())

            frames = []
            for division in DIVISIONS:
                rows = backfill_division(feats, division, season)
                if not rows.empty:
                    frames.append(rows)
                    result["steps"][division] = len(rows)

            if frames:
                ensure_datasets()
                allrows = pd.concat(frames, ignore_index=True)
                load(allrows, cfb_config.CTX.season_dataset, "game_predictions",
                     partition_field="game_date", cluster_fields=["season", "division"])

        else:
            return ({"error": f"unknown mode: {mode}"}, 400)

        result["status"] = "ok"
        return (result, 200)

    except Exception as exc:
        logger.error("cfb pipeline failed: %s", exc)
        logger.error(traceback.format_exc())
        return ({"status": "error", "mode": mode, "error": str(exc)}, 500)
