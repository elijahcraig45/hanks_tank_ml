"""College drive-simulator shadow (`cfb_drive_sim_v1`): one week's pregame slate ->
cfb_season.game_sim_distributions + cfb_season.game_predictions_drive_sim rows.

The simulator itself is the shared src/nfl/drive_sim.py run with its CFB config
(division covariates, college overtime, wider score states; frozen on 2022 only).
Backtest, stated plainly (research/football_2026_09/cfb_drive_sim/, 2025 holdout):
winners LOSE to the margin ridge (log loss +0.034 [+0.025, +0.044]), margins and totals
lose to the ridge and the market, and the one validated gain is the exact-margin SHAPE
centred on the spread (log score -0.087 vs a normal at the spread). So the served
predictions never read these tables; they exist to score the sim live.

Pregame only: the slate is filtered exactly as pipeline.predict_week filters it (no
result, kickoff strictly after now), so a rerun after kickoff can never replace a
pregame row. Drives come from cfb_historical.drives, never from CFBD at predict time.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DIST_TABLE = "game_sim_distributions"
PRED_TABLE = "game_predictions_drive_sim"
CLUSTER = ["season", "week"]
_FLOAT_COLS = ("p_home_cover", "spread_line", "total_line", "vegas_implied_home_prob",
               "model_vs_vegas_edge")


def pregame_slate(upcoming: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Unplayed games whose kickoff is strictly after `now` (UTC)."""
    if upcoming is None or upcoming.empty:
        return pd.DataFrame()
    up = upcoming[upcoming["home_won"].isna()].copy()
    up["game_date"] = pd.to_datetime(up["game_date"])
    ko = up["game_date"]
    ko = ko.dt.tz_localize("UTC") if ko.dt.tz is None else ko.dt.tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    up = up[ko > now].copy()
    up["kickoff"] = ko[ko > now]
    return up


def load_lines(season: int) -> pd.DataFrame:
    """cfb_season.betting_lines (spread positive = home favoured). Missing is fine."""
    try:
        import cfb_config
        from google.cloud import bigquery

        table = f"{cfb_config.CTX.project}.{cfb_config.CTX.season_dataset}.betting_lines"
        cfg = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("s", "INT64", int(season))])
        return bigquery.Client(project=cfb_config.CTX.project).query(
            f"SELECT game_id, ANY_VALUE(spread_line) AS spread_line, "
            f"ANY_VALUE(total_line) AS total_line FROM `{table}` WHERE season = @s "
            f"GROUP BY game_id", job_config=cfg).to_dataframe()
    except Exception as exc:
        logger.warning("betting lines unavailable (%s); margin_exact will be raw", exc)
        return pd.DataFrame(columns=["game_id", "spread_line", "total_line"])


def predict_week_drive_sim(season: int, week: int,
                           played: pd.DataFrame | None = None,
                           upcoming: pd.DataFrame | None = None,
                           drives: pd.DataFrame | None = None,
                           lines: pd.DataFrame | None = None,
                           now: pd.Timestamp | None = None,
                           n_sims: int | None = None):
    """(distribution rows, prediction rows, info) for the pregame games of one week."""
    import drive_sim as ds
    from cfb_drives import load_drives
    from espn_data import resolve_team_divisions
    from pipeline import confidence_tier, load_played_games

    cfg = ds.CFB
    if upcoming is None:
        from espn_data import fetch_scheduled

        upcoming = fetch_scheduled(season, week)
    slate = pregame_slate(upcoming, now)
    if slate.empty:
        return pd.DataFrame(), pd.DataFrame(), {"games": 0, "note": "no pregame games"}
    played = load_played_games() if played is None else played
    if drives is None:
        drives = load_drives(season - cfg.fit_seasons)
    if drives is None or drives.empty or (drives["season"] == season - 1).sum() == 0:
        raise RuntimeError(f"cfb drive_sim: no {season - 1} drives in cfb_historical.drives "
                           "— run the drives backfill before the shadow")
    played_ids = set(played.loc[(played["season"] == season) & (played["week"] < week),
                                "game_id"].astype(str))
    have_ids = set(drives.loc[drives["season"] == season, "game_id"].astype(str))
    lag_games = len(played_ids - have_ids)
    if lag_games:
        logger.warning("cfb drives missing for %d played %d games", lag_games, season)

    games = pd.concat([played, slate], ignore_index=True)
    fbs = {k: int(v == "fbs") for k, v in resolve_team_divisions(played).items()}
    teams = pd.Index(sorted(set(games["home_team"]) | set(games["away_team"])))
    lines = load_lines(season) if lines is None else lines
    s = slate.merge(lines[["game_id", "spread_line", "total_line"]], on="game_id", how="left")
    s["gameday"] = s["kickoff"]
    s["game_type"] = "REG"
    s["season"] = int(season)
    s["week"] = int(week)
    dist, pred, timing = ds.simulate_slate(
        s, ds.prep_drives(drives, games, cfg), games, season, week, n=n_sims or cfg.n_sims,
        now=(now.to_pydatetime() if now is not None else None), cfg=cfg, fbs=fbs, teams=teams)
    if len(pred):
        names = s.set_index("game_id")[["home_team_name", "away_team_name", "kickoff"]]
        names = names[~names.index.duplicated()]
        gid = pred["game_id"]
        pred["home_team_name"] = gid.map(names.home_team_name).values
        pred["away_team_name"] = gid.map(names.away_team_name).values
        # game_predictions keeps the kickoff instant in game_date; so does this table.
        pred["game_date"] = pd.to_datetime(gid.map(names.kickoff).values, utc=True)
        p = pred["home_win_probability"].to_numpy(float)
        pred["predicted_winner"] = np.where(p > 0.5, pred.home_team_name, pred.away_team_name)
        pred["confidence_tier"] = [confidence_tier(x) for x in p]
    for df in (dist, pred):
        for c in _FLOAT_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return dist, pred, {**timing, "drives_missing_games": lag_games, "games": len(dist),
                        "skipped_started_or_final": int(len(upcoming) - len(slate))}


def write(dist: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """game_id-scoped replace into tables that must already exist (CREATE_NEVER)."""
    import cfb_config
    from backfill_cfb import replace_game_ids

    ds_ = cfb_config.CTX.season_dataset
    n1 = replace_game_ids(dist, ds_, DIST_TABLE, partition_field="game_date",
                          cluster_fields=CLUSTER, create_disposition="CREATE_NEVER")
    n2 = replace_game_ids(pred, ds_, PRED_TABLE, partition_field="game_date",
                          cluster_fields=CLUSTER, create_disposition="CREATE_NEVER")
    return {DIST_TABLE: n1, PRED_TABLE: n2}
