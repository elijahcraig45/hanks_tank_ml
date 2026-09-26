"""Production runner for the PA simulator.

Writes to `mlb_2026_season.game_predictions_sim` -- a SHADOW table with the same
contract as game_predictions -- so the simulator can run alongside V10 in production
without touching what the live site serves. Promoting it is then a one-line change to
the writer target, made deliberately rather than as a side effect of deploying.

Two repo lessons are applied here on purpose:
  * MODEL_LESSONS_LEARNED #4 -- Cloud Run's filesystem is read-only, so every cache
    write sits in its own try/except and is treated as optional; the fetch that must
    succeed is in a separate block.
  * the CF flattens `src/`, so nothing below resolves a repo-root-relative path.
"""
from __future__ import annotations
import os, json, logging
from datetime import date, datetime, timedelta, timezone
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

PROJECT = os.environ.get("GCP_PROJECT", "hankstank")
DATASET = os.environ.get("MLB_2026_DATASET", "mlb_2026_season")
SHADOW_TABLE = os.environ.get("PA_SIM_TABLE", "game_predictions_sim")
MODEL_VERSION = "pa_sim_v1"
N_EPISODES = int(os.environ.get("PA_SIM_EPISODES", "1000"))
CACHE_DIR = os.environ.get("PA_SIM_CACHE", "/tmp/pa_sim")

PA_SQL = """
WITH ev AS (
  SELECT game_year, game_date, game_pk, batter, pitcher, stand, p_throws, inning,
         inning_topbot, home_team, away_team,
         CASE
           WHEN events IN ('strikeout','strikeout_double_play') THEN 'K'
           WHEN events IN ('walk','intent_walk','hit_by_pitch','catcher_interf') THEN 'BB'
           WHEN events = 'single' THEN '1B'
           WHEN events = 'double' THEN '2B'
           WHEN events = 'triple' THEN '3B'
           WHEN events = 'home_run' THEN 'HR'
           WHEN events = 'field_error' THEN '1B'
           WHEN events IN ('grounded_into_double_play','double_play',
                           'sac_fly_double_play','triple_play') THEN 'DP'
           WHEN events IN ('field_out','force_out','sac_fly','sac_bunt',
                           'fielders_choice','fielders_choice_out') THEN 'OUT'
           ELSE NULL END AS ev
  FROM `{proj}.{ds}.statcast_pitches`
  WHERE game_type='R' AND events IS NOT NULL AND events != '' AND game_date < @cutoff
),
fi AS (SELECT game_pk, pitcher, MIN(inning) fi FROM ev GROUP BY 1,2)
SELECT e.* EXCEPT(ev), e.ev, (f.fi = 1) AS pitcher_is_starter
FROM ev e JOIN fi f ON f.game_pk=e.game_pk AND f.pitcher=e.pitcher
WHERE e.ev IS NOT NULL
"""

SLATE_SQL = """
WITH snap AS (
  SELECT game_pk, game_date, team_type, batting_order, player_id, player_name, fetched_at, game_time_utc,
         ROW_NUMBER() OVER (PARTITION BY game_pk, team_type, batting_order
                            ORDER BY fetched_at DESC) rn
  FROM `{proj}.{ds}.lineups`
  WHERE game_date = @d AND batting_order BETWEEN 1 AND 9
    AND fetched_at < game_time_utc
),
lu AS (SELECT * FROM snap WHERE rn=1),
sp AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT game_pk, home_starter_id, away_starter_id, home_starter_name, away_starter_name,
           game_time_utc, home_team_id, away_team_id, home_team_name, away_team_name,
           ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY predicted_at DESC) rn
    FROM `{proj}.{ds}.game_predictions` WHERE game_date = @d) WHERE rn=1
),
pf AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT game_pk, home_park_factor,
           ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) rn
    FROM `{proj}.{ds}.game_v10_features` WHERE game_date = @d) WHERE rn=1
)
SELECT lu.game_pk, lu.game_date, lu.team_type, lu.batting_order, lu.player_id, lu.player_name,
       sp.home_starter_id, sp.away_starter_id, sp.home_starter_name, sp.away_starter_name,
       sp.game_time_utc, sp.home_team_id, sp.away_team_id,
       sp.home_team_name, sp.away_team_name,
       IFNULL(pf.home_park_factor, 1.0) AS home_park_factor
FROM lu JOIN sp USING (game_pk) LEFT JOIN pf USING (game_pk)
ORDER BY lu.game_pk, lu.team_type, lu.batting_order
"""


def _load_pa(bq, cutoff: date) -> pd.DataFrame:
    """Fetch PA history. The fetch must succeed; caching it is best-effort."""
    cache = os.path.join(CACHE_DIR, f"pa_{cutoff.isoformat()}.parquet")
    try:
        if os.path.exists(cache):
            df = pd.read_parquet(cache)
            logger.info("pa_sim: loaded %d PAs from cache", len(df))
            return df
    except Exception as e:                      # a corrupt cache must never be fatal
        logger.warning("pa_sim: cache read failed (%s); refetching", e)

    from google.cloud import bigquery
    hist = PA_SQL.format(proj=PROJECT, ds="mlb_historical_data")
    cur = PA_SQL.format(proj=PROJECT, ds=DATASET)
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("cutoff", "DATE", cutoff)])
    df = pd.concat([bq.query(hist, job_config=cfg).to_dataframe(),
                    bq.query(cur, job_config=cfg).to_dataframe()], ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["ev"] = df["ev"].astype(str)

    try:                                        # separate block: read-only FS is fine
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(cache, index=False)
    except Exception as e:
        logger.debug("pa_sim: could not cache locally (non-fatal): %s", e)
    return df


def run_slate(target: date, dry_run: bool = False, n_episodes: int = N_EPISODES,
              alpha: float | None = None) -> dict:
    """Simulate every game on `target`'s slate and write to the shadow table."""
    from google.cloud import bigquery
    from pa_sim.predict import SimEngine

    bq = bigquery.Client(project=PROJECT)
    cutoff = target                              # strictly-before: no same-day leakage
    pa = _load_pa(bq, cutoff)
    if len(pa) < 50_000:
        return {"step": "pa_sim", "status": "skipped",
                "reason": f"only {len(pa)} PAs before {cutoff}"}

    a = alpha if alpha is not None else float(os.environ.get("PA_SIM_ALPHA", "1.1551"))
    eng = SimEngine(n_episodes=n_episodes, alpha=a).fit(pa, pd.Timestamp(target))
    hand = pa.groupby("pitcher", observed=True)["p_throws"].agg(
        lambda s: s.mode().iloc[0]).to_dict()
    name2ab = _team_abbr_map(bq)

    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("d", "DATE", target)])
    slate = bq.query(SLATE_SQL.format(proj=PROJECT, ds=DATASET), job_config=cfg).to_dataframe()
    if slate.empty:
        return {"step": "pa_sim", "status": "no_games", "date": str(target)}

    rows = []
    for gid, g in slate.groupby("game_pk", sort=False):
        h = g[g.team_type == "home"].sort_values("batting_order")
        aw = g[g.team_type == "away"].sort_values("batting_order")
        r0 = g.iloc[0]
        if len(h) != 9 or len(aw) != 9 or pd.isna(r0.home_starter_id) or pd.isna(r0.away_starter_id):
            logger.warning("pa_sim: skipping %s (incomplete lineup/starter)", gid)
            continue
        res = eng.predict_game(
            [int(x) for x in h.player_id], [int(x) for x in aw.player_id],
            int(r0.home_starter_id), int(r0.away_starter_id),
            name2ab.get(r0.home_team_name, ""), name2ab.get(r0.away_team_name, ""),
            park_factor=float(r0.home_park_factor or 1.0),
            home_starter_hand=hand.get(int(r0.home_starter_id)),
            away_starter_hand=hand.get(int(r0.away_starter_id)),
            n_episodes=n_episodes, seed=int(gid) % 10_000)
        p = res["home_win_prob"]
        rows.append(dict(
            game_pk=int(gid), game_date=target,
            home_team_id=int(r0.home_team_id), away_team_id=int(r0.away_team_id),
            home_team_name=r0.home_team_name, away_team_name=r0.away_team_name,
            home_starter_id=int(r0.home_starter_id), away_starter_id=int(r0.away_starter_id),
            home_win_probability=p, away_win_probability=1 - p,
            predicted_winner=r0.home_team_name if p >= .5 else r0.away_team_name,
            confidence_tier=_tier(p), model_version=MODEL_VERSION,
            n_episodes=int(res["n_episodes"]),
            mean_home_runs=res["mean_home"], mean_away_runs=res["mean_away"],
            p_extra_innings=res["p_extra"], park_alpha=res["park_alpha"],
            game_time_utc=r0.game_time_utc,
            predicted_at=datetime.now(timezone.utc)))

    out = {"step": "pa_sim", "date": str(target), "games": len(rows),
           "n_episodes": n_episodes, "alpha": a, "table": SHADOW_TABLE}
    if dry_run:
        out["status"] = "dry_run"
        out["sample"] = rows[:3]
        return out
    if rows:
        df = pd.DataFrame(rows)
        tbl = f"{PROJECT}.{DATASET}.{SHADOW_TABLE}"
        bq.query(f"DELETE FROM `{tbl}` WHERE game_date = @d AND model_version = @m",
                 job_config=bigquery.QueryJobConfig(query_parameters=[
                     bigquery.ScalarQueryParameter("d", "DATE", target),
                     bigquery.ScalarQueryParameter("m", "STRING", MODEL_VERSION)])).result()
        bq.load_table_from_dataframe(
            df, tbl, job_config=bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",
                schema_update_options=["ALLOW_FIELD_ADDITION"])).result()
    out["status"] = "ok"
    return out


def _team_abbr_map(bq) -> dict:
    """Full team name -> statcast abbreviation, learned by joining on game_pk.

    Statcast carries abbreviations (ATH, NYM); the prediction/games tables carry full
    names. Deriving the map from a join is reliable; guessing from the name is not.
    An unmatched name yields "", which makes the bullpen lookup fall back to the
    league reliever baseline rather than silently using the wrong team.
    """
    sql = f"""
    SELECT g.home_team_name AS name, ANY_VALUE(s.home_team) AS ab
    FROM `{PROJECT}.{DATASET}.games` g
    JOIN (SELECT DISTINCT game_pk, home_team FROM `{PROJECT}.{DATASET}.statcast_pitches`) s
      USING (game_pk)
    GROUP BY name
    """
    try:
        df = bq.query(sql).to_dataframe()
        return dict(zip(df.name, df.ab))
    except Exception as e:
        logger.warning("pa_sim: team abbreviation map unavailable (%s); "
                       "bullpens fall back to league baseline", e)
        return {}


def _tier(p: float) -> str:
    c = max(p, 1 - p)
    return "high" if c >= 0.64 else ("medium" if c >= 0.57 else "low")
