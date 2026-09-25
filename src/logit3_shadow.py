"""3-feature L1 logistic, the "simple baseline", as a SHADOW writer.

What it is: logistic regression on three pregame differentials,
    elo_differential, pythag_differential, sp_quality_composite_diff,
median-imputed, standardised, L1 penalty with C=0.557. That configuration is what
two independent autosearch runs (450 trials, 7 model families, 161 features;
research/backtest_2026/15_autosearch.py, data/backtest_2026/autosearch_best.json)
converged on. It is refit every run on THIS season's completed games only, because
training on 2015-2025 measured worse than the current season alone (FINDINGS.md).

Leakage rules, both enforced in SQL:
  * a feature row counts only if it was computed strictly before first pitch
    (computed_at < game_time_utc), latest such row per game — game_v10_features is
    rewritten by backfills, and the latest row overall is the contaminated one;
  * training games are Final regular-season games dated strictly before the target.

Table: mlb_2026_season.game_predictions_logit3. The load uses CREATE_NEVER, so until the
DDL in scripts/gcp/2026_season/create_game_predictions_logit3.sql is approved and run, a
real run returns status "table_missing" (non-fatal) instead of creating anything.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT = os.environ.get("GCP_PROJECT", "hankstank")
DATASET = os.environ.get("MLB_2026_DATASET", "mlb_2026_season")
TABLE = os.environ.get("LOGIT3_TABLE", "game_predictions_logit3")
MODEL_VERSION = "logit3_l1_v1"
FEATURES = ["elo_differential", "pythag_differential", "sp_quality_composite_diff"]
C = 0.5567860808442991          # autosearch_best.json: logit_l1, manual3
MIN_TRAIN = 200

# Latest feature row per game computed before first pitch. game_time_utc comes from the
# prediction rows (the feature table has none); MIN over rows, since it is one value.
FEATURES_SQL = """
WITH gt AS (
  SELECT game_pk, MIN(game_time_utc) AS game_time_utc,
         ANY_VALUE(home_team_id) AS home_team_id, ANY_VALUE(away_team_id) AS away_team_id,
         ANY_VALUE(home_team_name) AS home_team_name, ANY_VALUE(away_team_name) AS away_team_name
  FROM `{proj}.{ds}.game_predictions`
  WHERE game_time_utc IS NOT NULL AND game_date <= @d
  GROUP BY game_pk
),
f AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT v.game_pk, v.game_date, v.elo_differential, v.pythag_differential,
           v.sp_quality_composite_diff, v.computed_at,
           ROW_NUMBER() OVER (PARTITION BY v.game_pk ORDER BY v.computed_at DESC) rn
    FROM `{proj}.{ds}.game_v10_features` v JOIN gt USING (game_pk)
    WHERE v.computed_at < gt.game_time_utc AND v.game_date <= @d)
  WHERE rn = 1
)
SELECT f.*, gt.game_time_utc, gt.home_team_id, gt.away_team_id,
       gt.home_team_name, gt.away_team_name,
       g.home_score, g.away_score, g.status, g.game_type
FROM f JOIN gt USING (game_pk)
LEFT JOIN (
  SELECT game_pk, ANY_VALUE(home_score) home_score, ANY_VALUE(away_score) away_score,
         ANY_VALUE(status) status, ANY_VALUE(game_type) game_type
  FROM `{proj}.{ds}.games` GROUP BY game_pk
) g USING (game_pk)
"""


def split_frame(df: pd.DataFrame, target: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows (Final, regular season, strictly before target, decided) and the
    target slate. Also re-applies the pregame guard, so a frame built elsewhere cannot
    smuggle a post-first-pitch feature row in."""
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    ct = pd.to_datetime(df["computed_at"], utc=True)
    gt = pd.to_datetime(df["game_time_utc"], utc=True)
    df = df[ct < gt]
    final = df["status"].fillna("").str.contains("Final") & df["home_score"].notna() \
        & df["away_score"].notna() & (df["home_score"] != df["away_score"])
    reg = df["game_type"].fillna("R").eq("R")
    train = df[final & reg & (df["game_date"] < target)].copy()
    train["y"] = (train["home_score"] > train["away_score"]).astype(int)
    slate = df[df["game_date"] == target].copy()
    return train, slate


def fit(train: pd.DataFrame):
    """Median-impute (train medians), standardise, L1 logistic. Returns a small model dict."""
    from sklearn.linear_model import LogisticRegression

    X = train[FEATURES].astype(float)
    med = X.median()
    X = X.fillna(med).values
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    m = LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=3000)
    m.fit((X - mu) / sd, train["y"].values)
    return {"median": med.to_dict(), "mu": mu.tolist(), "sd": sd.tolist(),
            "intercept": float(m.intercept_[0]), "coef": [float(c) for c in m.coef_[0]],
            "n_train": int(len(train))}


def predict(model: dict, frame: pd.DataFrame) -> np.ndarray:
    X = frame[FEATURES].astype(float).fillna(pd.Series(model["median"])).values
    z = (X - np.array(model["mu"])) / np.array(model["sd"])
    return 1 / (1 + np.exp(-(model["intercept"] + z @ np.array(model["coef"]))))


def tier(p: float) -> str:
    c = max(p, 1 - p)
    return "high" if c >= 0.64 else ("medium" if c >= 0.57 else "low")


def build_rows(model: dict, slate: pd.DataFrame, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    p = predict(model, slate)
    coef_json = json.dumps({"features": FEATURES, "intercept": model["intercept"],
                            "coef": model["coef"], "mu": model["mu"], "sd": model["sd"],
                            "C": C, "penalty": "l1"})
    rows = []
    for (_, r), ph in zip(slate.iterrows(), p):
        rows.append(dict(
            game_pk=int(r.game_pk), game_date=r.game_date,
            home_team_id=int(r.home_team_id), away_team_id=int(r.away_team_id),
            home_team_name=r.home_team_name, away_team_name=r.away_team_name,
            home_win_probability=float(ph), away_win_probability=float(1 - ph),
            predicted_winner=r.home_team_name if ph >= 0.5 else r.away_team_name,
            confidence_tier=tier(ph), model_version=MODEL_VERSION,
            n_train=model["n_train"], coef_json=coef_json,
            elo_differential=_f(r.elo_differential), pythag_differential=_f(r.pythag_differential),
            sp_quality_composite_diff=_f(r.sp_quality_composite_diff),
            features_computed_at=pd.Timestamp(r.computed_at).to_pydatetime(),
            game_time_utc=pd.Timestamp(r.game_time_utc).to_pydatetime(),
            predicted_at=now))
    return rows


def _f(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else float(v)


def write(bq, rows: list[dict], target: date) -> int:
    """DELETE this date's rows for this version, then append with CREATE_NEVER."""
    from google.cloud import bigquery

    tbl = f"{PROJECT}.{DATASET}.{TABLE}"
    bq.query(f"DELETE FROM `{tbl}` WHERE game_date = @d AND model_version = @m",
             job_config=bigquery.QueryJobConfig(query_parameters=[
                 bigquery.ScalarQueryParameter("d", "DATE", target),
                 bigquery.ScalarQueryParameter("m", "STRING", MODEL_VERSION)])).result()
    bq.load_table_from_dataframe(
        pd.DataFrame(rows), tbl,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND",
                                          create_disposition="CREATE_NEVER")).result()
    return len(rows)


def _missing_table(e: Exception) -> bool:
    s = str(e)
    return "Not found: Table" in s or "notFound" in s or "404" in s


def run_slate(target: date, dry_run: bool = False, bq=None) -> dict:
    out = {"step": "logit3", "date": str(target), "table": TABLE, "model_version": MODEL_VERSION}
    if bq is None:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT)
    from google.cloud import bigquery as _bq

    cfg = _bq.QueryJobConfig(query_parameters=[_bq.ScalarQueryParameter("d", "DATE", target)])
    df = bq.query(FEATURES_SQL.format(proj=PROJECT, ds=DATASET), job_config=cfg).to_dataframe()
    train, slate = split_frame(df, target)
    out["n_train"] = int(len(train))
    if len(train) < MIN_TRAIN:
        return {**out, "status": "skipped", "reason": f"only {len(train)} training games"}
    if slate.empty:
        return {**out, "status": "no_games"}
    model = fit(train)
    rows = build_rows(model, slate)
    out.update(games=len(rows), coef=dict(zip(FEATURES, model["coef"])))
    if dry_run:
        return {**out, "status": "dry_run", "sample": [
            {k: (str(v) if isinstance(v, (datetime, date)) else v) for k, v in r.items()
             if k != "coef_json"} for r in rows[:3]]}
    try:
        write(bq, rows, target)
    except Exception as e:
        if _missing_table(e):
            logger.warning("logit3: %s.%s does not exist (CREATE_NEVER); nothing written",
                           DATASET, TABLE)
            return {**out, "status": "table_missing", "error": str(e)[:300]}
        raise
    return {**out, "status": "ok"}
