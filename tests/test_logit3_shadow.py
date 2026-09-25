import os
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import logit3_shadow as L


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d0 = pd.Timestamp("2026-04-01")
    rows = []
    for i in range(n):
        gd = d0 + pd.Timedelta(days=i // 10)
        gt = gd + pd.Timedelta(hours=23)
        elo = rng.normal(0, 40)
        py = rng.normal(0, 0.08)
        sp = rng.normal(0, 10)
        p = 1 / (1 + np.exp(-(0.1 + elo / 80 + py * 4 + sp / 40)))
        hw = rng.random() < p
        rows.append(dict(game_pk=1000 + i, game_date=gd.date(), elo_differential=elo,
                         pythag_differential=py, sp_quality_composite_diff=sp,
                         computed_at=gt - pd.Timedelta(hours=1), game_time_utc=gt,
                         home_team_id=1, away_team_id=2, home_team_name="Home",
                         away_team_name="Away", home_score=5 if hw else 2,
                         away_score=2 if hw else 5, status="Final", game_type="R"))
    return pd.DataFrame(rows)


def test_split_excludes_target_day_and_later_from_training():
    df = _frame()
    target = df.game_date.iloc[250]
    train, slate = L.split_frame(df, target)
    assert (train.game_date < target).all()
    assert (slate.game_date == target).all() and len(slate) == 10


def test_split_drops_rows_computed_after_first_pitch():
    df = _frame()
    target = df.game_date.iloc[250]
    df.loc[df.game_date == target, "computed_at"] = df.loc[df.game_date == target, "game_time_utc"] + pd.Timedelta(minutes=5)
    df.loc[5, "computed_at"] = df.loc[5, "game_time_utc"]          # at first pitch: not pregame
    train, slate = L.split_frame(df, target)
    assert slate.empty
    assert 1005 not in set(train.game_pk)


def test_split_skips_non_final_ties_and_spring():
    df = _frame()
    df.loc[0, "status"] = "Scheduled"
    df.loc[1, ["home_score", "away_score"]] = [3, 3]
    df.loc[2, "game_type"] = "S"
    train, _ = L.split_frame(df, date(2026, 12, 1))
    assert not {1000, 1001, 1002} & set(train.game_pk)


def test_fit_recovers_signs_and_predicts_probabilities():
    df = _frame(2000)
    train, _ = L.split_frame(df, date(2026, 12, 1))
    m = L.fit(train)
    assert all(c > 0 for c in m["coef"])
    p = L.predict(m, train)
    assert ((p > 0) & (p < 1)).all()
    # standardised median imputation: a row of NaNs predicts at the median point
    blank = train.head(1).copy()
    blank[L.FEATURES] = np.nan
    assert 0.3 < L.predict(m, blank)[0] < 0.7


def test_build_rows_contract():
    df = _frame()
    target = df.game_date.iloc[300]
    train, slate = L.split_frame(df, target)
    rows = L.build_rows(L.fit(train), slate)
    r = rows[0]
    for k in ("game_pk", "game_date", "home_team_id", "away_team_id", "home_team_name",
              "away_team_name", "home_win_probability", "away_win_probability",
              "predicted_winner", "confidence_tier", "model_version", "n_train",
              "coef_json", "game_time_utc", "predicted_at"):
        assert k in r
    assert r["model_version"] == "logit3_l1_v1"
    assert abs(r["home_win_probability"] + r["away_win_probability"] - 1) < 1e-12


def test_tier_cutoffs():
    assert L.tier(0.66) == "high" and L.tier(0.30) == "high"
    assert L.tier(0.58) == "medium" and L.tier(0.52) == "low"


class _Job:
    def __init__(self, df=None, exc=None):
        self.df, self.exc = df, exc

    def to_dataframe(self):
        return self.df

    def result(self):
        if self.exc:
            raise self.exc


class _BQ:
    def __init__(self, df, load_exc=None):
        self.df, self.load_exc, self.loads = df, load_exc, []

    def query(self, sql, job_config=None):
        return _Job(self.df) if "SELECT" in sql and "DELETE" not in sql else _Job()

    def load_table_from_dataframe(self, df, tbl, job_config=None):
        self.loads.append((tbl, job_config))
        return _Job(exc=self.load_exc)


def test_run_slate_uses_create_never_and_reports_missing_table():
    pytest.importorskip("google.cloud.bigquery")
    df = _frame()
    target = df.game_date.iloc[300]
    bq = _BQ(df)
    early = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out = L.run_slate(target, bq=bq, now=early)
    assert out["status"] == "ok"
    cfg = bq.loads[0][1]
    assert cfg.create_disposition == "CREATE_NEVER"
    bq2 = _BQ(df, load_exc=Exception("404 Not found: Table hankstank:mlb_2026_season.game_predictions_logit3"))
    assert L.run_slate(target, bq=bq2, now=early)["status"] == "table_missing"


def test_run_slate_is_append_only_skips_started_games_and_filters_game_pks():
    pytest.importorskip("google.cloud.bigquery")
    df = _frame()
    target = df.game_date.iloc[300]
    bq = _BQ(df)
    sqls = []
    orig = bq.query
    bq.query = lambda sql, job_config=None: (sqls.append(sql), orig(sql, job_config))[1]
    # after every game on the slate has started: nothing written
    late = datetime(2027, 1, 1, tzinfo=timezone.utc)
    assert L.run_slate(target, bq=bq, now=late)["status"] == "no_upcoming_games"
    assert not bq.loads
    early = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pk = int(df[df.game_date == target].game_pk.iloc[0])
    out = L.run_slate(target, bq=bq, now=early, game_pks=[pk])
    assert out["games"] == 1 and out["status"] == "ok"
    assert not any("DELETE" in s for s in sqls)


def test_run_slate_skips_small_samples_and_dry_run_writes_nothing():
    pytest.importorskip("google.cloud.bigquery")
    df = _frame(150)
    bq = _BQ(df)
    assert L.run_slate(df.game_date.iloc[-1], bq=bq)["status"] == "skipped"
    df = _frame()
    bq = _BQ(df)
    assert L.run_slate(df.game_date.iloc[300], dry_run=True, bq=bq)["status"] == "dry_run"
    assert not bq.loads
