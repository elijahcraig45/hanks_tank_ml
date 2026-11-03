"""The margin ridge is the football headline (NFL from 2026 week 9, CFB from week 11).

What has to hold: the ridge's rows are the ones written to game_predictions and to the ridge's own table; the former headline (XGBoost) goes only to
game_predictions_xgb_shadow; a failing shadow write or score never costs the headline anything; score mode scores both the headline and the XGBoost table.

Mocks only: no network, no BigQuery."""
import os
import sys
import unittest.mock

import pandas as pd

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "nfl"))

import predict_nfl  # noqa: E402
from test_nfl_drive_sim import BODY as NFL_BODY, _load_main as _load_nfl, _Req  # noqa: E402

RIDGE_TABLE, XGB_TABLE = "game_predictions_ridge_shadow", "game_predictions_xgb_shadow"
PRIOR_TABLE = "game_predictions_ridge_prior_shadow"


def _frames():
    """One distinct frame per model, so a test can tell whose rows were written where."""
    return {m: pd.DataFrame({"game_id": ["g1"], "home_win_probability": [p], "confidence_tier": ["low"], "model_version": [m]})
            for m, p in (("ridge", 0.61), ("xgb", 0.52), ("ridge_prior", 0.63))}


def _versions(calls):
    return {c.args[2]: c.args[0]["model_version"].iloc[0] for c in calls}


# --------------------------------------------------------------------------------------------------------------------------- constants
def test_both_sports_name_the_ridge_as_headline_and_give_the_xgboost_its_own_table():
    sys.path.insert(0, os.path.join(SRC, "cfb"))
    import pipeline

    for mod in (predict_nfl, pipeline):
        assert mod.HEADLINE_KEY == "ridge"
        assert mod.SHADOW_TABLE == RIDGE_TABLE and mod.XGB_SHADOW_TABLE == XGB_TABLE


# --------------------------------------------------------------------------------------------------------------------------------- NFL
def _nfl_predict(monkeypatch, predict=None, upsert=None):
    import bq_io
    import model_control

    main = _load_nfl()
    for k in ("NFL_DRIVE_SIM_SHADOW", "NFL_RIDGE_SHADOW", "NFL_XGB_REG_SHADOW", "FPI_SNAPSHOT"):
        monkeypatch.delenv(k, raising=False)
    frames = _frames()
    predict = predict or (lambda season, week, model="xgb", **kw: frames[model])
    with unittest.mock.patch.object(model_control, "get_state", return_value=None), \
         unittest.mock.patch.object(predict_nfl, "predict_week", side_effect=predict) as pw, \
         unittest.mock.patch.object(bq_io, "upsert_week", side_effect=upsert, return_value=1) as up, \
         unittest.mock.patch.object(bq_io, "ensure_dataset"):
        out, status = main.nfl_pipeline(_Req(NFL_BODY))
    return out, status, pw, up


def test_nfl_publishes_the_ridge_and_files_the_xgboost_under_its_own_table(monkeypatch):
    out, status, pw, up = _nfl_predict(monkeypatch)
    assert status == 200 and out["status"] == "ok"
    assert [c.kwargs.get("model") for c in pw.call_args_list] == ["ridge", "xgb"]
    assert _versions(up.call_args_list) == {"game_predictions": "ridge", RIDGE_TABLE: "ridge", XGB_TABLE: "xgb"}
    assert out["steps"]["predicted"] == 1 and out["steps"]["shadow_xgb"] == 1


def test_nfl_a_failing_xgboost_never_costs_the_headline(monkeypatch):
    frames = _frames()

    def predict(season, week, model="xgb", **kw):
        if model == "xgb":
            raise SystemExit("no EPA")       # _upcoming / _require_epa style failure
        return frames[model]

    out, status, pw, up = _nfl_predict(monkeypatch, predict=predict)
    assert status == 200 and out["status"] == "ok" and out["steps"]["predicted"] == 1
    assert out["steps"]["shadow_xgb"] == {"error": "no EPA"}
    assert [c.args[2] for c in up.call_args_list] == ["game_predictions", RIDGE_TABLE]


def test_nfl_a_failing_ridge_table_write_never_costs_the_headline(monkeypatch):
    def upsert(df, dataset, table, *a, **kw):
        if table == RIDGE_TABLE:
            raise RuntimeError("quota")
        return 1

    out, status, pw, up = _nfl_predict(monkeypatch, upsert=upsert)
    assert status == 200 and out["status"] == "ok" and out["steps"]["predicted"] == 1
    assert out["steps"]["ridge_table"] == {"error": "quota"} and out["steps"]["shadow_xgb"] == 1


def _nfl_score(monkeypatch, query):
    import bq_io

    main = _load_nfl()
    with unittest.mock.patch.object(bq_io, "query", side_effect=query) as q:
        out, status = main.nfl_pipeline(_Req({"mode": "score"}))
    return out, status, q


def test_nfl_score_updates_the_headline_and_the_xgboost_table(monkeypatch):
    out, status, q = _nfl_score(monkeypatch, lambda sql: None)
    assert status == 200 and out["steps"] == {"scored": "ok", "scored_xgb": "ok"}
    sqls = [c.args[0] for c in q.call_args_list]
    assert ".game_predictions` p" in sqls[0] and f".{XGB_TABLE}` p" in sqls[1]


def test_nfl_a_failing_xgboost_score_never_costs_the_headline_its_scores(monkeypatch):
    def query(sql):
        if XGB_TABLE in sql:
            raise RuntimeError("not found")

    out, status, q = _nfl_score(monkeypatch, query)
    assert status == 200 and out["steps"]["scored"] == "ok" and out["steps"]["scored_xgb"] == {"error": "not found"}


# --------------------------------------------------------------------------------------------------------------------------------- CFB
def _cfb_predict(monkeypatch, predict=None, replace=None):
    import backfill_cfb
    import model_control
    import pipeline
    from test_cfb_drive_sim import _clear_env, _load_main as _load_cfb

    main = _load_cfb()
    _clear_env(monkeypatch)
    monkeypatch.delenv("FPI_SNAPSHOT", raising=False)
    frames = _frames()
    predict = predict or (lambda season, week, model="xgb", **kw: frames[model])
    with unittest.mock.patch.object(model_control, "get_state", return_value=None), \
         unittest.mock.patch.object(pipeline, "predict_week", side_effect=predict) as pw, \
         unittest.mock.patch.object(backfill_cfb, "replace_game_ids", side_effect=replace, return_value=1) as rep, \
         unittest.mock.patch.object(backfill_cfb, "ensure_datasets"):
        out, status = main.cfb_pipeline(_Req({"mode": "predict_week", "season": 2026, "week": 11}))
    return out, status, pw, rep


def test_cfb_publishes_the_ridge_and_files_the_xgboost_under_its_own_table(monkeypatch):
    out, status, pw, rep = _cfb_predict(monkeypatch)
    assert status == 200 and out["status"] == "ok"
    assert [c.kwargs.get("model") for c in pw.call_args_list] == ["ridge", "xgb", "ridge_prior"]
    assert _versions(rep.call_args_list) == {"game_predictions": "ridge", RIDGE_TABLE: "ridge", XGB_TABLE: "xgb", PRIOR_TABLE: "ridge_prior"}
    for c in rep.call_args_list:                      # every table keeps the partitioning the headline always had
        assert c.kwargs["partition_field"] == "game_date" and c.kwargs["cluster_fields"] == ["season", "division"]


def test_cfb_a_failing_xgboost_never_costs_the_headline(monkeypatch):
    frames = _frames()

    def predict(season, week, model="xgb", **kw):
        if model == "xgb":
            raise RuntimeError("too few rows")
        return frames[model]

    out, status, pw, rep = _cfb_predict(monkeypatch, predict=predict)
    assert status == 200 and out["status"] == "ok" and out["steps"]["predicted"] == 1
    assert out["steps"]["shadow_xgb"] == {"error": "too few rows"}
    assert [c.args[2] for c in rep.call_args_list] == ["game_predictions", RIDGE_TABLE, PRIOR_TABLE]


def test_cfb_score_updates_both_tables_and_each_fails_alone(monkeypatch):
    from google.cloud import bigquery
    from test_cfb_drive_sim import _load_main as _load_cfb

    main = _load_cfb()
    client = unittest.mock.MagicMock()

    def query(sql):
        if XGB_TABLE in sql:
            raise RuntimeError("not found")
        job = unittest.mock.MagicMock()
        job.num_dml_affected_rows = 7
        return job

    client.query.side_effect = query
    steps = {}
    with unittest.mock.patch.object(bigquery, "Client", return_value=client):
        main._score_predictions(steps)
    assert steps == {"scored": 7, "scored_xgb": {"error": "not found"}}
    assert [XGB_TABLE in c.args[0] for c in client.query.call_args_list] == [False, True]
