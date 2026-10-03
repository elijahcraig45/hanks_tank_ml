"""The regularized-XGBoost shadow: its settings, its flags, that it writes only to its own table, and that it can never cost the headline anything.

Synthetic frames and mocks only: no network, no BigQuery."""
import os
import sys
import unittest.mock

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "nfl"))

import models  # noqa: E402
import predict_nfl  # noqa: E402
from test_nfl_drive_sim import BODY, _ctl_state, _load_main, _run_ctl  # noqa: E402  (the control-plane harness the other shadows use)

SHADOW_TABLE = "game_predictions_xgb_reg_shadow"


# ----------------------------------------------------------------------------------------------------------------------------- settings
def test_the_regularized_model_differs_from_the_headline_in_exactly_the_three_documented_settings():
    a, b = models.build_xgb().get_params(), models.build_xgb_reg().get_params()
    same = lambda x, y: x == y or (x != x and y != y)  # noqa: E731  XGBoost's `missing` is NaN, and NaN is not equal to itself
    changed = {k for k in a if not same(a[k], b[k])}
    assert changed == {"learning_rate", "n_estimators", "min_child_weight"}
    assert (b["learning_rate"], b["n_estimators"], b["min_child_weight"]) == (0.017, 131, 26)
    assert models.build_xgb().get_params()["n_estimators"] == 400  # the headline is untouched


def test_the_shadow_has_its_own_table_and_model_version():
    assert predict_nfl.XGB_REG_SHADOW_TABLE == SHADOW_TABLE
    assert predict_nfl.XGB_REG_MODEL_VERSION not in (predict_nfl.MODEL_VERSION, predict_nfl.RIDGE_MODEL_VERSION)


# -------------------------------------------------------------------------------------------------------------------------------- flags
def test_off_by_default_on_by_request_or_deployment(monkeypatch):
    main = _load_main()
    monkeypatch.delenv("NFL_XGB_REG_SHADOW", raising=False)
    assert not main._xgb_reg_enabled({})
    assert main._xgb_reg_enabled({"shadow_xgb_reg": True})
    monkeypatch.setenv("NFL_XGB_REG_SHADOW", "1")
    assert main._xgb_reg_enabled({})
    monkeypatch.setenv("NFL_XGB_REG_SHADOW", "0")
    assert not main._xgb_reg_enabled({})


def test_the_deploy_script_turns_it_on_with_the_other_shadows():
    text = open(os.path.join(HERE, "..", "scripts", "gcp", "nfl", "deploy_nfl.sh")).read()
    assert "NFL_XGB_REG_SHADOW=1" in text


# ------------------------------------------------------------------------------------------------------------- the handler writes it safely
def test_when_enabled_it_predicts_with_its_model_and_writes_only_its_own_table(monkeypatch):
    monkeypatch.setenv("NFL_XGB_REG_SHADOW", "1")
    out, status, pw, sim, up, *_ = _run_ctl(BODY, monkeypatch)
    monkeypatch.setenv("NFL_XGB_REG_SHADOW", "1")
    out, status, pw, sim, up, *_ = _run_ctl({**BODY, "shadow_xgb_reg": True}, monkeypatch)
    assert status == 200 and out["status"] == "ok"
    assert [c.kwargs.get("model") for c in pw.call_args_list] == [None, "xgb_reg"]
    assert [c.args[2] for c in up.call_args_list] == ["game_predictions", SHADOW_TABLE]
    assert out["steps"]["shadow_xgb_reg"] == 2


def test_when_not_asked_for_nothing_extra_is_predicted_or_written(monkeypatch):
    monkeypatch.delenv("NFL_XGB_REG_SHADOW", raising=False)
    out, status, pw, sim, up, *_ = _run_ctl(BODY, monkeypatch)
    assert [c.kwargs.get("model") for c in pw.call_args_list] == [None]
    assert [c.args[2] for c in up.call_args_list] == ["game_predictions"]
    assert "shadow_xgb_reg" not in out["steps"]


def test_a_paused_shadow_is_skipped_like_a_disabled_one(monkeypatch):
    out, status, pw, sim, up, *_ = _run_ctl({**BODY, "shadow_xgb_reg": True}, monkeypatch, _ctl_state({"target": "xgb_reg", "run_state": "paused"}))
    assert status == 200 and out["status"] == "ok"
    assert [c.kwargs.get("model") for c in pw.call_args_list] == [None] and [c.args[2] for c in up.call_args_list] == ["game_predictions"]


def test_a_failing_shadow_never_costs_the_headline_anything(monkeypatch):
    import bq_io
    import model_control
    main = _load_main()
    rows = pd.DataFrame({"game_id": ["a"], "home_win_probability": [0.6], "confidence_tier": ["high"]})

    def predict(season, week, model="xgb", **kw):
        if model == "xgb_reg":
            raise RuntimeError("boom")
        return rows

    class Req:
        def get_json(self, silent=True):
            return {**BODY, "shadow_xgb_reg": True}

    with unittest.mock.patch.object(model_control, "get_state", return_value=None), \
         unittest.mock.patch.object(predict_nfl, "predict_week", side_effect=predict), \
         unittest.mock.patch.object(bq_io, "upsert_week", return_value=1) as up, unittest.mock.patch.object(bq_io, "ensure_dataset"):
        out, status = main.nfl_pipeline(Req())
    assert status == 200 and out["status"] == "ok" and out["steps"]["predicted"] == 1                # the headline was written
    assert out["steps"]["shadow_xgb_reg"] == {"error": "boom"}
    assert [c.args[2] for c in up.call_args_list] == ["game_predictions"]                           # and nothing half-written to the shadow table


# -------------------------------------------------------------------------------------------------- the prediction path uses the right builder
class _FakeModel:
    def __init__(self, tag):
        self.tag = tag

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = 0.7 if self.tag == "reg" else 0.55
        return np.column_stack([1 - np.full(len(X), p), np.full(len(X), p)])


def _predict(model, monkeypatch):
    feats = pd.DataFrame({"game_id": ["g1", "g2", "g3", "g4"], "game_date": pd.to_datetime(["2026-09-01"] * 2 + ["2026-10-04"] * 2), "season": 2026, "week": [1, 1, 4, 4],
                          "home_team": ["A", "B", "C", "D"], "away_team": ["B", "A", "D", "C"], "home_won": [1.0, 0.0, np.nan, np.nan], "f1": [0.1, 0.2, 0.3, 0.4]})
    for col in ("elo_differential", "home_pythag_season", "away_pythag_season", "pythag_differential", "home_point_diff_3g", "away_point_diff_3g",
                "home_current_streak", "away_current_streak", "h2h_win_pct", "spread_line", "net_epa_8g", "home_off_epa_play_8g", "away_off_epa_play_8g"):
        feats[col] = 0.0
    feats["elo_home_win_prob"], feats["is_divisional"] = 0.5, 0
    monkeypatch.setattr(predict_nfl, "build_xgb", lambda: _FakeModel("headline"))
    monkeypatch.setattr(predict_nfl, "build_xgb_reg", lambda: _FakeModel("reg"))
    monkeypatch.setattr(predict_nfl, "build_features", lambda combined, epa=None: feats)
    monkeypatch.setattr(predict_nfl, "feature_columns", lambda f, include_market=False: ["f1"])
    monkeypatch.setattr(predict_nfl, "_upcoming", lambda schedule, season, week, now=None: feats[feats.home_won.isna()])
    monkeypatch.setattr(predict_nfl, "_require_epa", lambda epa, season: epa)
    monkeypatch.setattr(predict_nfl, "_epa_era", lambda played, epa: played)
    monkeypatch.setattr(predict_nfl, "_epa_lag_weeks", lambda *a: 0)
    played = feats[feats.home_won.notna()]
    return predict_nfl.predict_week(2026, 4, model=model, played=played, schedule=pd.DataFrame(), epa=pd.DataFrame({"x": [1]}))


def test_predict_week_with_xgb_reg_uses_the_regularized_builder_and_its_version(monkeypatch):
    rows = _predict("xgb_reg", monkeypatch)
    assert set(rows["model_version"]) == {predict_nfl.XGB_REG_MODEL_VERSION} and list(rows["home_win_probability"].round(2)) == [0.7, 0.7]


def test_the_headline_path_is_unchanged(monkeypatch):
    rows = _predict("xgb", monkeypatch)
    assert set(rows["model_version"]) == {predict_nfl.MODEL_VERSION} and list(rows["home_win_probability"].round(2)) == [0.55, 0.55]
