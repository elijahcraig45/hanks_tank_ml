"""Regression tests for the 2026-09-25 football write incident.

A {"dry_run": true} request ran for real (the handlers never read it) and replaced a
week of production predictions: the predict paths re-predicted games already under way,
and the week-wide DELETE removed pregame rows it never rewrote.
"""
import importlib.util
import os
import sys
import unittest.mock

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, os.path.join(SRC, "nfl"))

import bq_io  # noqa: E402
import predict_nfl  # noqa: E402


def _load_main(sport: str):
    """Load src/<sport>/main.py under a unique name; both files are called main.py."""
    path = os.path.join(SRC, sport, "main.py")
    spec = importlib.util.spec_from_file_location(f"{sport}_main_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Req:
    def __init__(self, body):
        self._body = body

    def get_json(self, silent=True):
        return self._body


def _schedule():
    return pd.DataFrame({
        "game_id": ["2026_03_ATL_GB", "2026_03_HOU_IND", "2026_03_PHI_CHI"],
        "season": [2026, 2026, 2026],
        "week": [3, 3, 3],
        "gameday": ["2026-09-24", "2026-09-27", "2026-09-28"],
        "gametime": ["20:15", "13:00", "20:15"],
        "result": [-21.0, np.nan, np.nan],
    })


def test_upcoming_drops_finished_and_started_games():
    # Sunday 14:00 ET: ATL@GB is final, HOU@IND kicked off at 13:00, PHI@CHI is Monday.
    now = pd.Timestamp("2026-09-27 18:00", tz="UTC")
    up = predict_nfl._upcoming(_schedule(), 2026, 3, now=now)
    assert up["game_id"].tolist() == ["2026_03_PHI_CHI"]
    assert up["home_won"].isna().all()


def test_upcoming_keeps_the_whole_week_before_the_first_kickoff():
    now = pd.Timestamp("2026-09-23 10:00", tz="UTC")
    sched = _schedule().assign(result=np.nan)
    up = predict_nfl._upcoming(sched, 2026, 3, now=now)
    assert len(up) == 3


def test_upsert_week_deletes_only_the_games_it_writes():
    df = pd.DataFrame({"game_id": ["2026_03_PHI_CHI"], "season": [2026], "week": [3]})
    fake = unittest.mock.MagicMock()
    with unittest.mock.patch.object(bq_io, "client", return_value=fake), \
         unittest.mock.patch.object(bq_io, "load_table", return_value=1) as load:
        assert bq_io.upsert_week(df, "nfl_season", "game_predictions", 2026, 3) == 1
    sql = fake.query.call_args[0][0]
    params = {p.name: p for p in fake.query.call_args[1]["job_config"].query_parameters}
    assert "game_id IN UNNEST(@ids)" in sql
    assert params["ids"].values == ["2026_03_PHI_CHI"]
    load.assert_called_once()


def test_upsert_week_with_nothing_to_write_deletes_nothing():
    fake = unittest.mock.MagicMock()
    with unittest.mock.patch.object(bq_io, "client", return_value=fake):
        assert bq_io.upsert_week(pd.DataFrame(columns=["game_id"]), "d", "t", 2026, 3) == 0
    fake.query.assert_not_called()


def test_dry_run_is_refused_for_modes_that_write():
    for sport, fn, mode in [("nfl", "nfl_pipeline", "ingest"), ("cfb", "cfb_pipeline", "ingest"),
                            ("nfl", "nfl_pipeline", "score"), ("cfb", "cfb_pipeline", "backfill")]:
        main = _load_main(sport)
        body, status = getattr(main, fn)(_Req({"mode": mode, "dry_run": True}))
        assert status == 400, (sport, mode)
        assert "dry_run" in body["error"]
