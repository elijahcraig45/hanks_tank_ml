"""The offline demo (demo/nfl_backtest.py) reproduces the numbers that demo/README.md and docs/EXPERIMENT_LOG.md quote.

It runs the real backtest on the committed nflverse snapshot, in about five seconds. No network, no BigQuery."""
import importlib.util
import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("nfl_backtest", os.path.join(HERE, "..", "demo", "nfl_backtest.py"))
demo = importlib.util.module_from_spec(_spec)
sys.modules["nfl_backtest"] = demo   # @dataclass looks the module up here
_spec.loader.exec_module(demo)

EXACT = 5e-5   # Elo, the ridge and Vegas are deterministic numpy
XGB = 2e-3     # XGBoost may differ slightly across platforms


@pytest.fixture(scope="module")
def result():
    return demo.run(weekly=False)


@pytest.fixture(scope="module")
def summary(result):
    return demo.summary(result.preds)


def test_the_snapshot_holds_every_completed_game_from_1999_to_2025():
    raw = pd.read_csv(demo.SNAPSHOT)
    assert (raw.season.min(), raw.season.max()) == (1999, 2025)
    assert len(raw) == 7276 and raw.result.notna().all()
    assert not raw.game_id.duplicated().any()


def test_tuning_on_2010_2016_recovers_the_settings_production_ships(result):
    assert demo.ships(result.ridge)
    assert result.n_games == 7261 and result.n_features == 84


def test_every_scored_game_has_all_four_predictions(result):
    preds = result.preds
    assert len(preds) == 2202 + 284
    assert preds[list(demo.MODELS)].notna().all().all()


@pytest.mark.parametrize("period, model, log_loss, tol", [
    ("2017-2024 walk-forward", "ridge", 0.6342, EXACT),
    ("2017-2024 walk-forward", "elo", 0.6462, EXACT),
    ("2017-2024 walk-forward", "vegas", 0.6086, EXACT),
    ("2017-2024 walk-forward", "xgb", 0.6456, XGB),
    ("2025 holdout", "ridge", 0.6339, EXACT),
    ("2025 holdout", "xgb", 0.6426, XGB),
])
def test_published_log_losses(summary, period, model, log_loss, tol):
    assert summary[period]["models"][model]["log_loss"] == pytest.approx(log_loss, abs=tol)


def test_the_ridge_beats_xgboost_over_2017_2024_with_an_interval_below_zero(summary):
    d, lo, hi = summary["2017-2024 walk-forward"]["ridge_vs_xgb"]
    assert d < 0 and hi < 0
