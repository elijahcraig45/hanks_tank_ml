"""Starter-K spread correction (pa_sim.dists.recalibrate_cdf / apply_calibration)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pa_sim import dists  # noqa: E402

CAL = os.path.join(os.path.dirname(__file__), "..", "src", "pa_sim", "player_calibration.json")


def _pmf(mu=5.0, n=28):
    from scipy.stats import binom
    return binom.pmf(np.arange(n), 24, mu / 24)


def _mean_sd(p):
    k = np.arange(len(p))
    m = (p * k).sum()
    return m, np.sqrt((p * (k - m) ** 2).sum())


def test_identity_when_a_b_are_one():
    p = _pmf()
    assert np.allclose(dists.recalibrate_cdf(p, 1.0, 1.0), p / p.sum())


def test_recal_below_one_widens_and_stays_a_pmf():
    p = _pmf()
    q = dists.recalibrate_cdf(p, 0.82, 0.84)
    assert q.shape == p.shape and (q >= 0).all() and q.sum() == pytest.approx(1.0)
    (m0, s0), (m1, s1) = _mean_sd(p), _mean_sd(q)
    assert s1 > s0 * 1.05                 # wider
    assert abs(m1 - m0) < 0.1             # a ~= b keeps the centre


def test_zero_tail_bins_stay_zero():
    p = np.r_[_pmf(n=15), np.zeros(13)]    # capped pmf padded with zeros
    q = dists.recalibrate_cdf(p, 0.82, 0.84)
    assert np.all(q[15:] == 0)


def test_apply_calibration_is_backward_compatible():
    p = _pmf()
    old = {"thin": 0.9631, "apply_thin": True}
    assert np.allclose(dists.apply_calibration(p, old), dists.thin(p, 0.9631))
    assert np.allclose(dists.apply_calibration(p, {"thin": 0.9, "apply_thin": False}), p / p.sum())
    assert np.allclose(dists.apply_calibration(p, None), p)
    # a blend._stat_cal-style dict: resolved thin, no apply_thin key
    assert np.allclose(dists.apply_calibration(p, {"thin": 0.95}), dists.thin(p, 0.95))


def test_apply_calibration_thins_then_recalibrates():
    p = _pmf()
    c = {"thin": 0.9631, "apply_thin": True, "cdf_recal": {"a": 0.82, "b": 0.84}}
    expect = dists.recalibrate_cdf(dists.thin(p, 0.9631), 0.82, 0.84)
    assert np.allclose(dists.apply_calibration(p, c), expect)


def test_calibration_file_starter_k_entry():
    k = json.load(open(CAL))["starter"]["K"]
    assert k["apply_thin"] and 0 < k["thin"] < 1
    assert 0 < k["cdf_recal"]["a"] < 1 and 0 < k["cdf_recal"]["b"] < 1
    # the flag must agree with the numbers the note reports
    assert k["calibrated"] is ("Passes the gate" in k["note"])
