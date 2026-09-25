"""Combine two run distributions into P(home win), with global run calibration.

The base-out chain uses conservative advancement assumptions, so its raw run totals
sit below league average. Rather than hand-tuning advancement, a single scalar boosts
hit-class odds and is fitted once against observed run totals -- exactly the quantity
Phase 2 validates on (~4,800 team-game run totals, far more power than 1,300 wins).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from . import CLASSES

HIT_IDX = [CLASSES.index(c) for c in ("1B", "2B", "3B", "HR")]


def apply_scale(probs: np.ndarray, alpha: float) -> np.ndarray:
    """Multiply hit-class odds by alpha and renormalise. alpha=1 is a no-op."""
    if alpha == 1.0:
        return probs
    p = probs.copy()
    p[..., HIT_IDX] *= alpha
    s = p.sum(axis=-1, keepdims=True)
    return p / np.where(s <= 0, 1e-9, s)


def expected_runs(dist: np.ndarray) -> float:
    return float(np.dot(np.arange(len(dist)), dist))


def win_prob(home: np.ndarray, away: np.ndarray, p_extra_home: float = 0.52) -> float:
    """P(home win) from the two independent run distributions.

    Ties go to extra innings; 2026 has zero tied final scores, so the tie mass is
    resolved by p_extra_home rather than split evenly.
    """
    nh, na = len(home), len(away)
    cum_away = np.cumsum(away)
    p_gt = 0.0
    for h in range(1, nh):
        if home[h] <= 0:
            continue
        p_gt += home[h] * cum_away[min(h - 1, na - 1)]
    p_tie = float(np.dot(home[:min(nh, na)], away[:min(nh, na)]))
    return float(np.clip(p_gt + p_tie * p_extra_home, 1e-6, 1 - 1e-6))


def fit_run_scale(sim_fn, targets: np.ndarray, lo: float = 0.6, hi: float = 2.5) -> float:
    """Find alpha so mean simulated runs matches mean observed runs.

    sim_fn(alpha) -> array of expected runs per team-game.
    """
    def f(a):
        return float(np.mean(sim_fn(a)) - np.mean(targets))
    try:
        flo, fhi = f(lo), f(hi)
        if flo > 0 or fhi < 0:
            return 1.0 if abs(flo) > abs(fhi) else hi
        return float(brentq(f, lo, hi, xtol=1e-3))
    except Exception:
        return 1.0
