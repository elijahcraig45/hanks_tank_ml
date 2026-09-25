"""PA-by-PA Monte Carlo game simulator, vectorised across episodes.

Plays real baseball sequentially: away bats the top of each inning, home the bottom,
the home half of the 9th is skipped when the home team already leads, walk-offs end a
half-inning the moment the home team takes the lead, and ties go to extra innings.

All `n_episodes` games are advanced in lockstep with numpy, so 1000 episodes cost about
the same as one Python-level game loop. Correctness is cross-checked against the exact
analytic chain in markov.py -- the two must agree on the run distribution.
"""
from __future__ import annotations
import numpy as np
from . import CLASSES
from .markov import NB, NO, RA, END

MAX_INNINGS = 18


def _sample(cum: np.ndarray, bidx: np.ndarray, rng) -> np.ndarray:
    """Draw an outcome per episode from each episode's current batter."""
    u = rng.random(len(bidx))
    c = cum[bidx]                                    # (n, 8)
    return (u[:, None] >= c).sum(axis=1).clip(0, len(CLASSES) - 1)


def _half_inning(cum_by_inning: np.ndarray, bidx: np.ndarray, rng,
                 stop_when_ahead: np.ndarray | None = None,
                 lead: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Play one half-inning for every episode. Returns (runs, next batter index).

    stop_when_ahead: episodes that end the half the moment `lead` turns positive
                     (walk-off). lead is runs_home - runs_away entering the PA.
    """
    n = len(bidx)
    bases = np.zeros(n, dtype=np.int8)
    outs = np.zeros(n, dtype=np.int8)
    runs = np.zeros(n, dtype=np.int32)
    live = np.ones(n, dtype=bool)
    b = bidx.copy()
    for _ in range(60):
        idx = np.flatnonzero(live)
        if idx.size == 0:
            break
        ev = _sample(cum_by_inning, b[idx], rng)
        bb, oo = bases[idx], outs[idx]
        runs[idx] += RA[bb, oo, ev]
        ended = END[bb, oo, ev]
        bases[idx] = np.where(ended, 0, NB[bb, oo, ev])
        outs[idx] = np.where(ended, 0, NO[bb, oo, ev])
        b[idx] = (b[idx] + 1) % 9
        live[idx] = ~ended
        if stop_when_ahead is not None:
            walk = stop_when_ahead[idx] & ((lead[idx] + runs[idx]) > 0)
            live[idx[walk]] = False
    return runs, b


def simulate_game(home_inning_cum: np.ndarray, away_inning_cum: np.ndarray,
                  n_episodes: int = 1000, seed: int = 0) -> dict:
    """Run n_episodes PA-by-PA games.

    home_inning_cum / away_inning_cum: (n_innings, 9, 8) cumulative outcome
    probabilities -- one 9-batter matrix per inning for the batting side, already
    combined against whoever is pitching that inning.

    Returns home/away run arrays and the derived win probability.
    """
    rng = np.random.default_rng(seed)
    n = n_episodes
    hr = np.zeros(n, dtype=np.int32)
    ar = np.zeros(n, dtype=np.int32)
    hb = np.zeros(n, dtype=np.int64)
    ab = np.zeros(n, dtype=np.int64)
    n_reg = home_inning_cum.shape[0]

    for inn in range(n_reg):
        r, ab = _half_inning(away_inning_cum[inn], ab, rng)      # top half
        ar += r
        if inn == n_reg - 1:
            # home bats the bottom of the 9th only if not already ahead
            need = hr <= ar
            if need.any():
                lead = (hr - ar).astype(np.int32)
                r, nb_ = _half_inning(home_inning_cum[inn], hb, rng,
                                      stop_when_ahead=need, lead=lead)
                hr = hr + np.where(need, r, 0)
                hb = np.where(need, nb_, hb)
        else:
            r, hb = _half_inning(home_inning_cum[inn], hb, rng)
            hr += r

    # regulation totals are kept separately: the analytic chain in markov.py models
    # nine innings only, so this is what a cross-check must compare against
    hr_reg, ar_reg = hr.copy(), ar.copy()

    # extra innings until decided
    last_h, last_a = home_inning_cum[-1], away_inning_cum[-1]
    for _ in range(MAX_INNINGS - n_reg):
        tied = hr == ar
        if not tied.any():
            break
        ti = np.flatnonzero(tied)
        r, nb_ = _half_inning(last_a, ab[ti], rng)
        ar[ti] += r; ab[ti] = nb_
        lead = (hr[ti] - ar[ti]).astype(np.int32)
        r, nb_ = _half_inning(last_h, hb[ti], rng,
                              stop_when_ahead=np.ones(len(ti), bool), lead=lead)
        hr[ti] += r; hb[ti] = nb_
    # anything still tied after the cap is broken by a coin flip
    still = hr == ar
    if still.any():
        hr[still] += rng.random(still.sum()) < 0.52

    return dict(home_runs=hr, away_runs=ar,
                home_runs_reg=hr_reg, away_runs_reg=ar_reg,
                home_win_prob=float(np.clip((hr > ar).mean(), 1e-6, 1 - 1e-6)),
                mean_home=float(hr.mean()), mean_away=float(ar.mean()),
                mean_home_reg=float(hr_reg.mean()), mean_away_reg=float(ar_reg.mean()),
                p_extra=float((hr_reg == ar_reg).mean()),
                n_episodes=n)


def to_cum(inning_probs: list[np.ndarray]) -> np.ndarray:
    """(n_innings, 9, 8) probabilities -> cumulative, for sampling."""
    P = np.asarray(inning_probs, dtype=float)
    P = P / P.sum(axis=-1, keepdims=True)
    return np.cumsum(P, axis=-1)
