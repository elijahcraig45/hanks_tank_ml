"""24-state base-out Markov chain -> exact per-inning run distribution.

Analytic rather than Monte Carlo: the edge being measured is ~0.005 nats wide, so
sampling noise would swamp it, and this vectorises cleanly.

State is (bases, outs) with bases a 3-bit occupancy mask (bit0=1st, bit1=2nd,
bit2=3rd). Advancement uses the standard simplified assumptions (runners advance one
base on a single, two on a double, all score on a triple). That is deliberately
slightly conservative on scoring; `game.py` fits a single global run-scaling factor
against observed run totals to absorb the bias, which is what Phase 2 validates.
"""
from __future__ import annotations
import numpy as np
from . import CLASSES

NB = np.zeros((8, 3, 8), dtype=np.int8)   # new bases
NO = np.zeros((8, 3, 8), dtype=np.int8)   # new outs
RA = np.zeros((8, 3, 8), dtype=np.int8)   # runs added
END = np.zeros((8, 3, 8), dtype=bool)     # inning ends (3 outs)


def _build():
    for bases in range(8):
        b1, b2, b3 = bases & 1, (bases >> 1) & 1, (bases >> 2) & 1
        for outs in range(3):
            for ei, ev in enumerate(CLASSES):
                if ev == "BB":                      # force advance only
                    if not b1:
                        nb, r = bases | 1, 0
                    elif not b2:
                        nb, r = 1 | 2 | (b3 << 2), 0
                    elif not b3:
                        nb, r = 1 | 2 | 4, 0
                    else:
                        nb, r = 1 | 2 | 4, 1        # bases loaded, runner forced home
                    no = outs
                elif ev == "1B":
                    nb, r, no = 1 | (b1 << 1) | (b2 << 2), b3, outs
                elif ev == "2B":
                    nb, r, no = 2 | (b1 << 2), b2 + b3, outs
                elif ev == "3B":
                    nb, r, no = 4, b1 + b2 + b3, outs
                elif ev == "HR":
                    nb, r, no = 0, 1 + b1 + b2 + b3, outs
                elif ev in ("K", "OUT"):
                    nb, r, no = bases, 0, outs + 1
                elif ev == "DP":
                    if b1 and outs <= 1:
                        nb, r, no = bases & ~1, 0, outs + 2
                    else:
                        nb, r, no = bases, 0, outs + 1
                else:
                    raise ValueError(ev)
                NB[bases, outs, ei] = nb
                RA[bases, outs, ei] = r
                if no >= 3:
                    END[bases, outs, ei] = True
                    NO[bases, outs, ei] = 0
                else:
                    NO[bases, outs, ei] = no


_build()
MAX_R = 12          # runs per half-inning are capped here; P(>12) is negligible
MAX_PA = 24


def half_inning(order_probs: np.ndarray) -> np.ndarray:
    """Run distribution for one half-inning, for every possible leadoff batter.

    order_probs: (9, 8) outcome probabilities for the nine batters, in lineup order,
                 already combined against whoever is pitching this inning.
    returns:     (9, MAX_R+1, 9) -> [leadoff_idx, runs, next_leadoff_idx]
    """
    assert order_probs.shape == (9, len(CLASSES))
    live = np.zeros((9, 8, 3, MAX_R + 1))
    live[np.arange(9), 0, 0, 0] = 1.0            # bases empty, 0 outs, 0 runs
    done = np.zeros((9, MAX_R + 1, 9))

    bases_i, outs_i, runs_i = np.meshgrid(np.arange(8), np.arange(3),
                                          np.arange(MAX_R + 1), indexing="ij")
    bflat, oflat, rflat = bases_i.ravel(), outs_i.ravel(), runs_i.ravel()

    for t in range(MAX_PA):
        if live.sum() < 1e-12:
            break
        new_live = np.zeros_like(live)
        for s in range(9):
            batter = (s + t) % 9
            p = order_probs[batter]
            cur = live[s].reshape(-1)
            if cur.sum() < 1e-15:
                continue
            nxt = (s + t + 1) % 9
            for ei in range(len(CLASSES)):
                if p[ei] <= 0:
                    continue
                mass = cur * p[ei]
                nz = mass > 1e-15
                if not nz.any():
                    continue
                bb, oo, rr = bflat[nz], oflat[nz], rflat[nz]
                m = mass[nz]
                ra = RA[bb, oo, ei]
                r2 = np.minimum(rr + ra, MAX_R)
                end = END[bb, oo, ei]
                if end.any():
                    np.add.at(done, (s, r2[end], nxt), m[end])
                ne = ~end
                if ne.any():
                    np.add.at(new_live, (s, NB[bb[ne], oo[ne], ei],
                                         NO[bb[ne], oo[ne], ei], r2[ne]), m[ne])
        live = new_live
    # any mass still live after MAX_PA (vanishing) is closed out where it stands
    if live.sum() > 0:
        for s in range(9):
            tot = live[s].sum(axis=(0, 1))
            done[s, :, (s + MAX_PA) % 9] += tot
    return done


def game_runs(inning_order_probs: list[np.ndarray], start_idx: int = 0) -> np.ndarray:
    """Run distribution for a full 9-inning side.

    inning_order_probs: one (9, 8) matrix per inning (pitcher may change per inning).
    returns: 1-D array over total runs 0..(9*MAX_R), summing to 1.
    """
    n_inn = len(inning_order_probs)
    cum = np.zeros((9, n_inn * MAX_R + 1))
    cum[start_idx, 0] = 1.0
    for probs in inning_order_probs:
        hi = half_inning(probs)                       # (9, MAX_R+1, 9)
        nxt = np.zeros_like(cum)
        for s in range(9):
            w = cum[s]
            if w.sum() < 1e-14:
                continue
            # convolve this inning's runs onto the running total
            for r in range(MAX_R + 1):
                col = hi[s, r]                        # (9,) over next leadoff
                if col.sum() < 1e-15:
                    continue
                shifted = np.zeros_like(w)
                if r == 0:
                    shifted = w
                else:
                    shifted[r:] = w[:-r]
                nxt += np.outer(col, shifted)
        cum = nxt
    total = cum.sum(axis=0)
    s = total.sum()
    return total / (s if s > 0 else 1.0)
