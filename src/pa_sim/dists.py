"""Distribution summaries of simulator output, in the one shape every table uses (Dist).

    Dist = {mean, sd, p05, p25, p50, p75, p95, min, max, n}

Percentiles of a count are integers: the smallest value whose CDF reaches the level
(numpy's "inverted_cdf"). `n` is the number of simulated games, not an effective sample
size, even when the episodes carry tilt weights.

The per-lane arrays come from v2.simulate(stats=True): one lane is one simulated game,
lanes for game i are gi == i. Batter arrays are indexed (lane, batting side, slot) with
side 0 = away; starter arrays (lane, defensive side) with 0 = the AWAY starter.
"""
from __future__ import annotations

import json

import numpy as np

QS = (0.05, 0.25, 0.50, 0.75, 0.95)
QNAMES = ("p05", "p25", "p50", "p75", "p95")
DIST_FIELDS = ("mean", "sd") + QNAMES + ("min", "max")

# stat -> (simulate() key, clip cap). Caps are far above anything a game produces; they
# only bound the pmf arrays.
BATTER_STATS = {"PA": "bpa", "H": "bh", "HR": "bhr", "TB": "btb", "BB": "bbb", "K": "bk"}
STARTER_STATS = {"K": "sp_k", "BF": "sp_bf", "IP_outs": "sp_outs", "ER": "sp_r",
                 "H_allowed": "sp_h", "BB_allowed": "sp_bb"}
CAPS = {"PA": 12, "H": 8, "HR": 5, "TB": 24, "BB": 8, "K": 8,
        "sp_K": 27, "BF": 45, "IP_outs": 30, "ER": 25, "H_allowed": 25, "BB_allowed": 15}
TOTAL_LINES = (6.5, 7.5, 8.5, 9.5, 10.5, 11.5)


def dist_from_pmf(pmf, support=None, n: int | None = None) -> dict:
    """Dist of a discrete distribution given as probabilities over `support`."""
    p = np.asarray(pmf, float)
    x = np.arange(len(p)) if support is None else np.asarray(support, float)
    tot = p.sum()
    if not np.isfinite(tot) or tot <= 0:
        return {**{k: None for k in DIST_FIELDS}, "n": n}
    p = p / tot
    mean = float((p * x).sum())
    sd = float(np.sqrt(max((p * (x - mean) ** 2).sum(), 0.0)))
    cdf = np.cumsum(p)
    out = {"mean": mean, "sd": sd}
    for name, q in zip(QNAMES, QS):
        j = int(np.searchsorted(cdf, q - 1e-12, side="left"))
        out[name] = int(round(x[min(j, len(x) - 1)]))
    nz = np.nonzero(p > 0)[0]
    out["min"] = int(round(x[nz[0]])); out["max"] = int(round(x[nz[-1]]))
    out["n"] = n
    return out


def dist_from_samples(values, weights=None, n: int | None = None) -> dict:
    """Dist of integer samples, optionally weighted (weights need not sum to 1)."""
    v = np.asarray(values).astype(np.int64)
    lo = int(v.min())
    w = None if weights is None else np.asarray(weights, float)
    pmf = np.bincount(v - lo, weights=w).astype(float)
    return dist_from_pmf(pmf, np.arange(lo, lo + len(pmf)), n=len(v) if n is None else n)


def p_at_least_1(pmf) -> float:
    p = np.asarray(pmf, float)
    return float(1.0 - p[0] / p.sum())


def pmfs(arr: np.ndarray, gi: np.ndarray, G: int, cap: int) -> np.ndarray:
    """Per-game pmf of a per-lane count array (lane, *rest) -> (G, *rest, cap+1)."""
    a = np.clip(np.asarray(arr), 0, cap).astype(np.int64)
    rest = a.shape[1:]
    R = int(np.prod(rest)) if rest else 1
    flat = (np.asarray(gi, np.int64)[:, None] * R + np.arange(R)[None, :]) * (cap + 1) + a.reshape(len(a), R)
    cnt = np.bincount(flat.ravel(), minlength=G * R * (cap + 1)).astype(float)
    per_game = np.bincount(np.asarray(gi, np.int64), minlength=G).astype(float)
    out = cnt.reshape(G, R, cap + 1) / np.maximum(per_game, 1)[:, None, None]
    return out.reshape((G,) + tuple(rest) + (cap + 1,))


def thin(pmf, m: float) -> np.ndarray:
    """Binomial thinning: keep each counted event with probability m (0 < m <= 1).

    The honest way to scale down an over-predicted count: the result is still a count
    distribution, its mean is m x the input mean, and P(0) rises accordingly."""
    from scipy.stats import binom

    p = np.asarray(pmf, float)
    if m >= 1.0:
        return p / p.sum()
    k = np.arange(len(p))
    M = binom.pmf(k[None, :], k[:, None], m)          # M[k, j] = P(j kept | k events)
    q = p @ M
    return q / q.sum()


def tilt_weights(total: np.ndarray, target: float) -> tuple[np.ndarray, float]:
    """Per-episode weights w ~ exp(theta * total) giving weighted mean total == target.

    The exponential tilt of the total-runs pmf (blend.tilt) lifted to the whole episode,
    so home runs, away runs, margin and every probability stay consistent with the
    tilted total. Returns (weights summing to 1, theta)."""
    from scipy.optimize import brentq

    t = np.asarray(total, float)
    c = t - t.mean()

    def mean_at(th):
        e = np.exp(th * c - np.max(th * c))
        return (e * t).sum() / e.sum()

    th = 0.0
    if np.isfinite(target) and t.std() > 0:
        try:
            th = brentq(lambda z: mean_at(z) - target, -1.5, 1.5)
        except ValueError:
            th = 0.0
    e = np.exp(th * c - np.max(th * c))
    return e / e.sum(), float(th)


def flat(prefix: str, d: dict) -> dict:
    """{prefix}_{field} columns of a Dist (n is reported once per row as n_sims)."""
    return {f"{prefix}_{k}": d[k] for k in DIST_FIELDS}


def game_distribution_row(home: np.ndarray, away: np.ndarray, innings: np.ndarray, sched: np.ndarray,
                          bias_runs: float, apply_bias: bool = True) -> dict:
    """game_sim_distributions fields for ONE game from its episodes.

    With apply_bias, every field is computed under tilt weights that move the mean total
    from the raw sim mean to (raw mean - bias_runs), the research's totals correction."""
    h = np.asarray(home, np.int64); a = np.asarray(away, np.int64)
    tot = h + a; mg = h - a
    raw_mean = float(tot.mean())
    if apply_bias:
        w, _ = tilt_weights(tot, raw_mean - float(bias_runs))
    else:
        w = np.full(len(tot), 1.0 / len(tot))
    n = len(tot)
    row = {}
    row.update(flat("home_runs", dist_from_samples(h, w, n)))
    row.update(flat("away_runs", dist_from_samples(a, w, n)))
    row.update(flat("total", dist_from_samples(tot, w, n)))
    row.update(flat("margin", dist_from_samples(mg, w, n)))
    row["p_home_win"] = float((w * (mg > 0)).sum())
    row["p_extra_innings"] = float((w * (np.asarray(innings) > np.asarray(sched))).sum())
    row["p_home_cover_rl"] = float((w * (mg >= 2)).sum())
    row["p_over_by_line"] = json.dumps({f"{L:.1f}": round(float((w * (tot > L)).sum()), 5)
                                        for L in TOTAL_LINES})
    row["totals_calibrated"] = bool(apply_bias)
    row["total_bias_shift"] = -float(bias_runs) if apply_bias else 0.0
    return row
