"""Empirical-Bayes player rates and log5 matchup combination.

Two-level shrinkage: a player's rate versus a given pitcher hand shrinks toward that
player's overall rate, which in turn shrinks toward the league rate. The shrinkage
constant per outcome class is estimated from the data by beta-binomial method of
moments rather than hand-set, so a 40-PA rookie is pulled hard toward league and a
600-PA regular is barely moved.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from . import CLASSES


def league_rates(pa: pd.DataFrame) -> np.ndarray:
    """League-wide outcome distribution, ordered as CLASSES."""
    v = pa["ev"].value_counts(normalize=True)
    return np.array([v.get(c, 0.0) for c in CLASSES], dtype=float)


def estimate_k(counts: np.ndarray, n: np.ndarray, mean: float) -> float:
    """Beta-binomial shrinkage constant by method of moments.

    counts/n are per-player successes/trials for one outcome class. Returns k = a+b,
    the prior "pseudo-PA" weight. Falls back to a wide prior when the moment estimate
    is degenerate (which happens for very rare classes such as 3B).
    """
    m = n >= 50
    if m.sum() < 30:
        return 300.0
    p = counts[m] / n[m]
    w = n[m] / n[m].sum()
    var_obs = float(np.sum(w * (p - mean) ** 2))
    var_binom = float(np.sum(w * mean * (1 - mean) / n[m]))
    prior_var = var_obs - var_binom
    if prior_var <= 1e-9:
        return 1000.0
    k = mean * (1 - mean) / prior_var - 1
    return float(np.clip(k, 20.0, 5000.0))


def _counts_by(pa: pd.DataFrame, key) -> tuple[pd.DataFrame, np.ndarray]:
    g = pa.groupby(key, observed=True)["ev"].value_counts().unstack(fill_value=0)
    for c in CLASSES:
        if c not in g.columns:
            g[c] = 0
    g = g[CLASSES]
    return g, g.sum(axis=1).values.astype(float)


class RateTable:
    """Shrunk per-player outcome rates, overall and split by opposing hand."""

    def __init__(self, pa: pd.DataFrame, role: str):
        """role: 'batter' (split by p_throws) or 'pitcher' (split by stand)."""
        assert role in ("batter", "pitcher")
        self.role = role
        self.split_col = "p_throws" if role == "batter" else "stand"
        self.league = league_rates(pa)

        overall, n_overall = _counts_by(pa, role)
        self.k = np.array([estimate_k(overall[c].values, n_overall, self.league[i])
                           for i, c in enumerate(CLASSES)])
        # level 1: player overall shrunk toward league
        num = overall.values + self.k * self.league
        den = (n_overall[:, None] + self.k)
        self.overall = pd.DataFrame(num / den, index=overall.index, columns=CLASSES)
        self.overall_n = pd.Series(n_overall, index=overall.index)

        # level 2: player-vs-hand shrunk toward that player's overall
        byhand, n_hand = _counts_by(pa, [role, self.split_col])
        idx = byhand.index
        prior = self.overall.reindex(idx.get_level_values(0)).values
        # split samples are ~half the size, so shrink them HARDER toward the
        # player's own overall rate, not softer (a k below the overall k lets
        # a 120-PA vs-LHP sample move the estimate more than the full sample can)
        k_hand = self.k * 3.0
        self.byhand = pd.DataFrame((byhand.values + k_hand * prior) / (n_hand[:, None] + k_hand),
                                   index=idx, columns=CLASSES)
        self._oa = {i: r for i, r in zip(self.overall.index, self.overall.values)}
        self._bh = {i: r for i, r in zip(self.byhand.index, self.byhand.values)}

    def get(self, player_id: int, hand: str | None = None) -> np.ndarray:
        if hand is not None:
            r = self._bh.get((player_id, hand))
            if r is not None:
                return r
        r = self._oa.get(player_id)
        return self.league if r is None else r


def log5(batter: np.ndarray, pitcher: np.ndarray, league: np.ndarray) -> np.ndarray:
    """Odds-ratio (log5) combination, applied per class then renormalised.

    raw_c = b_c * p_c / l_c  is the standard multiplicative extension of log5 to a
    multinomial. Renormalising keeps it a proper distribution.
    """
    l = np.where(league <= 0, 1e-9, league)
    raw = np.clip(batter, 1e-9, 1) * np.clip(pitcher, 1e-9, 1) / l
    s = raw.sum(axis=-1, keepdims=True)
    return raw / np.where(s <= 0, 1e-9, s)
