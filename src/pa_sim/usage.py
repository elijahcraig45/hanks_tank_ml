"""Starter hook curve and bullpen composite.

Relief is ~35% of innings, so a fixed "starter goes 6" assumption is the largest
avoidable error in the simulator. Instead each inning blends the starter's and the
bullpen's outcome probabilities by P(starter still pitching in that inning), estimated
empirically from how far starters actually go.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from .rates import RateTable, log5
from . import CLASSES


def hook_curve(pa: pd.DataFrame, n_innings: int = 9) -> np.ndarray:
    """P(the game's starter is still pitching) per inning, league-average.

    Estimated as: of PAs in inning i, what fraction were thrown by that game's starter.
    """
    sub = pa[pa.inning <= n_innings]
    g = sub.groupby("inning")["pitcher_is_starter"].mean()
    curve = np.array([float(g.get(i, 0.0)) for i in range(1, n_innings + 1)])
    return np.clip(curve, 0.0, 1.0)


def bullpen_composite(pa: pd.DataFrame, pit: RateTable, team_pitchers: dict,
                      team: int, hand: str | None = None) -> np.ndarray:
    """Usage-weighted average of a team's relievers' rates.

    Falls back to the league reliever baseline when a team's relievers are unknown.
    """
    ids = team_pitchers.get(team, [])
    if not ids:
        return pit.league
    rows, wts = [], []
    for pid, w in ids:
        rows.append(pit.get(pid, hand))
        wts.append(max(w, 1.0))
    W = np.array(wts, dtype=float)
    return np.average(np.array(rows), axis=0, weights=W / W.sum())


def inning_probs(lineup_rates: np.ndarray, starter: np.ndarray, bullpen: np.ndarray,
                 league: np.ndarray, curve: np.ndarray) -> list[np.ndarray]:
    """Per-inning (9, 8) matrices, blending starter and bullpen by the hook curve.

    lineup_rates: (9, 8) the nine batters' own rates (already hand-split)
    """
    out = []
    for i, p_start in enumerate(curve):
        opp = p_start * starter + (1.0 - p_start) * bullpen
        opp = opp / opp.sum()
        out.append(log5(lineup_rates, opp[None, :], league[None, :]))
    return out
