"""Production entry point: assemble a game and predict it by Monte Carlo.

SimEngine.fit() trains player rates on every plate appearance strictly BEFORE a given
date, so a prediction can never see its own game. predict_game() then plays the game
n_episodes times PA by PA and reports the outcome frequency.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from . import CLASSES
from .rates import RateTable, log5
from .usage import hook_curve
from .game import apply_scale
from .sim import simulate_game, to_cum

N_INNINGS = 9


class SimEngine:
    def __init__(self, n_episodes: int = 1000, alpha: float = 1.14, seed: int = 0):
        self.n_episodes = n_episodes
        self.alpha = alpha
        self.seed = seed
        self.fitted_through = None

    def fit(self, pa: pd.DataFrame, as_of: pd.Timestamp | None = None):
        """Train on PAs strictly before `as_of`. Leak-free by construction."""
        d = pa if as_of is None else pa[pa.game_date < as_of]
        if len(d) < 50_000:
            raise ValueError(f"only {len(d)} PAs before {as_of}; too few to fit")
        self.bat = RateTable(d, "batter")
        self.pit = RateTable(d, "pitcher")
        self.league = self.bat.league
        self.curve = hook_curve(d, N_INNINGS)
        rel = d[~d.pitcher_is_starter]
        pt = np.where(rel.inning_topbot.values == "Top",
                      rel.home_team.values, rel.away_team.values)
        rel = rel.assign(pitch_team=pt)
        self.bullpen = {}
        for tm, g in rel.groupby("pitch_team", observed=True):
            vc = g.pitcher.value_counts().head(12)
            rows = np.array([self.pit.get(int(p)) for p in vc.index])
            w = vc.values.astype(float)
            self.bullpen[tm] = np.average(rows, axis=0, weights=w / w.sum())
        lr = np.array([(rel.ev == c).mean() for c in CLASSES]); self.bullpen_league = lr / lr.sum()
        self.fitted_through = as_of
        return self

    def _bp(self, team):
        return self.bullpen.get(team, self.bullpen_league)

    def _side_innings(self, order, opp_starter, opp_team, hand_of_opp, alpha):
        lineup = np.array([self.bat.get(int(b), hand_of_opp) for b in order])
        sp = self.pit.get(int(opp_starter))
        bp = self._bp(opp_team)
        out = []
        for p_start in self.curve:
            opp = p_start * sp + (1 - p_start) * bp
            opp = opp / opp.sum()
            out.append(apply_scale(log5(lineup, opp[None, :], self.league[None, :]), alpha))
        return out

    def predict_game(self, home_order, away_order, home_starter, away_starter,
                     home_team, away_team, park_factor: float = 1.0,
                     home_starter_hand=None, away_starter_hand=None,
                     n_episodes: int | None = None, seed: int | None = None) -> dict:
        # park factor scales the run environment for BOTH sides
        a = self.alpha * float(np.clip(park_factor, 0.7, 1.4)) ** 0.7
        home_inn = self._side_innings(home_order, away_starter, away_team,
                                      away_starter_hand, a)
        away_inn = self._side_innings(away_order, home_starter, home_team,
                                      home_starter_hand, a)
        r = simulate_game(to_cum(home_inn), to_cum(away_inn),
                          n_episodes=n_episodes or self.n_episodes,
                          seed=self.seed if seed is None else seed)
        r["park_alpha"] = a
        return r
