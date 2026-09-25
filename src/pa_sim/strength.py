"""Team-strength logistic: the "strength" benchmark of research/backtest_2026/43_benchmarks.py.

    elo_d   Elo difference + 24 home points. K=4 scaled by log1p(|run margin|); each team
            regresses 1/3 toward 1500 at its first game of a season.
    pyth_d  season-to-date pythagorean win% difference, exponent 1.83, with a prior of
            10 games at 4.5 runs scored and allowed (so April is not noise).
    p       LogisticRegression(C=1) on [elo_d, pyth_d], fit on the 3 seasons before the
            target season.

Pure functions over a games DataFrame (columns: game_pk, game_date, year, home, away,
h_runs, a_runs — runs NaN for unplayed games). Features for a game use only games
before it in (game_date, game_pk) order, and unplayed games never update state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K, HFA, REGRESS = 4.0, 24.0, 2.0 / 3.0
PY_EXP, PY_GAMES, PY_RPG = 1.83, 10, 4.5


def features(games: pd.DataFrame) -> pd.DataFrame:
    G = games.sort_values(["game_date", "game_pk"]).reset_index(drop=True).copy()
    elo, last_year, rs, ra = {}, {}, {}, {}
    e_d = np.zeros(len(G)); py = np.zeros(len(G))
    prior = PY_GAMES * PY_RPG

    def pyth(t, y):
        a, b = rs[(t, y)] + prior, ra[(t, y)] + prior
        return a ** PY_EXP / (a ** PY_EXP + b ** PY_EXP)

    for i, r in enumerate(G.itertuples()):
        y = int(r.year)
        for t in (r.home, r.away):
            if last_year.get(t) != y:
                elo[t] = 1500 + (elo.get(t, 1500) - 1500) * REGRESS
                last_year[t] = y
                rs[(t, y)] = 0.0; ra[(t, y)] = 0.0
        e_d[i] = elo[r.home] - elo[r.away] + HFA
        py[i] = pyth(r.home, y) - pyth(r.away, y)
        if pd.isna(r.h_runs) or pd.isna(r.a_runs):
            continue
        hw = float(r.h_runs > r.a_runs)
        pe = 1 / (1 + 10 ** (-e_d[i] / 400))
        mov = np.log1p(abs(r.h_runs - r.a_runs))
        elo[r.home] += K * mov * (hw - pe); elo[r.away] -= K * mov * (hw - pe)
        rs[(r.home, y)] += r.h_runs; ra[(r.home, y)] += r.a_runs
        rs[(r.away, y)] += r.a_runs; ra[(r.away, y)] += r.h_runs
    G["elo_d"] = e_d; G["pyth_d"] = py
    G["y"] = np.where(G.h_runs.notna() & G.a_runs.notna(), (G.h_runs > G.a_runs).astype(float), np.nan)
    return G


def fit(G: pd.DataFrame, target_year: int, n_seasons: int = 3):
    from sklearn.linear_model import LogisticRegression

    tr = (G.year < target_year) & (G.year >= target_year - n_seasons) & G.y.notna()
    if tr.sum() < 1000:
        return None
    return LogisticRegression(C=1.0).fit(G.loc[tr, ["elo_d", "pyth_d"]].values, G.y[tr].astype(int))


def predict(model, G: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(G[["elo_d", "pyth_d"]].values)[:, 1]


GAMES_SQL = """
SELECT game_pk, game_date, EXTRACT(YEAR FROM game_date) AS year,
       CAST(home_team_id AS STRING) AS home, CAST(away_team_id AS STRING) AS away,
       home_score AS h_runs, away_score AS a_runs
FROM `{proj}.{hist}.games_historical`
WHERE game_type = 'R' AND home_score IS NOT NULL AND away_score IS NOT NULL
  AND game_date < @cutoff AND EXTRACT(YEAR FROM game_date) >= @min_year
UNION ALL
SELECT game_pk, ANY_VALUE(game_date), EXTRACT(YEAR FROM ANY_VALUE(game_date)),
       CAST(ANY_VALUE(home_team_id) AS STRING), CAST(ANY_VALUE(away_team_id) AS STRING),
       -- a game is final only once; the collector inserts a scoreless skeleton first
       MAX(IF(status LIKE '%Final%', home_score, NULL)),
       MAX(IF(status LIKE '%Final%', away_score, NULL))
FROM `{proj}.{ds}.games`
WHERE game_type = 'R' AND game_date <= @cutoff
GROUP BY game_pk
"""
