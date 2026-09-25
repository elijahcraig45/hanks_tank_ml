"""Different target: predict the SERIES winner instead of the single game.

A single MLB game is close to a coin flip, but a 3-game series averages out some
variance, so the same per-game edge should translate into a larger edge on the
series. Series are reconstructed as consecutive-date blocks of the same
(home_team, away_team) pair.
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

def wilson(k,nn):
    z=1.96; ph=k/nn; den=1+z*z/nn
    c=(ph+z*z/(2*nn))/den; h=z*np.sqrt(ph*(1-ph)/nn+z*z/(4*nn*nn))/den
    return (c-h)*100,(c+h)*100

d = pd.read_parquet("data/backtest_2026/games_2026_pregame.parquet")
d = d.sort_values(["home_team_id","away_team_id","game_date"]).reset_index(drop=True)
d["gap"] = d.groupby(["home_team_id","away_team_id"]).game_date.diff().dt.days
d["new"] = (d.gap.isna() | (d.gap > 3)).astype(int)
d["series_id"] = (d.groupby(["home_team_id","away_team_id"]).new.cumsum().astype(str)
                  + "_" + d.home_team_id.astype(str) + "_" + d.away_team_id.astype(str))

g = d.groupby("series_id").agg(
    n_games=("game_pk","size"), home_wins=("home_win","sum"),
    mean_p=("home_win_probability","mean"),
    mean_elo=("elo_differential","mean"),
    mean_pythag=("pythag_differential","mean"),
    start=("game_date","min")).reset_index()
g = g[g.n_games >= 3].copy()                      # need a real series
g = g[g.home_wins != g.n_games/2]                 # drop exact ties (even-length splits)
g["home_took_series"] = (g.home_wins > g.n_games/2).astype(int)

print(f"series with >=3 games and a decided winner: {len(g)}")
print(f"  covering {int(g.n_games.sum())} games, {g.n_games.mean():.2f} games/series")
print(f"  home team takes the series {g.home_took_series.mean()*100:.2f}% of the time")
print(f"  (single-game home win rate for comparison: {d.home_win.mean()*100:.2f}%)\n")

print("="*92)
print("SERIES-WINNER ACCURACY  (n>=310 needed for a credible 60% claim)")
print("="*92)
yy = g.home_took_series.values
print(f"{'predictor':38}{'n':>6}{'acc':>8}{'95% CI':>18}")
def rep(name, pred, m=None):
    m = np.ones(len(g),bool) if m is None else m
    corr = (pred[m] == yy[m]); k=corr.sum(); nn=int(m.sum())
    if nn < 30: return
    lo,hi = wilson(k,nn)
    print(f"{name:38}{nn:>6}{k/nn*100:>8.2f}{f'[{lo:.1f},{hi:.1f}]':>18}")

rep("always home takes series", np.ones(len(g),int))
rep("V10 mean prob over series", (g.mean_p.values>=.5).astype(int))
rep("Elo differential sign", (g.mean_elo.values>=0).astype(int))
rep("pythag differential sign", (g.mean_pythag.values>=0).astype(int))
print()
# selectivity on series
conf = np.abs(g.mean_p.values-.5)+.5
for t in [.53,.55,.57,.60]:
    rep(f"V10 mean prob, conf>={t}", (g.mean_p.values>=.5).astype(int), conf>=t)
print()
for q in [.4,.6,.75]:
    thr=np.quantile(np.abs(g.mean_elo.values),q)
    rep(f"Elo sign, |elo|>=q{int(q*100)}", (g.mean_elo.values>=0).astype(int),
        np.abs(g.mean_elo.values)>=thr)
print()
agree = ((g.mean_p.values>=.5)==(g.mean_elo.values>=0))
rep("V10+Elo agree on series", (g.mean_p.values>=.5).astype(int), agree)
rep("V10+Elo agree & conf>=0.55", (g.mean_p.values>=.5).astype(int), agree&(conf>=.55))
