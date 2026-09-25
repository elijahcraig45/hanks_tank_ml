"""Series prediction using ONLY first-game pregame features -- no leakage.

11_series.py averaged elo_differential across the series, but Elo is updated after
every game, so games 2-3 carry game-1's result. That is look-ahead. A real forecast
is made before the series opens, so only the FIRST game's pregame row may be used.
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
d["seq"] = d.groupby("series_id").cumcount()

first = d[d.seq == 0].set_index("series_id")            # pregame row of game 1 ONLY
agg = d.groupby("series_id").agg(n_games=("game_pk","size"), home_wins=("home_win","sum"))
g = agg.join(first[["home_win_probability","elo_differential","pythag_differential",
                    "elo_home_win_prob","sp_quality_composite_diff"]])
g = g[g.n_games >= 3].copy()
undec = int((g.home_wins == g.n_games/2).sum())
g = g[g.home_wins != g.n_games/2]
g["home_took_series"] = (g.home_wins > g.n_games/2).astype(int)
yy = g.home_took_series.values

print(f"series >=3 games: {len(g)} decided  ({undec} even splits dropped as undecided)")
print(f"home takes series {yy.mean()*100:.2f}%   (single-game home rate {d.home_win.mean()*100:.2f}%)")
print(f"features are game-1 pregame values only -- no post-game-1 information\n")
print("="*94)
print("SERIES WINNER, FIRST-GAME FEATURES ONLY   (n>=310 needed for a credible 60% claim)")
print("="*94)
print(f"{'predictor':40}{'n':>6}{'acc':>8}{'95% CI':>18}{'vs base':>9}")
base = yy.mean()*100
def rep(name, pred, m=None):
    m = np.ones(len(g),bool) if m is None else m
    corr=(pred[m]==yy[m]); k=int(corr.sum()); nn=int(m.sum())
    if nn<30: return
    lo,hi=wilson(k,nn)
    flag = "  <-- 60%+ POWERED" if (k/nn>=.60 and nn>=310 and lo>base) else ""
    print(f"{name:40}{nn:>6}{k/nn*100:>8.2f}{f'[{lo:.1f},{hi:.1f}]':>18}{k/nn*100-base:>+9.2f}{flag}")

elo = g.elo_differential.values; pyt = g.pythag_differential.values
p10 = g.home_win_probability.values
rep("always home", np.ones(len(g),int))
rep("Elo differential sign", (elo>=0).astype(int))
rep("pythag differential sign", (pyt>=0).astype(int))
rep("V10 game-1 prob sign", (p10>=.5).astype(int))
print()
for q in [.3,.4,.5,.6,.75]:
    thr=np.quantile(np.abs(elo),q)
    rep(f"Elo sign, |elo|>=q{int(q*100)}", (elo>=0).astype(int), np.abs(elo)>=thr)
print()
ag = ((p10>=.5)==(elo>=0))
rep("Elo+V10 agree", (elo>=0).astype(int), ag)
for q in [.3,.5]:
    thr=np.quantile(np.abs(elo),q)
    rep(f"Elo+V10 agree & |elo|>=q{int(q*100)}", (elo>=0).astype(int), ag&(np.abs(elo)>=thr))
print()
ag3 = ag & ((pyt>=0)==(elo>=0))
rep("Elo+V10+pythag all agree", (elo>=0).astype(int), ag3)

# split-half stability check
h = len(g)//2
print("\nSTABILITY -- same rules on first vs second half of the season:")
order = d[d.seq==0].sort_values("game_date").series_id
order = [s for s in order if s in g.index]
gi = g.loc[order]
for name, pred_fn in [("Elo sign (all)", lambda x: (x.elo_differential.values>=0).astype(int)),
                      ("Elo sign |elo|>=q50", None)]:
    if name.startswith("Elo sign (all)"):
        for lab, sl in [("H1", gi.iloc[:h]), ("H2", gi.iloc[h:])]:
            pr=(sl.elo_differential.values>=0).astype(int); ay=sl.home_took_series.values
            print(f"  {name:24} {lab}: n={len(sl):3d} acc={(pr==ay).mean()*100:.2f}%")
    else:
        thr=np.quantile(np.abs(gi.elo_differential.values),.5)
        for lab, sl in [("H1", gi.iloc[:h]), ("H2", gi.iloc[h:])]:
            m=np.abs(sl.elo_differential.values)>=thr
            pr=(sl.elo_differential.values>=0).astype(int); ay=sl.home_took_series.values
            print(f"  {name:24} {lab}: n={int(m.sum()):3d} acc={(pr[m]==ay[m]).mean()*100:.2f}%")
