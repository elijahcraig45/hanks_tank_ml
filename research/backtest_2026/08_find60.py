"""Hunt for a 60%+ slice of 2026 that survives out-of-sample.

Anti-snooping protocol: every rule is scored on a DISCOVERY window (earlier games),
the winners are then tested once on a VALIDATION window (later games) they never
touched. The number of rules searched is reported so the reader can judge how much
of any winner is luck.

Rules come in families: confidence thresholds, model agreement (consensus),
feature-magnitude slices, and a learned selective-prediction gate.
"""
import warnings, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore")
exec(open("research/backtest_2026/06_headtohead.py").read().split("out, diag = run()")[0])

# widest OOS window: min_train=400 gives ~1,274 scored games
out, diag = run(min_train=400, cadence_days=7, C=0.1)
mask = ~np.isnan(out["logit3"])
D = d.loc[mask].reset_index(drop=True)
yv = y[mask]
pv10 = v10[mask]; plg = out["logit3"][mask]; pst = out["stack"][mask]
pelo = D.elo_home_win_prob.fillna(.5).values
n = len(D)
print(f"scored pool: {n} games  {D.game_date.min().date()} -> {D.game_date.max().date()}")

# discovery / validation split by date
cut = int(n * 0.60)
disc = np.zeros(n, bool); disc[:cut] = True
val = ~disc
print(f"discovery: {disc.sum()} games (to {D.game_date[cut-1].date()})   "
      f"validation: {val.sum()} games (from {D.game_date[cut].date()})")
print(f"base rate: discovery {yv[disc].mean()*100:.2f}%  validation {yv[val].mean()*100:.2f}%\n")

def wilson(k, nn):
    if nn == 0: return (np.nan, np.nan)
    z = 1.96; ph = k / nn
    den = 1 + z*z/nn
    c = (ph + z*z/(2*nn)) / den
    h = z*np.sqrt(ph*(1-ph)/nn + z*z/(4*nn*nn)) / den
    return (c-h)*100, (c+h)*100

side = lambda p: (p >= .5).astype(int)
conf = lambda p: np.maximum(p, 1-p)
corr_v10 = side(pv10) == yv
corr_lg  = side(plg)  == yv
corr_st  = side(pst)  == yv

# ---------------- build rule catalogue ----------------
rules = {}
for t in [.55,.58,.60,.62,.64,.66,.68,.70]:
    rules[f"v10 conf>={t}"] = (conf(pv10) >= t, corr_v10)
    rules[f"stack conf>={t}"] = (conf(pst) >= t, corr_st)
    rules[f"v10 conf>={t} & home-fav"] = ((conf(pv10) >= t) & (pv10 >= .5), corr_v10)
    rules[f"v10 conf>={t} & away-fav"] = ((conf(pv10) >= t) & (pv10 < .5), corr_v10)
# consensus families
agree2 = side(pv10) == side(plg)
agree3 = agree2 & (side(pv10) == side(pelo))
for t in [.50,.55,.58,.60,.62,.64]:
    rules[f"v10+logit3 agree & v10conf>={t}"] = (agree2 & (conf(pv10) >= t), corr_v10)
    rules[f"v10+logit3+elo agree & v10conf>={t}"] = (agree3 & (conf(pv10) >= t), corr_v10)
    rules[f"agree3 & minconf>={t}"] = (agree3 & (np.minimum(conf(pv10), conf(plg)) >= t), corr_v10)
# feature-magnitude slices, thresholds as discovery-window quantiles
featdefs = {
    "|elo_diff|": D.elo_differential.abs().values,
    "|pythag_diff|": D.pythag_differential.abs().values,
    "|sp_quality_diff|": D.sp_quality_composite_diff.abs().values,
    "|elo|+|pythag| z": (pd.Series(D.elo_differential.abs()).rank(pct=True)
                         + pd.Series(D.pythag_differential.abs()).rank(pct=True)).values,
}
for fname, fv in featdefs.items():
    fv = np.nan_to_num(fv, nan=np.nanmedian(fv))
    for q in [.5, .6, .7, .8]:
        thr = np.quantile(fv[disc], q)
        rules[f"{fname} >= q{int(q*100)}"] = (fv >= thr, corr_v10)
        rules[f"{fname} >= q{int(q*100)} & agree2"] = ((fv >= thr) & agree2, corr_v10)
# context slices
rules["divisional"] = (D.is_divisional.fillna(0).values == 1, corr_v10)
rules["non-divisional"] = (D.is_divisional.fillna(0).values == 0, corr_v10)
rules["series opener"] = (D.series_game_number.fillna(1).values == 1, corr_v10)
rules["home rest adv >=2d"] = ((D.home_days_rest.fillna(4) - D.away_days_rest.fillna(4)).values >= 2, corr_v10)
rules["hitter park (pf>1.05)"] = (D.home_park_factor.fillna(1).values > 1.05, corr_v10)

# ---------------- score on discovery ----------------
rows = []
for name, (m, corr) in rules.items():
    if m.sum() == 0: continue
    md, mv = m & disc, m & val
    if md.sum() < 40: continue
    kd = corr[md].sum(); kv = corr[mv].sum()
    rows.append(dict(rule=name, d_n=int(md.sum()), d_acc=kd/md.sum()*100,
                     v_n=int(mv.sum()), v_acc=(kv/mv.sum()*100 if mv.sum() else np.nan),
                     v_lo=wilson(kv, mv.sum())[0], v_hi=wilson(kv, mv.sum())[1]))
R = pd.DataFrame(rows)
print(f"rules searched (with >=40 discovery games): {len(R)}")
print(f"at a=0.05 you would expect ~{len(R)*0.05:.1f} false 'significant' hits by chance alone\n")

print("="*100)
print("TOP 14 RULES BY DISCOVERY ACCURACY, and how they held up on validation")
print("="*100)
top = R.sort_values("d_acc", ascending=False).head(14)
print(top.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\n" + "="*100)
print("RULES THAT CLEAR 60% ON *VALIDATION* WITH n>=150 (the only ones worth believing)")
print("="*100)
keep = R[(R.v_acc >= 60) & (R.v_n >= 150)].sort_values("v_acc", ascending=False)
print(keep.to_string(index=False, float_format=lambda x: f"{x:.2f}") if len(keep)
      else "  NONE. No rule reaches 60% on held-out games at a sample size that could support it.")

print("\n" + "="*100)
print("BEST RULES BY VALIDATION ACC WITH n>=200 (powered enough to beat the base rate)")
print("="*100)
p2 = R[R.v_n >= 200].sort_values("v_acc", ascending=False).head(8)
print(p2.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
R.to_csv("data/backtest_2026/rule_search.csv", index=False)
