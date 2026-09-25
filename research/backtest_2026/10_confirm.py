"""Final confirmation on the full season + a learned selective-prediction gate.

Two things:
  (a) Re-score the surviving rule families on ALL 1,763 pregame games. V10-only rules
      need no training, so they get the full season and therefore real statistical power.
  (b) Try a genuinely different approach: train a gate to predict whether V10 will be
      CORRECT, fit on discovery only, and take its top-scoring games on validation.
      This is selective prediction / learning to abstain.
"""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
exec(open("research/backtest_2026/06_headtohead.py").read().split("out, diag = run()")[0])

def wilson(k, nn):
    z=1.96; ph=k/nn; den=1+z*z/nn
    c=(ph+z*z/(2*nn))/den; h=z*np.sqrt(ph*(1-ph)/nn+z*z/(4*nn*nn))/den
    return (c-h)*100,(c+h)*100

# ---------- (a) full-season power for V10-only rules ----------
FULL = d.reset_index(drop=True); yf = y; pf = v10
conf = np.maximum(pf, 1-pf); corr = (pf>=.5).astype(int)==yf
base = corr.mean()*100
print(f"FULL SEASON: n={len(FULL)}  V10 acc={base:.2f}%  home rate={yf.mean()*100:.2f}%")
print(f"(a 60% claim needs n>=310 to beat the 53% base rate; n>=200 minimum)\n")
print("="*98)
print("V10 CONFIDENCE TIERS ON THE FULL SEASON -- with Wilson CIs and required power")
print("="*98)
print(f"{'rule':34}{'n':>6}{'acc':>8}{'95% CI':>18}{'lift':>8}   verdict")
rows=[]
spq = FULL.sp_quality_composite_diff.abs().values
spq = np.nan_to_num(spq, nan=np.nanmedian(spq))
cands = {}
for t in [.55,.58,.60,.62,.64,.66,.68]:
    cands[f"v10 conf>={t}"] = conf>=t
cands["v10 conf>=0.58 & home-fav"] = (conf>=.58)&(pf>=.5)
cands["v10 conf>=0.60 & home-fav"] = (conf>=.60)&(pf>=.5)
for q in [.5,.6,.7]:
    cands[f"|sp_quality_diff|>=q{int(q*100)}"] = spq>=np.quantile(spq,q)
cands["|sp_q|>=q60 & conf>=0.58"] = (spq>=np.quantile(spq,.6))&(conf>=.58)
for name,m in cands.items():
    if m.sum()<40: continue
    k=corr[m].sum(); nn=int(m.sum()); a=k/nn*100; lo,hi=wilson(k,nn)
    v = "60%+ AND powered" if (a>=60 and nn>=310 and lo>53) else \
        ("60%+ but UNDERPOWERED" if a>=60 else ("beats base rate" if lo>base else "not distinguishable"))
    print(f"{name:34}{nn:>6}{a:>8.2f}{f'[{lo:.1f},{hi:.1f}]':>18}{a-base:>+8.2f}   {v}")

# ---------- (b) learned selective-prediction gate ----------
out, diag = run(min_train=400, cadence_days=7, C=0.1)
mask = ~np.isnan(out["logit3"]); D=d.loc[mask].reset_index(drop=True); yv=y[mask]
p10=v10[mask]; plg=out["logit3"][mask]
n=len(D); cut=int(n*.6); disc=np.zeros(n,bool); disc[:cut]=True; val=~disc
c10=(p10>=.5).astype(int)==yv
GF=["elo_differential","pythag_differential","sp_quality_composite_diff","home_park_factor",
    "home_days_rest","away_days_rest","h2h_win_pct_3yr","is_divisional","series_game_number",
    "home_sp_k_pct","away_sp_k_pct","matchup_advantage_home"]
X=D[GF].apply(pd.to_numeric,errors="coerce")
X["v10_conf"]=np.maximum(p10,1-p10); X["v10_side"]=(p10>=.5).astype(int)
X["agree"]=((p10>=.5)==(plg>=.5)).astype(int)
X["disagree_mag"]=np.abs(p10-plg)
X=X.fillna(X[disc].median()).fillna(0)
print("\n" + "="*98)
print("(b) LEARNED GATE -- predict whether V10 will be CORRECT; fit on discovery, test on validation")
print("="*98)
print(f"    gate trained on {disc.sum()} games, evaluated on {val.sum()} held-out games")
print(f"    V10 baseline on validation = {c10[val].mean()*100:.2f}%\n")
for label, mdl in [("logistic gate", LogisticRegression(C=0.1,max_iter=2000)),
                   ("gbm gate", GradientBoostingClassifier(n_estimators=120,max_depth=2,
                                                           learning_rate=0.05,random_state=0))]:
    sc=StandardScaler().fit(X[disc]); mdl.fit(sc.transform(X[disc]), c10[disc])
    s=mdl.predict_proba(sc.transform(X))[:,1]
    print(f"  {label}:  gate AUC on validation = "
          f"{__import__('sklearn.metrics',fromlist=['roc_auc_score']).roc_auc_score(c10[val],s[val]):.4f}")
    for frac in [.10,.20,.30,.50]:
        thr=np.quantile(s[disc],1-frac); m=val&(s>=thr)
        if m.sum()<20: continue
        k=c10[m].sum(); nn=int(m.sum()); a=k/nn*100; lo,hi=wilson(k,nn)
        print(f"    top {int(frac*100):>2}% by gate score: n={nn:>4} acc={a:>6.2f}%  "
              f"CI[{lo:.1f},{hi:.1f}]  lift={a-c10[val].mean()*100:+.2f}pp")
