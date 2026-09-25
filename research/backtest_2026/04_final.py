"""Round 3: fit the blend weight and shrinkage INSIDE each training fold.

Nothing here is tuned on test games. For each walk-forward step the blend weight
w and shrink factor lam are chosen by minimising log-loss on the training window,
then applied unchanged to the next block.
"""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
warnings.filterwarnings("ignore")

MIN_TRAIN, STEP = 600, 50
d = pd.read_parquet("data/backtest_2026/games_2026_pregame.parquet")
d = d.sort_values(["game_date","game_pk"]).reset_index(drop=True)
y = d.home_win.values
SMALL = ["elo_differential","pythag_differential","sp_quality_composite_diff"]
for c in SMALL: d[c] = pd.to_numeric(d[c], errors="coerce")
v10 = np.clip(d.home_win_probability.values.astype(float), 1e-6, 1-1e-6)

def ll(p, t):
    p = np.clip(p, 1e-6, 1-1e-6)
    return -(t*np.log(p) + (1-t)*np.log(1-p)).mean()

n = len(d)
oof = {k: np.full(n, np.nan) for k in ["logit3","blend_tuned","v10_shrunk_tuned","final_stack"]}
chosen_w, chosen_lam = [], []

for s in range(MIN_TRAIN, n, STEP):
    tr, te = np.arange(0, s), np.arange(s, min(s+STEP, n))
    Xtr, Xte = d.loc[tr, SMALL].astype(float), d.loc[te, SMALL].astype(float)
    med = Xtr.median(); Xtr, Xte = Xtr.fillna(med).fillna(0), Xte.fillna(med).fillna(0)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xtr), y[tr])
    p_tr = m.predict_proba(sc.transform(Xtr))[:,1]
    p_te = m.predict_proba(sc.transform(Xte))[:,1]
    oof["logit3"][te] = p_te
    base = y[tr].mean()

    # shrink V10 toward the train-window base rate
    lams = np.linspace(0.1, 1.0, 19)
    lam = lams[np.argmin([ll(base + l*(v10[tr]-base), y[tr]) for l in lams])]
    oof["v10_shrunk_tuned"][te] = base + lam*(v10[te]-base)

    # blend weight on the shrunk V10 + logit3
    v10s_tr = base + lam*(v10[tr]-base); v10s_te = base + lam*(v10[te]-base)
    ws = np.linspace(0, 1, 21)
    w = ws[np.argmin([ll(wi*p_tr + (1-wi)*v10s_tr, y[tr]) for wi in ws])]
    oof["blend_tuned"][te] = w*p_te + (1-w)*v10s_te

    # logistic stack on the two probability logits
    def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
    S_tr = np.column_stack([lg(p_tr), lg(v10[tr])]); S_te = np.column_stack([lg(p_te), lg(v10[te])])
    st = LogisticRegression(C=1.0, max_iter=2000).fit(S_tr, y[tr])
    oof["final_stack"][te] = st.predict_proba(S_te)[:,1]
    chosen_w.append(w); chosen_lam.append(lam)

mask = ~np.isnan(oof["logit3"]); yy = y[mask]
print(f"test games {mask.sum()}  home rate {yy.mean():.4f}")
print(f"tuned blend weight on logit3: mean {np.mean(chosen_w):.2f} (range {min(chosen_w):.2f}-{max(chosen_w):.2f})")
print(f"tuned V10 shrink lambda     : mean {np.mean(chosen_lam):.2f} (range {min(chosen_lam):.2f}-{max(chosen_lam):.2f})\n")

cands = dict(oof); cands["prod_v10 (incumbent)"] = v10
cands["always_home"] = np.full(n, y[:MIN_TRAIN].mean())
rows=[]
for k,p in cands.items():
    pp=np.clip(p[mask].astype(float),1e-6,1-1e-6); corr=((pp>=.5).astype(int)==yy)
    rows.append(dict(model=k, acc=corr.mean()*100, auc=roc_auc_score(yy,pp),
                     brier=brier_score_loss(yy,pp), logloss=log_loss(yy,pp), avg_p=pp.mean()))
r=pd.DataFrame(rows).sort_values("logloss")
print(r.to_string(index=False,float_format=lambda x:f"{x:.4f}"))

# significance vs incumbent
print("\npaired bootstrap vs prod_v10 (4000 resamples):")
pb=np.clip(v10[mask],1e-6,1-1e-6); lb=-(yy*np.log(pb)+(1-yy)*np.log(1-pb))
for k in ["blend_tuned","v10_shrunk_tuned","final_stack","logit3"]:
    pa=np.clip(oof[k][mask],1e-6,1-1e-6); la=-(yy*np.log(pa)+(1-yy)*np.log(1-pa))
    dd=lb-la; bs=np.array([dd[np.random.randint(0,len(dd),len(dd))].mean() for _ in range(4000)])
    acc_a=((pa>=.5).astype(int)==yy).mean()*100; acc_b=((pb>=.5).astype(int)==yy).mean()*100
    print(f"  {k:18s} logloss {dd.mean():+.5f} CI[{np.percentile(bs,2.5):+.5f},{np.percentile(bs,97.5):+.5f}] "
          f"P={ (bs>0).mean():.3f}  acc {acc_a-acc_b:+.2f}pp")

# calibration of incumbent vs best
print("\ncalibration -- predicted vs actual (incumbent V10):")
for lo,hi in [(0,.45),(.45,.5),(.5,.55),(.55,.6),(.6,.65),(.65,1)]:
    m2=(v10[mask]>=lo)&(v10[mask]<hi)
    if m2.sum()>=15: print(f"  pred {lo:.2f}-{hi:.2f}: n={m2.sum():4d} mean_pred={v10[mask][m2].mean():.3f} actual={yy[m2].mean():.3f}")

print("\nconfidence curve -- blend_tuned vs prod_v10:")
print(f"{'thr':>5} {'blend_n':>8}{'blend_acc':>11} {'v10_n':>7}{'v10_acc':>9}")
for t in [.50,.52,.55,.58,.60,.62,.65]:
    pa=oof["blend_tuned"][mask]; ca=np.maximum(pa,1-pa)>=t
    cb=np.maximum(pb,1-pb)>=t
    aa=(((pa>=.5).astype(int)==yy)[ca].mean()*100) if ca.sum()>=20 else np.nan
    ab=(((pb>=.5).astype(int)==yy)[cb].mean()*100) if cb.sum()>=20 else np.nan
    print(f"{t:5.2f} {ca.sum():8d}{aa:11.2f} {cb.sum():7d}{ab:9.2f}")
