"""Does 10 years of training data beat half a season?

Trains the same 2-feature logistic (elo_differential, pythag_differential -- the
two features present in BOTH the historical V8 parquet and the 2026 stored
prediction rows) on 2015-2025 history, and scores it on the identical 2026
walk-forward test games used in 04_final.py.
"""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
warnings.filterwarnings("ignore")

F = ["elo_differential", "pythag_differential"]
MIN_TRAIN, STEP = 600, 50

hist = pd.concat([pd.read_parquet("data/training/train_v8_2015_2024.parquet"),
                  pd.read_parquet("data/training/val_v8_2025.parquet")], ignore_index=True)
hist = hist[F + ["home_won"]].apply(pd.to_numeric, errors="coerce").dropna()

d = pd.read_parquet("data/backtest_2026/games_2026_pregame.parquet")
d = d.sort_values(["game_date","game_pk"]).reset_index(drop=True)
for c in F: d[c] = pd.to_numeric(d[c], errors="coerce")
y = d.home_win.values

print("distribution check (must be comparable or the transfer is invalid):")
for c in F:
    print(f"  {c:24s} hist mean={hist[c].mean():+.4f} sd={hist[c].std():.4f} | "
          f"2026 mean={d[c].mean():+.4f} sd={d[c].std():.4f}")
print(f"  home_won rate            hist={hist.home_won.mean():.4f} | 2026={y.mean():.4f}\n")

# model A: trained once on all history
Xh = hist[F]; sch = StandardScaler().fit(Xh)
mh = LogisticRegression(C=1.0, max_iter=2000).fit(sch.transform(Xh), hist.home_won.values)

n = len(d)
pA = np.full(n, np.nan); pB = np.full(n, np.nan); pC = np.full(n, np.nan)
for s in range(MIN_TRAIN, n, STEP):
    tr, te = np.arange(0, s), np.arange(s, min(s+STEP, n))
    Xte = d.loc[te, F].astype(float); Xte = Xte.fillna(d.loc[tr, F].astype(float).median()).fillna(0)
    pA[te] = mh.predict_proba(sch.transform(Xte))[:,1]                    # history-trained
    Xtr = d.loc[tr, F].astype(float).fillna(d.loc[tr, F].astype(float).median()).fillna(0)
    sc = StandardScaler().fit(Xtr)
    mb = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xtr), y[tr])
    pB[te] = mb.predict_proba(sc.transform(Xte))[:,1]                     # season-trained
    # C: history-trained, then recalibrated (Platt) on the in-season window
    lg = lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    Xtr_h = sch.transform(d.loc[tr, F].astype(float).fillna(Xtr.median()).fillna(0))
    z_tr = lg(mh.predict_proba(Xtr_h)[:,1]).reshape(-1,1)
    cal = LogisticRegression(C=1e6, max_iter=2000).fit(z_tr, y[tr])
    pC[te] = cal.predict_proba(lg(pA[te]).reshape(-1,1))[:,1]

mask = ~np.isnan(pB); yy = y[mask]
v10 = np.clip(d.home_win_probability.values.astype(float),1e-6,1-1e-6)
rows=[]
for k,p in [("hist_2015_2025_trained",pA),("season_2026_trained",pB),
            ("hist_trained + in-season recalib",pC),("prod_v10 (incumbent)",v10)]:
    pp=np.clip(p[mask],1e-6,1-1e-6); corr=((pp>=.5).astype(int)==yy)
    rows.append(dict(model=k,n_train=("~26.9k" if k.startswith("hist") else "600-1713"),
                     acc=corr.mean()*100,auc=roc_auc_score(yy,pp),
                     brier=brier_score_loss(yy,pp),logloss=log_loss(yy,pp),avg_p=pp.mean()))
print(f"scored on the same {mask.sum()} walk-forward test games (home rate {yy.mean():.4f})")
print(pd.DataFrame(rows).sort_values("logloss").to_string(index=False,float_format=lambda x:f"{x:.4f}"))
