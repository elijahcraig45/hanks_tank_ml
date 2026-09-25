"""Combine the two mechanisms that survive: in-season low-dimensional + historical GBM.

They carry different information. The season model uses 3 features including
sp_quality_composite_diff (V10-only, absent from history). The historical GBM uses 54
features across 11 z-scored, recency-weighted seasons. Their AUCs differ (0.5814 vs
0.5566) so their errors are not the same errors.

Any winner must be better in BOTH the search window and the untouched holdout -- the
standard the margin models failed.
"""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
warnings.filterwarnings("ignore")

HOLD_FROM = pd.Timestamp("2026-08-08"); STEP = 100
h = pd.concat([pd.read_parquet("data/training/train_v8_2015_2024.parquet"),
               pd.read_parquet("data/training/val_v8_2025.parquet")], ignore_index=True)
c = pd.read_parquet("data/backtest_2026/full_features_2026.parquet")
c = c.sort_values(["game_date","game_pk"]).reset_index(drop=True)
common = sorted(set(h.columns) & set(c.columns))
F54 = [x for x in common if h[x].dtype.kind in "fiub" and c[x].dtype.kind in "fiub"
       and x not in {"game_pk","home_team_id","away_team_id","home_won","home_win","year"}]
F3 = ["elo_differential","pythag_differential","sp_quality_composite_diff"]
h["season"] = pd.to_datetime(h.game_date).dt.year
yh = h.home_won.values.astype(int); yc = c.home_win.values
Xh = h[F54].astype(float).replace([np.inf,-np.inf],np.nan); Xh = Xh.fillna(Xh.median())
Xc54 = c[F54].astype(float).replace([np.inf,-np.inf],np.nan).fillna(Xh.median())
Xc3 = c[F3].astype(float).replace([np.inf,-np.inf],np.nan)
Xc3 = Xc3.fillna(Xc3.median())

Zh = Xh.copy()
for s,g in h.groupby("season"):
    mu,sd = Xh.loc[g.index].mean(), Xh.loc[g.index].std().replace(0,1)
    Zh.loc[g.index] = (Xh.loc[g.index]-mu)/sd
wt = h.season.map({s:0.85**(2026-s) for s in h.season.unique()}).values

# the historical GBM is trained ONCE on history only -- it never sees 2026
gbm = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05,
        l2_regularization=1.0, random_state=0).fit(Zh, yh, sample_weight=wt)

hold = (c.game_date>=HOLD_FROM).values
sidx, hidx = np.where(~hold)[0], np.where(hold)[0]
lg = lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))

def preds(idx):
    """walk-forward: season model refit per block; gbm fixed; weights fit on train"""
    out = {k: np.full(len(c), np.nan) for k in
           ["season3","histgbm","avg","tuned","stack"]}
    start = idx[0]
    for s in range(start, len(c), STEP):
        blk = np.arange(s, min(s+STEP, len(c))); blk = blk[np.isin(blk, idx)]
        if len(blk)==0: continue
        tr = np.arange(0, s)
        if len(tr) < 300: continue
        # A: in-season 3-feature logistic
        A, B = Xc3.iloc[tr], Xc3.iloc[blk]
        sc = StandardScaler().fit(A)
        m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(A), yc[tr])
        pa_tr = m.predict_proba(sc.transform(A))[:,1]
        pa = m.predict_proba(sc.transform(B))[:,1]
        # B: historical GBM, z-scored with in-season stats known so far
        ref = Xc54.iloc[tr]; mu, sd = ref.mean(), ref.std().replace(0,1)
        pb_tr = gbm.predict_proba((ref-mu)/sd)[:,1]
        pb = gbm.predict_proba((Xc54.iloc[blk]-mu)/sd)[:,1]
        out["season3"][blk] = pa; out["histgbm"][blk] = pb
        out["avg"][blk] = 0.5*pa + 0.5*pb
        # weight tuned on the training window only
        ws = np.linspace(0,1,21)
        best = min(ws, key=lambda w: log_loss(yc[tr], np.clip(w*pa_tr+(1-w)*pb_tr,1e-6,1-1e-6)))
        out["tuned"][blk] = best*pa + (1-best)*pb
        # logistic stack on the two logits, fit on the training window
        st = LogisticRegression(C=1.0, max_iter=3000).fit(
            np.column_stack([lg(pa_tr), lg(pb_tr)]), yc[tr])
        out["stack"][blk] = st.predict_proba(np.column_stack([lg(pa), lg(pb)]))[:,1]
    return out

NAMES = {"season3":"in-season 3-feature logistic","histgbm":"historical GBM (11 seasons)",
         "avg":"average of the two","tuned":"weight tuned on train window",
         "stack":"logistic stack of the two"}
store = {}
for tag, idx in [("2026 SEARCH WINDOW", sidx), ("UNTOUCHED HOLDOUT", hidx)]:
    P = preds(idx)
    print(f"\n{'='*88}\n{tag}  (n={len(idx)})\n{'='*88}")
    print(f"{'mechanism':34}{'n':>6}{'acc':>8}{'auc':>9}{'brier':>9}{'logloss':>10}")
    store[tag] = {}
    for k, nm in NAMES.items():
        m = ~np.isnan(P[k][idx]); pp = np.clip(P[k][idx][m],1e-6,1-1e-6); t = yc[idx][m]
        r = dict(acc=((pp>=.5).astype(int)==t).mean()*100, auc=roc_auc_score(t,pp),
                 brier=brier_score_loss(t,pp), ll=log_loss(t,pp), p=pp, t=t, n=int(m.sum()))
        store[tag][k]=r
        print(f"{nm:34}{r['n']:>6}{r['acc']:>8.2f}{r['auc']:>9.4f}{r['brier']:>9.4f}{r['ll']:>10.5f}")
    bs_=yc[idx].mean()
    print(f"{'   always home':34}{len(idx):>6}{max(bs_,1-bs_)*100:>8.2f}{0.5:>9.4f}"
          f"{np.mean((bs_-yc[idx])**2):>9.4f}{log_loss(yc[idx],np.full(len(idx),bs_)):>10.5f}")

print(f"\n{'='*88}\nCONSISTENCY GATE -- log-loss gain over the in-season 3-feature model\n{'='*88}")
print(f"{'mechanism':34}{'search':>11}{'holdout':>11}   verdict")
for k, nm in NAMES.items():
    if k=="season3": continue
    gs = store["2026 SEARCH WINDOW"]["season3"]["ll"] - store["2026 SEARCH WINDOW"][k]["ll"]
    gh = store["UNTOUCHED HOLDOUT"]["season3"]["ll"] - store["UNTOUCHED HOLDOUT"][k]["ll"]
    v = "BETTER IN BOTH" if (gs>0 and gh>0) else ("worse in both" if (gs<0 and gh<0) else "inconsistent")
    print(f"{nm:34}{gs:>+11.5f}{gh:>+11.5f}   {v}")

# significance for anything that passed the gate
print()
for k, nm in NAMES.items():
    if k=="season3": continue
    gs = store["2026 SEARCH WINDOW"]["season3"]["ll"] - store["2026 SEARCH WINDOW"][k]["ll"]
    gh = store["UNTOUCHED HOLDOUT"]["season3"]["ll"] - store["UNTOUCHED HOLDOUT"][k]["ll"]
    if not (gs>0 and gh>0): continue
    for tag in ["2026 SEARCH WINDOW","UNTOUCHED HOLDOUT"]:
        a=store[tag][k]; b=store[tag]["season3"]
        t=a["t"]; pa,pb=a["p"],b["p"]
        la=-(t*np.log(pa)+(1-t)*np.log(1-pa)); lb=-(t*np.log(pb)+(1-t)*np.log(1-pb))
        dd=lb-la; rng=np.random.default_rng(0)
        bt=np.array([dd[rng.integers(0,len(dd),len(dd))].mean() for _ in range(4000)])
        print(f"{nm} vs season3 [{tag}]: logloss {dd.mean():+.5f} "
              f"CI[{np.percentile(bt,2.5):+.5f},{np.percentile(bt,97.5):+.5f}] "
              f"P(better)={(bt>0).mean():.3f}  acc {a['acc']-b['acc']:+.2f}pp  AUC {a['auc']-b['auc']:+.4f}")
