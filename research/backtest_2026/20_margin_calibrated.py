"""The margin model ranks best but is mis-calibrated. Learn the mapping instead.

19_ mapped predicted margin -> P(win) by assuming a t/normal residual distribution,
which produced the best holdout AUC (0.5889) but the worst log-loss -- a scaling
defect, not a ranking one. Here the margin->probability map is FIT on the training
window (logistic on the predicted margin), and the margin signal is also offered to a
logistic alongside the three binary features.
"""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
warnings.filterwarnings("ignore")

HOLD_FROM = pd.Timestamp("2026-08-08"); MIN_TRAIN, STEP = 500, 100
b = pd.read_parquet("data/backtest_2026/full_features_2026.parquet")
nf = pd.read_parquet("data/backtest_2026/statcast_new_features.parquet").drop(columns=["game_date"])
d = b.merge(nf, on="game_pk", how="inner").sort_values(["game_date","game_pk"]).reset_index(drop=True)
d["sp_form_k_diff"] = d.home_sp_form_k - d.away_sp_form_k
d["sp_form_xwoba_diff"] = d.away_sp_form_xwoba - d.home_sp_form_xwoba
y = d.home_win.values
mrg = (d.home_score - d.away_score).values.astype(float)
F3 = ["elo_differential","pythag_differential","sp_quality_composite_diff"]
F5 = F3 + ["sp_form_k_diff","sp_form_xwoba_diff"]
hold = (d.game_date >= HOLD_FROM).values
sidx, hidx = np.where(~hold)[0], np.where(hold)[0]

def prep(cols, tr, te):
    A,B = d.loc[tr,cols].astype(float), d.loc[te,cols].astype(float)
    A=A.replace([np.inf,-np.inf],np.nan); B=B.replace([np.inf,-np.inf],np.nan)
    m=A.median(); return A.fillna(m).fillna(0), B.fillna(m).fillna(0)

def margin_mu(cols, tr, te, alpha=10.0):
    A,B = prep(cols,tr,te); sc=StandardScaler().fit(A)
    r=Ridge(alpha=alpha).fit(sc.transform(A), mrg[tr])
    return r.predict(sc.transform(A)), r.predict(sc.transform(B))

def f_binary(cols,C=0.1):
    def fn(tr,te):
        A,B=prep(cols,tr,te); sc=StandardScaler().fit(A)
        m=LogisticRegression(C=C,max_iter=3000).fit(sc.transform(A),y[tr])
        return m.predict_proba(sc.transform(B))[:,1]
    return fn

def f_margin_cal(cols,C=1.0):
    """ridge on margin, then a logistic fit on that predicted margin"""
    def fn(tr,te):
        mu_tr,mu_te = margin_mu(cols,tr,te)
        cal=LogisticRegression(C=C,max_iter=3000).fit(mu_tr.reshape(-1,1),y[tr])
        return cal.predict_proba(mu_te.reshape(-1,1))[:,1]
    return fn

def f_margin_plus(cols,C=0.1):
    """logistic on [predicted margin] + the binary features together"""
    def fn(tr,te):
        mu_tr,mu_te = margin_mu(cols,tr,te)
        A,B=prep(F3,tr,te)
        Atr=np.column_stack([mu_tr,A.values]); Ate=np.column_stack([mu_te,B.values])
        sc=StandardScaler().fit(Atr)
        m=LogisticRegression(C=C,max_iter=3000).fit(sc.transform(Atr),y[tr])
        return m.predict_proba(sc.transform(Ate))[:,1]
    return fn

def f_avg(cols):
    a,bb = f_margin_cal(cols), f_binary(F3)
    return lambda tr,te: 0.5*a(tr,te)+0.5*bb(tr,te)

CANDS={
 "binary logistic, 3 feat (control)": f_binary(F3),
 "margin->logistic cal, 3 feat":      f_margin_cal(F3),
 "margin->logistic cal, 5 feat":      f_margin_cal(F5),
 "margin + 3 feat jointly":           f_margin_plus(F5),
 "avg(margin cal, binary)":           f_avg(F5),
}

def wf(idx,fn):
    p=np.full(len(d),np.nan)
    for s in range(MIN_TRAIN,len(idx),STEP):
        tr,te=idx[:s],idx[s:min(s+STEP,len(idx))]; p[te]=fn(tr,te)
    return p
def serve(fn):
    p=np.full(len(d),np.nan)
    for s in range(len(sidx),len(d),STEP):
        tr,te=np.arange(0,s),np.arange(s,min(s+STEP,len(d))); p[te]=fn(tr,te)
    return p

for tag,idx,runner in [("SEARCH SET — walk-forward",sidx,wf),("UNTOUCHED HOLDOUT",hidx,serve)]:
    print(f"\n{'='*84}\n{tag}  (n={len(idx)})\n{'='*84}")
    print(f"{'mechanism':36}{'acc':>8}{'auc':>9}{'brier':>9}{'logloss':>10}")
    store={}
    for k,fn in CANDS.items():
        p = runner(idx,fn) if runner is wf else runner(fn)
        m=~np.isnan(p[idx]); pp=np.clip(p[idx][m],1e-6,1-1e-6); t=y[idx][m]
        store[k]=(pp,t)
        print(f"{k:36}{((pp>=.5).astype(int)==t).mean()*100:>8.2f}"
              f"{roc_auc_score(t,pp):>9.4f}{brier_score_loss(t,pp):>9.4f}{log_loss(t,pp):>10.5f}")
    bs_=y[idx].mean()
    print(f"{'   always home':36}{max(bs_,1-bs_)*100:>8.2f}{0.5:>9.4f}"
          f"{np.mean((bs_-y[idx])**2):>9.4f}{log_loss(y[idx],np.full(len(idx),bs_)):>10.5f}")
    if tag.startswith("UNTOUCHED"):
        kc="binary logistic, 3 feat (control)"
        best=min([k for k in store if k!=kc], key=lambda k: log_loss(store[k][1],store[k][0]))
        pa,t=store[best]; pb,_=store[kc]
        la=-(t*np.log(pa)+(1-t)*np.log(1-pa)); lb=-(t*np.log(pb)+(1-t)*np.log(1-pb))
        dd=lb-la; rng=np.random.default_rng(0)
        bt=np.array([dd[rng.integers(0,len(dd),len(dd))].mean() for _ in range(4000)])
        print(f"\nbest ({best}) vs control:")
        print(f"  logloss {dd.mean():+.5f} CI[{np.percentile(bt,2.5):+.5f},{np.percentile(bt,97.5):+.5f}]"
              f" P(better)={(bt>0).mean():.3f}")
        print(f"  acc {(((pa>=.5).astype(int)==t).mean()-((pb>=.5).astype(int)==t).mean())*100:+.2f}pp"
              f"  AUC {roc_auc_score(t,pa)-roc_auc_score(t,pb):+.4f}")
