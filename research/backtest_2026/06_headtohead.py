"""Decision-grade head-to-head: candidates vs the incumbent V10.

Operational simulation, not an idealised one. The model refits on a fixed calendar
cadence (default weekly, matching a Cloud Scheduler retrain), trains only on games
completed before the refit date, and serves those frozen parameters for the whole
following week. Shrinkage lambda is refit the same way on completed games only.

Everything -- imputation medians, scaler, coefficients, lambda, blend weight -- is
fit inside the training window. No candidate sees a game before predicting it.
"""
import warnings, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
warnings.filterwarnings("ignore")

SMALL = ["elo_differential", "pythag_differential", "sp_quality_composite_diff"]
d = pd.read_parquet("data/backtest_2026/games_2026_pregame.parquet")
d = d.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
for c in SMALL: d[c] = pd.to_numeric(d[c], errors="coerce")
y = d.home_win.values
v10 = np.clip(d.home_win_probability.values.astype(float), 1e-6, 1 - 1e-6)
dates = d.game_date.values

def ll(p, t):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()

def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def run(min_train=600, cadence_days=7, C=0.1):
    """Returns dict of candidate -> out-of-sample probability array (NaN where unserved)."""
    n = len(d)
    out = {k: np.full(n, np.nan) for k in ["logit3", "v10_shrunk", "stack", "blend50"]}
    diag = []
    uniq = np.array(sorted(pd.unique(dates)))
    # candidate refit dates: every cadence_days, once min_train games are complete
    start_i = min_train
    refit_dates = []
    cur = dates[start_i]
    while cur <= uniq[-1]:
        refit_dates.append(cur)
        cur = cur + np.timedelta64(cadence_days, "D")
    for i, rd in enumerate(refit_dates):
        nxt = rd + np.timedelta64(cadence_days, "D")
        tr = np.where(dates < rd)[0]
        te = np.where((dates >= rd) & (dates < nxt))[0]
        if len(te) == 0 or len(tr) < min_train: continue
        Xtr = d.loc[tr, SMALL].astype(float); med = Xtr.median()
        Xtr = Xtr.fillna(med).fillna(0)
        Xte = d.loc[te, SMALL].astype(float).fillna(med).fillna(0)
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), y[tr])
        p_tr = m.predict_proba(sc.transform(Xtr))[:, 1]
        p_te = m.predict_proba(sc.transform(Xte))[:, 1]
        out["logit3"][te] = p_te
        base = y[tr].mean()
        lams = np.linspace(0.1, 1.0, 19)
        lam = lams[int(np.argmin([ll(base + l * (v10[tr] - base), y[tr]) for l in lams]))]
        out["v10_shrunk"][te] = base + lam * (v10[te] - base)
        S_tr = np.column_stack([lg(p_tr), lg(v10[tr])])
        st = LogisticRegression(C=1.0, max_iter=2000).fit(S_tr, y[tr])
        out["stack"][te] = st.predict_proba(np.column_stack([lg(p_te), lg(v10[te])]))[:, 1]
        out["blend50"][te] = 0.5 * p_te + 0.5 * (base + lam * (v10[te] - base))
        diag.append((str(pd.Timestamp(rd).date()), len(tr), len(te), lam, base))
    return out, pd.DataFrame(diag, columns=["refit_date", "n_train", "n_test", "lambda", "base"])

def ece(p, t, bins=10):
    conf = np.maximum(p, 1 - p); corr = ((p >= .5).astype(int) == t)
    e, edges = 0.0, np.linspace(0.5, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return e

def metrics(p, t):
    p = np.clip(p, 1e-6, 1 - 1e-6); corr = ((p >= .5).astype(int) == t)
    conf = np.maximum(p, 1 - p)
    hi = conf >= .64; mid = (conf >= .57) & (conf < .64)
    return dict(acc=corr.mean() * 100, auc=roc_auc_score(t, p),
                brier=brier_score_loss(t, p), logloss=log_loss(t, p), ece=ece(p, t),
                hi_n=int(hi.sum()), hi_acc=(corr[hi].mean() * 100 if hi.sum() >= 20 else np.nan),
                mid_n=int(mid.sum()), mid_acc=(corr[mid].mean() * 100 if mid.sum() >= 20 else np.nan))

out, diag = run()
mask = ~np.isnan(out["logit3"]); yy = y[mask]
print(f"OPERATIONAL SIM: weekly refit, min_train=600, C=0.1")
print(f"  refits={len(diag)}  test games={mask.sum()}  "
      f"{pd.Timestamp(dates[mask][0]).date()} -> {pd.Timestamp(dates[mask][-1]).date()}")
print(f"  lambda: mean={diag['lambda'].mean():.2f} range {diag['lambda'].min():.2f}-{diag['lambda'].max():.2f}")
print(f"  home win rate on test = {yy.mean():.4f}\n")

cands = {"prod_v10 (INCUMBENT)": v10[mask], "v10_shrunk": out["v10_shrunk"][mask],
         "logit3": out["logit3"][mask], "stack": out["stack"][mask],
         "blend50": out["blend50"][mask],
         "always_home": np.full(mask.sum(), 0.53)}
rows = [dict(model=k, **metrics(p, yy)) for k, p in cands.items()]
r = pd.DataFrame(rows).sort_values("logloss")
print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
np.save("data/backtest_2026/h2h_mask.npy", mask)
pd.DataFrame({k: v for k, v in cands.items()}).to_parquet("data/backtest_2026/h2h_preds.parquet")
diag.to_csv("data/backtest_2026/h2h_refit_diag.csv", index=False)
