"""Is the high-conviction tier preserved? Plus significance, stability, sensitivity.

Key question 06 raised: shrinkage and the small logistic are better calibrated but
almost never clear a FIXED 0.64 confidence threshold, so the 'high' tier empties out.
But shrinkage is a MONOTONE transform -- it cannot change which games a model ranks as
most confident. So the tier only collapses because the threshold is an absolute number.
Re-deriving tiers as QUANTILES tests pick quality independently of calibration.
"""
import warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from scipy.stats import binomtest
warnings.filterwarnings("ignore")
exec(open("research/backtest_2026/06_headtohead.py").read().split("out, diag = run()")[0])

out, diag = run()
mask = ~np.isnan(out["logit3"]); yy = y[mask]
P = {"prod_v10 (INCUMBENT)": v10[mask], "v10_shrunk": out["v10_shrunk"][mask],
     "logit3": out["logit3"][mask], "stack": out["stack"][mask], "blend50": out["blend50"][mask]}
N = mask.sum()

print("="*78)
print("1. TOP-K PICK QUALITY -- accuracy on the k games each model is most sure about")
print("   (calibration-free: pure ranking test, which is what a conviction product needs)")
print("="*78)
ks = [50, 100, 150, 200, 300, 500]
rows = []
for k, p in P.items():
    conf = np.maximum(p, 1 - p); corr = ((p >= .5).astype(int) == yy)
    order = np.argsort(-conf)
    rows.append({"model": k, **{f"top{kk}": corr[order[:kk]].mean() * 100 for kk in ks}})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\n  Proof that shrinkage preserves the pick set (monotone transform):")
ci = np.maximum(P["prod_v10 (INCUMBENT)"], 1 - P["prod_v10 (INCUMBENT)"])
cs = np.maximum(P["v10_shrunk"], 1 - P["v10_shrunk"])
for kk in [100, 130, 200]:
    a = set(np.argsort(-ci)[:kk]); b = set(np.argsort(-cs)[:kk])
    print(f"    top{kk}: {len(a & b)}/{kk} identical games  (Spearman rho={pd.Series(ci).corr(pd.Series(cs), method='spearman'):.4f})")

print("\n" + "="*78)
print("2. SIGNIFICANCE vs INCUMBENT (paired, same games)")
print("="*78)
pb = np.clip(P["prod_v10 (INCUMBENT)"], 1e-6, 1-1e-6)
lb = -(yy*np.log(pb)+(1-yy)*np.log(1-pb)); cb = ((pb >= .5).astype(int) == yy)
rng = np.random.default_rng(0)
for k in ["v10_shrunk", "logit3", "stack", "blend50"]:
    pa = np.clip(P[k], 1e-6, 1-1e-6)
    la = -(yy*np.log(pa)+(1-yy)*np.log(1-pa)); ca = ((pa >= .5).astype(int) == yy)
    dd = lb - la
    bs = np.array([dd[rng.integers(0, N, N)].mean() for _ in range(4000)])
    n01 = int(((~ca) & cb).sum()); n10 = int((ca & (~cb)).sum())
    pv = binomtest(n10, n01+n10, 0.5).pvalue if (n01+n10) > 0 else 1.0
    print(f"  {k:12s} logloss {dd.mean():+.5f} CI[{np.percentile(bs,2.5):+.5f},{np.percentile(bs,97.5):+.5f}] "
          f"P(better)={(bs>0).mean():.3f} | acc {(ca.mean()-cb.mean())*100:+.2f}pp "
          f"flips {n10}W/{n01}L McNemar p={pv:.3f}")

print("\n" + "="*78)
print("3. MONTHLY STABILITY (accuracy; does any edge persist or is it one hot month?)")
print("="*78)
mo = pd.Series(pd.to_datetime(dates[mask])).dt.strftime("%Y-%m").values
tab = []
for k, p in P.items():
    corr = ((p >= .5).astype(int) == yy)
    row = {"model": k}
    for m in sorted(set(mo)):
        s = mo == m
        row[m] = corr[s].mean()*100
    tab.append(row)
t = pd.DataFrame(tab)
t["n_games"] = ""
print(t.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("   games/month: " + ", ".join(f"{m}={int((mo==m).sum())}" for m in sorted(set(mo))))

print("\n" + "="*78)
print("4. SENSITIVITY -- does the result survive different harness settings?")
print("="*78)
print(f"{'min_train':>9} {'cadence':>8} {'C':>6} {'n_test':>7} | " +
      " ".join(f"{k[:11]:>11}" for k in ["v10(incumb)","logit3","stack","blend50"]) + "   (log-loss)")
for mt in [400, 600, 800]:
    for cad in [7, 14]:
        for C in [0.03, 0.1, 0.3]:
            o, _ = run(min_train=mt, cadence_days=cad, C=C)
            mk = ~np.isnan(o["logit3"]); ty = y[mk]
            if mk.sum() < 200: continue
            vals = [log_loss(ty, np.clip(v10[mk],1e-6,1-1e-6))] + \
                   [log_loss(ty, np.clip(o[k][mk],1e-6,1-1e-6)) for k in ["logit3","stack","blend50"]]
            best = int(np.argmin(vals))
            cells = " ".join(f"{v:>11.5f}" + ("*" if i == best else " ") for i, v in enumerate(vals))
            print(f"{mt:>9} {cad:>8} {C:>6} {mk.sum():>7} | {cells}")
print("   * = best in row")
