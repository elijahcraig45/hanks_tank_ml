"""Is the sim's total-runs SHAPE informative beyond the market line?

Compares on dev 2017-19 (earlier seasons for fitting):
  mkt_nb    NB with mean a+b*line                     (44_eval)
  mkt_emp   empirical pmf of totals among training games with the same closing line,
            tilted to the line-implied mean  (a strong, non-parametric shape baseline)
  sim_mkt   sim pmf tilted to the line-implied mean
Also P(total >= k) tail calibration for k in 12, 14.
"""
import sys, importlib, numpy as np, pandas as pd
sys.path.insert(0, "research/backtest_2026")
E = importlib.import_module("44_eval")
v = sys.argv[1] if len(sys.argv) > 1 else "full"
years = [2016, 2017, 2018, 2019] if len(sys.argv) < 3 else list(range(int(sys.argv[2].split("-")[0]), int(sys.argv[2].split("-")[1]) + 1))
df, z = E.summarize_variant(v, years); df = E.totals_calibrated(df, z)
th = z["tot_hist"]; K = th.shape[1]; y = df.tot.values.astype(int); yc = np.clip(y, 0, K - 1)
df["crps_emp"] = np.nan; df["ls_emp"] = np.nan
for yr in years:
    tr = ((df.year < yr) & df.total_line.notna()).values; te = ((df.year == yr) & df.mu_mkt.notna()).values
    if tr.sum() < 500 or te.sum() == 0: continue
    line_tr = df.total_line.values[tr]; tot_tr = np.clip(y[tr], 0, K - 1)
    allpmf = np.bincount(tot_tr, minlength=K) + 0.5; allpmf = allpmf / allpmf.sum()
    pm = np.zeros((te.sum(), K))
    for j, L in enumerate(df.total_line.values[te]):
        m = np.abs(line_tr - L) <= 0.5
        h = np.bincount(tot_tr[m], minlength=K) + 20 * allpmf
        pm[j] = h / h.sum()
    q = E.tilt(pm, df.mu_mkt.values[te])
    df.loc[te, "crps_emp"] = E.crps_discrete(q, y[te])
    df.loc[te, "ls_emp"] = -np.log(q[np.arange(te.sum()), yc[te]] * 0.999 + 0.001 / K)
m = df.crps_emp.notna() & df.crps_simmkt.notna()
d = df.game_date.dt.strftime("%Y%m%d").values
print(v, "n", m.sum())
for nm in ("mkt", "emp", "simmkt"):
    print(f"  {nm:7} CRPS {df['crps_'+nm][m].mean():.4f}  LS {df['ls_'+nm][m].mean():.4f}")
print("  sim_mkt vs mkt_emp  LS gain", np.round(E.boot((df.ls_emp - df.ls_simmkt)[m].values, d[m]), 5),
      " CRPS gain", np.round(E.boot((df.crps_emp - df.crps_simmkt)[m].values, d[m]), 5))
