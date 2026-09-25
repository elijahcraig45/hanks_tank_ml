"""Totals metrics (raw / bias-calibrated / tilted-to-market) for several variants, paired vs REF."""
import sys, importlib, numpy as np
sys.path.insert(0, "research/backtest_2026")
E = importlib.import_module("44_eval")
vs = sys.argv[1].split(","); ref = vs[0]
a, b = (sys.argv[2] if len(sys.argv) > 2 else "2016-2019").split("-"); years = list(range(int(a), int(b) + 1))
F = {}
for v in vs:
    df, z = E.summarize_variant(v, years); F[v] = E.totals_calibrated(df, z)
R0 = F[ref]; d = R0.game_date.dt.strftime("%Y%m%d").values
m = R0.crps_simcal.notna().values
for v, df in F.items():
    s = f"{v:14} bias {np.mean(df.mean_sim-df.tot):+.3f} CRPS raw {df.crps_sim[m].mean():.4f} cal {df.crps_simcal[m].mean():.4f} LS cal {df.ls_simcal[m].mean():.4f}"
    mm = m & df.crps_simmkt.notna().values
    if mm.any(): s += f" | LS simmkt {df.ls_simmkt[mm].mean():.4f}"
    if v != ref:
        g = E.boot((R0.crps_simcal - df.crps_simcal).values[m], d[m])
        g2 = E.boot((R0.ls_simcal - df.ls_simcal).values[m], d[m])
        s += f" | dCRPS vs {ref} {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}] dLS {g2[0]:+.4f} [{g2[1]:+.4f},{g2[2]:+.4f}]"
    print(s)
