"""Consolidate every measured number for the current-vs-simulated HTML report."""
import sys, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

HOLD_FROM = pd.Timestamp("2026-08-08")
E = {}

sb = json.load(open("data/backtest_2026/sim_backtest.json"))
E["alpha"] = sb["alpha"]; E["n_episodes"] = sb["n_episodes"]

d = pd.read_parquet("data/backtest_2026/ensemble_preds.parquet")
y = d.home_win.values
def wilson(k, n):
    z=1.96; ph=k/n; den=1+z*z/n
    c=(ph+z*z/(2*n))/den; h=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return round((c-h)*100,2), round((c+h)*100,2)

COLS = {"PA simulator": "sim_cal2", "3-feature logistic": "logit3", "V10 (current)": "v10",
        "sim + logit3 (stack)": "sim + logit3 (stack)"}
E["windows"] = {}
for tag, mask in [("search", (d.game_date < HOLD_FROM).values),
                  ("holdout", (d.game_date >= HOLD_FROM).values)]:
    rows = []
    for nm, c in COLS.items():
        p = d[c].values
        m = mask & np.isfinite(p)
        pp = np.clip(p[m].astype(float), 1e-6, 1-1e-6); t = y[m]
        corr = (pp >= .5).astype(int) == t
        lo, hi = wilson(int(corr.sum()), int(m.sum()))
        rows.append(dict(model=nm, n=int(m.sum()), acc=round(corr.mean()*100,2),
                         lo=lo, hi=hi, auc=round(roc_auc_score(t,pp),4),
                         brier=round(brier_score_loss(t,pp),4),
                         logloss=round(log_loss(t,pp),5)))
    b = y[mask].mean()
    rows.append(dict(model="always home", n=int(mask.sum()),
                     acc=round(max(b,1-b)*100,2), lo=None, hi=None, auc=0.5,
                     brier=round(float(np.mean((b-y[mask])**2)),4),
                     logloss=round(log_loss(y[mask], np.full(int(mask.sum()), b)),5)))
    E["windows"][tag] = rows
E["window_meta"] = dict(
    search_from=str(d.game_date.min().date()), search_to="2026-08-07",
    hold_from="2026-08-08", hold_to=str(d.game_date.max().date()))

# gate table vs the 3-feature baseline
base = {t: {r["model"]: r["logloss"] for r in E["windows"][t]} for t in ("search","holdout")}
E["gate"] = []
for nm in ["PA simulator", "V10 (current)", "sim + logit3 (stack)"]:
    gs = base["search"]["3-feature logistic"] - base["search"][nm]
    gh = base["holdout"]["3-feature logistic"] - base["holdout"][nm]
    E["gate"].append(dict(model=nm, search=round(gs,5), holdout=round(gh,5),
        verdict="better in both" if (gs>0 and gh>0) else
                ("worse in both" if (gs<0 and gh<0) else "inconsistent")))

# phase gates (measured earlier in this build)
E["phases"] = [
 dict(phase="1. PA outcome model", target="held-out plate appearances", n="183k / 164k",
      metric="log-loss vs league baseline", result="+0.0157 / +0.0111 nats", verdict="PASS"),
 dict(phase="2. Run generator", target="team-game run totals", n="4,856",
      metric="run bias / correlation", result="-0.094 runs, r=0.104", verdict="PASS"),
 dict(phase="3. Bullpen & usage", target="hook curve from PA data", n="1.9M PAs",
      metric="relief share of PAs", result="40.4% modelled per inning", verdict="PASS"),
 dict(phase="4. Game backtest", target="2026 games, both windows", n="1,749",
      metric="log-loss vs V10", result="+0.00428 / +0.00329", verdict="PASS"),
 dict(phase="4b. vs best baseline", target="2026 games, both windows", n="1,749",
      metric="log-loss vs 3-feature logistic", result="-0.0058 / -0.0004", verdict="FAIL"),
]
E["pa_model"] = [
 dict(model="league baseline", ll2025=1.53928, ll2026=1.54187),
 dict(model="batter only", ll2025=1.52749, ll2026=1.53285),
 dict(model="pitcher only", ll2025=1.53539, ll2026=1.53973),
 dict(model="log5(batter, pitcher)", ll2025=1.52361, ll2026=1.53069),
 dict(model="log5 + handedness", ll2025=1.52354, ll2026=1.53081),
]
E["run_quintiles"] = [
 dict(q="Q1", pred=3.64, actual=3.90, n=976), dict(q="Q2", pred=4.06, actual=4.40, n=974),
 dict(q="Q3", pred=4.33, actual=4.51, n=964), dict(q="Q4", pred=4.61, actual=4.45, n=971),
 dict(q="Q5", pred=5.14, actual=4.98, n=971)]
E["mc_convergence"] = [
 dict(n=1000, diff=-0.0872, se=0.0900), dict(n=20000, diff=0.0054, se=0.0204),
 dict(n=100000, diff=0.0016, se=0.0091)]
E["substrate"] = [
 dict(name="Game outcomes, 2026", rows=1300),
 dict(name="Game outcomes, 2015-2026", rows=28600),
 dict(name="Plate appearances, 2015-2026", rows=2067301),
 dict(name="Pitches, 2015-2026", rows=8850000)]
E["corr"] = dict(sim_logit3=round(float(np.corrcoef(d.sim_cal2.fillna(d.sim_cal2.mean()), d.logit3)[0,1]),4),
                 sim_v10=round(float(np.corrcoef(d.sim_cal2.fillna(d.sim_cal2.mean()), d.v10)[0,1]),4),
                 logit3_v10=round(float(np.corrcoef(d.logit3, d.v10)[0,1]),4))
E["league_rates"] = dict(zip(["K","BB","1B","2B","3B","HR","OUT","DP"],
    [22.188,9.498,14.971,4.387,0.407,3.111,43.337,2.102]))
json.dump(E, open("data/backtest_2026/comparison.json","w"), indent=1)
print("wrote data/backtest_2026/comparison.json")
for t in ("search","holdout"):
    print(f"\n{t}:")
    for r in E["windows"][t]:
        print(f"  {r['model']:24} n={r['n']:5} acc={r['acc']:6.2f} auc={r['auc']:.4f} ll={r['logloss']:.5f}")
print("\ngate:", json.dumps(E["gate"], indent=1))
