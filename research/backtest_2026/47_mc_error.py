"""How many Monte Carlo episodes are enough? Same games, same model, N varied.

Theory: with p_N ~ p + noise of variance p(1-p)/N, expected log-loss excess ~ 1/(2N) nats.
Measured on 2017 June-July games with config `full`; the N=20000 run stands in for exact p.
"""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "research/backtest_2026")
import importlib, json, time, numpy as np, pandas as pd
from pa_sim import v2
run = importlib.import_module("42_run_v2"); E = importlib.import_module("44_eval")
pa, meta, games = run.load()
year = 2017
sub = games[(games.year == year) & (games.game_date >= "2017-06-05") & (games.game_date < "2017-07-31")]
keep = games[(games.year != year) | games.game_pk.isin(sub.game_pk)]
res = {}
for N in (100, 300, 1000, 3000, 20000):
    t = time.time()
    cat = run.main("full", year, N, pa, meta, keep, tag=f"mcN{N}")
    res[N] = dict(p=cat["p_home"], gp=cat["game_pk"], sec=time.time() - t)
ref = res[20000]["p"]
y = sub.set_index("game_pk").loc[res[20000]["gp"]]
yy = (y.h_runs > y.a_runs).values.astype(int)
out = {}
for N, r in res.items():
    out[N] = dict(ll=float(E.ll(yy, r["p"]).mean()), excess_vs_20k=float(E.ll(yy, r["p"]).mean() - E.ll(yy, ref).mean()),
                  sd_p_err=float(np.std(r["p"] - ref)), theory_excess=1 / (2 * N), sec=r["sec"], n_games=len(yy))
    print(N, out[N])
json.dump(out, open("data/backtest_2026/rich/mc_error.json", "w"), indent=1)
