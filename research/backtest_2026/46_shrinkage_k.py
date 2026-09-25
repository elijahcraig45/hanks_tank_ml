"""Beta-binomial shrinkage constants (stabilisation points) as the v2 rate model sees them.

k per outcome class = pseudo-PAs of league-average prior; a player with k PAs gets 50%
weight on his own rate. Reported for tau=inf (all history equal) and tau=365 at 2019-07-01.
"""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "research/backtest_2026")
import importlib, json, numpy as np, pandas as pd
from pa_sim import v2
run = importlib.import_module("42_run_v2")
pa, meta, games = run.load()
venue_of = dict(zip(meta.game_pk, meta.venue_id))
vh = games.groupby("venue_id").home_team.agg(lambda s: s.value_counts().index[0]).to_dict()
out = {}
for tau in (np.inf, 365.0):
    D = v2.Data(pa[pa.game_date < "2019-12-31"], venue_of)
    Rt = v2.Rates(D, v2.Config(tau=tau)); Rt.advance((pd.Timestamp("2019-07-01") - v2.DAY0).days)
    S = Rt.snapshot(vh)
    out[str(tau)] = {r: dict(zip(v2.CLASSES, np.round(S[r]["k"], 0).tolist())) for r in ("bat", "pit")}
    print(tau, out[str(tau)])
json.dump(out, open("data/backtest_2026/rich/shrinkage_k.json", "w"), indent=1)
