"""Practicality: fit time, per-slate simulate time and peak memory for the frozen config."""
import sys, time, resource, importlib, numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "research/backtest_2026")
from pa_sim import v2
run = importlib.import_module("42_run_v2")
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # bytes on macOS
t = time.time(); pa, meta, games = run.load(); print(f"load {time.time()-t:.1f}s peak {rss():.2f} GB")
pa26 = pa[pa.game_date >= "2023-01-01"]                  # 3 seasons is all tau=365 needs (weights e^-3)
venue_of = dict(zip(meta.game_pk, meta.venue_id))
for label, P in (("all 2015-26", pa), ("2023-26 only", pa26)):
    t = time.time()
    D = v2.Data(P[P.game_date < "2026-12-31"], venue_of, None, v2.build_xtable(pa[pa.game_year == 2015]))
    vh = games.groupby("venue_id").home_team.agg(lambda s: s.value_counts().index[0]).to_dict()
    hook = pd.read_parquet("data/backtest_2026/rich/hook.parquet").merge(
        D.starts[["game_pk", "pitcher", "exp_p"]].rename(columns={"pitcher": "sp"}), on=["game_pk", "sp"], how="left")
    hook["exp_p"] = hook.exp_p.fillna(88.)
    cfg = v2.Config(xw=0.5)
    eng = v2.Engine(D, cfg, pd.read_parquet("data/backtest_2026/rich/trans.parquet"), hook, vh, run.est_tto(pa[pa.game_year == 2015]))
    hand = pa.groupby("pitcher").p_throws.agg(lambda s: int(s.mean() >= .5)).to_dict()
    st = pd.DataFrame(dict(b=D.bi, t=D.throws, s=D.stand)).groupby(["b", "t"]).s.mean()
    eng.prepare(hand, {k: int(v >= .5) for k, v in st.items()})
    t1 = time.time(); eng.fit(pd.Timestamp("2026-09-14")); tf = time.time() - t1
    g = games[(games.game_date == "2026-09-14")].merge(meta[["game_pk", "venue_id"]], on="game_pk", suffixes=("", "_m"))
    specs = [v2.GameSpec(list(r.h_lineup), list(r.a_lineup), r.h_sp, r.a_sp, r.home_team, r.away_team,
                         int(r.venue_id), 9, True) for r in g.itertuples()]
    for N in (1000, 3000, 10000):
        t2 = time.time(); res = v2.simulate(eng, specs, n=N, seed=0); v2.summarize(res, len(specs))
        print(f"[{label}] {len(specs)} games N={N}: {time.time()-t2:.2f}s ({(time.time()-t2)/len(specs)*1000:.0f} ms/game)  peak {rss():.2f} GB")
    print(f"[{label}] setup {t1-t:.1f}s  fit (first, walks all history) {tf:.1f}s")
