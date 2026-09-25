"""Peak memory of a TRIMMED process (what a 1 GB Cloud Function would load):
PAs from 2023 on only, hook/transition tables for the 2-3 seasons actually used."""
import sys, time, resource, numpy as np, pandas as pd
sys.path.insert(0, "src")
from pa_sim import v2
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
R = "data/backtest_2026/rich/"
cols = ["game_pk", "game_date", "game_year", "batter", "pitcher", "stand", "p_throws", "inning", "top", "home_team",
        "away_team", "cls", "n_pitches", "launch_speed", "launch_angle", "is_starter"]
x15 = pd.read_parquet(R + "pa_all.parquet", columns=cols, filters=[("game_year", "==", 2015)])
xt = v2.build_xtable(x15); del x15
pa = pd.read_parquet(R + "pa_all.parquet", columns=cols, filters=[("game_year", ">=", 2023)])
meta = pd.read_parquet(R + "sched_meta.parquet", columns=["game_pk", "venue_id"])
games = pd.read_parquet(R + "games.parquet", filters=[("year", ">=", 2023)])
print(f"loaded {len(pa)} PAs, peak {rss():.2f} GB")
D = v2.Data(pa, dict(zip(meta.game_pk, meta.venue_id)), None, xt)
hook = pd.read_parquet(R + "hook.parquet", filters=[("game_year", ">=", 2024)])
hook = hook.merge(D.starts[["game_pk", "pitcher", "exp_p"]].rename(columns={"pitcher": "sp"}), on=["game_pk", "sp"], how="left")
hook["exp_p"] = hook.exp_p.fillna(88.)
trans = pd.read_parquet(R + "trans.parquet", filters=[("year", ">=", 2022)])
vh = games.groupby("venue_id" if "venue_id" in games else "home_team").size()
vh = games.merge(meta, on="game_pk").groupby("venue_id").home_team.agg(lambda s: s.value_counts().index[0]).to_dict()
eng = v2.Engine(D, v2.Config(xw=0.5), trans, hook, vh, None)
hand = pa.groupby("pitcher").p_throws.agg(lambda s: int(s.mean() >= .5)).to_dict()
st = pd.DataFrame(dict(b=D.bi, t=D.throws, s=D.stand)).groupby(["b", "t"]).s.mean()
eng.prepare(hand, {k: int(v >= .5) for k, v in st.items()})
t = time.time(); eng.fit(pd.Timestamp("2026-09-01")); print(f"fit {time.time()-t:.1f}s peak {rss():.2f} GB")
g = games[games.game_date.between("2026-09-01", "2026-09-02")].merge(meta, on="game_pk")
specs = [v2.GameSpec(list(r.h_lineup), list(r.a_lineup), r.h_sp, r.a_sp, r.home_team, r.away_team, int(r.venue_id), 9, True)
         for r in g.itertuples()]
t = time.time(); res = v2.simulate(eng, specs, n=3000); v2.summarize(res, len(specs))
print(f"{len(specs)} games N=3000 {time.time()-t:.1f}s, peak {rss():.2f} GB")
