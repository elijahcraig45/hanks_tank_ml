import sys, os
W = sys.argv[1]
sys.argv = ["exp_v2b.py", W, W + "/mod", "3", "100"]
sys.path.insert(0, W)
import numpy as np, pandas as pd
import exp_v2b as B
import exp_v2 as X
from concurrent.futures import ProcessPoolExecutor

pd.set_option("display.width", 200)
rows = pd.read_parquet(f"{W}/v2b_rows.parquet")
rows["bucket"] = pd.cut(rows.week, [3, 6, 9, 12], labels=["wk4-6", "wk7-9", "wk10-12"])
keep = ["v1 shipped", "R0 add .5 sum v1order", "R1 addfloor .5 sum v1order", "R2 mult 0.5 sum v1order", "R2 mult 0.75 sum v1order"]
print("== A. by stage of the season (all seasons; acc = next-3-weeks, top25 = contradicted top-25 pairs, rel = split-half)")
t = rows[rows.arm.isin(keep)].groupby(["arm", "bucket"], observed=True).agg(fn=("fn", "sum"), fw=("fw", "sum"), tn=("tn", "sum"), tc=("tc", "sum"), rel=("rel", "mean"))
t["acc"] = 1 - t.fw / t.fn; t["top25"] = t.tc / t.tn.clip(lower=1)
print(t[["acc", "top25", "rel"]].round(3).unstack("bucket").reindex(keep).to_string())

print("\n== B. benchmark size (mult .5, v1 ordering z=.5); BENCH_N = how many top teams define 'a typical top-N team'")
one = {"R2 mult 0.5 sum v1order": B.ARMS["R2 mult 0.5 sum v1order"]}
for n in (10, 25, 40, 70):
    B.v2.BENCH_N = n
    B.ARMS = one
    with ProcessPoolExecutor(3, initializer=X.init, initargs=(X.ev.load_games("cfb"),)) as pool:
        res = list(pool.map(B.run, [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]))
    r = pd.concat([x[0] for x in res]); r = r[r.arm == "R2 mult 0.5 sum v1order"]
    r["part"] = np.where(r.season.isin([2017, 2018, 2019, 2021]), "tune", "eval")
    for part, d in r.groupby("part"):
        print(f"N={n:3d} {part}: acc {1 - d.fw.sum() / d.fn.sum():.4f}  top25 {d.tc.sum() / d.tn.sum():.4f}  rel {d.rel.mean():.4f}")
