"""Tune drive-model shrinkage (C) and decay (tau) on 2010-2016 by next-week drive log loss.
No game simulation needed; nothing from 2017+ is touched."""
import sys, warnings, itertools, numpy as np, pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
import sim_core as sc
g = pd.read_csv("../nfl.csv")
d = sc.prep_drives(pd.read_parquet("nfl_drives.parquet"), g)
teams = pd.Index(sorted(d.posteam.unique()))
variant = sys.argv[1] if len(sys.argv) > 1 else "b"
blocks = sorted(set(d[d.season.between(2010, 2016)].t))

def one(C, tau, t_now):
    import warnings; warnings.filterwarnings("ignore")
    tr = d[(d.t < t_now) & (d.t >= t_now - 2 * sc.WPS)]
    te = d[d.t == t_now]
    m = sc.DriveModel(variant, C=C, tau=tau).fit(tr, t_now, teams)
    l = m.drive_logloss(te)
    return C, tau, l.sum(), len(l)

grid = list(itertools.product([0.005, 0.01, 0.02], [32, 64, 1000]))
res = Parallel(n_jobs=10)(delayed(one)(C, tau, t) for C, tau in grid for t in blocks[::2])
r = pd.DataFrame(res, columns=["C", "tau", "s", "n"]).groupby(["C", "tau"]).sum()
r["ll"] = r.s / r.n
print(variant, r.ll.unstack().round(5))
print("best", r.ll.idxmin(), r.ll.min())
