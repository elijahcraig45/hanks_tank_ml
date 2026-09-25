import time, numpy as np, pandas as pd
from sim_core import *
g = pd.read_csv("../nfl.csv")
d = prep_drives(pd.read_parquet("nfl_drives.parquet"), g)
teams = pd.Index(sorted(d.posteam.unique()))
s, w = 2019, 5; t_now = s * WPS + w
tr = d[(d.t < t_now) & (d.t >= t_now - 2 * WPS)]
te = d[d.t == t_now]
for v in "abc":
    t0 = time.time(); m = DriveModel(v).fit(tr, t_now, teams); t1 = time.time()
    ll = m.drive_logloss(te).mean()
    hs, as_ = simulate(m, "KC", "DET", False, s, "REG", N=4000)
    t2 = time.time()
    print(v, f"fit {t1-t0:.1f}s sim {1000*(t2-t1):.0f}ms drive-ll {ll:.4f}", "home mean", hs.mean(), "away", as_.mean(), "pwin", (hs > as_).mean(), "tie", (hs == as_).mean())
# league-average sanity: simulate all week games and compare totals
gw = g[(g.season == s) & (g.week == w)]
m = DriveModel("b").fit(tr, t_now, teams)
for _, r in gw.iterrows():
    hs, as_ = simulate(m, r.home_team.replace("OAK","LV"), r.away_team.replace("OAK","LV"), r.location == "Neutral", s, "REG", N=4000)
    print(r.home_team, r.away_team, f"sim {hs.mean():.1f}-{as_.mean():.1f} tot {hs.mean()+as_.mean():.1f} line {r.total_line} spr {r.spread_line} simspr {hs.mean()-as_.mean():.1f} actual {r.home_score}-{r.away_score}")
