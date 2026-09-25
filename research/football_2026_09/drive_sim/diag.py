import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import sim_core as sc
g = pd.read_csv("../nfl.csv")
d = sc.prep_drives(pd.read_parquet("nfl_drives.parquet"), g)
teams = pd.Index(sorted(d.posteam.unique()))
s, w = 2019, 5; t_now = s * sc.WPS + w
tr = d[(d.t < t_now) & (d.t >= t_now - 2 * sc.WPS)]
m = sc.DriveModel("b").fit(tr, t_now, teams)
# instrument: count drives + outcomes in sim
cnt = np.zeros(sc.NO); ndr = [0]
orig = np.random.Generator.random
cum, pace = m.game_tables("CIN", "NYG", False)
print("pace", pace, "ko median", np.median(m.ko), "td_p", m.td_p)
print("actual 2018-19 per game: drives", len(tr[tr.ot==0]) / tr.game_id.nunique(), "outcome mix", tr[tr.ot==0].res.value_counts(normalize=True).round(3).to_dict())
print("mean dur by outcome actual", tr[tr.ot==0].groupby('res').dur.mean().round(0).to_dict())
print("sim D medians tb3", {sc.OUT[o]: round(float(m.D[o,3,0].mean()),0) for o in range(sc.NO)})
# emulate a simple loop to count drives
rng = np.random.default_rng(1)
N=4000; tot=0; outs=np.zeros(sc.NO)
for half in (0,1):
    clock=np.full(N,1800.); yl=m.ko[rng.integers(0,sc.Q,N)]; pos=rng.integers(0,2,N); score=np.zeros((N,2))
    alive=np.ones(N,bool)
    while alive.any():
        i=np.where(alive)[0]; n=len(i)
        tb=sc.tb_of(clock[i]); per=sc.per_of(np.full(n,half),clock[i]); ylb=np.clip((yl[i]//5).astype(int),0,19)
        c=cum[pos[i],ylb,tb,per,np.full(n,4)]; out=np.minimum((rng.random(n)[:,None]>c).sum(1),8)
        outs+=np.bincount(out,minlength=9); tot+=n
        dur=m.D[out,tb,0,rng.integers(0,sc.Q,n)]*pace[pos[i]]
        sb=np.clip((yl[i]//20).astype(int),0,4); yl[i]=m.K[out,sb,rng.integers(0,sc.Q,n)]
        pos[i]=np.where(out==5,pos[i],1-pos[i]); clock[i]=np.where(out==8,0,clock[i]-dur); alive=clock>0
print("sim drives/game", tot/N, "mix", dict(zip(sc.OUT, (outs/outs.sum()).round(3))))
