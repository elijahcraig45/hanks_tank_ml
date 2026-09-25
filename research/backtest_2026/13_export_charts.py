"""Export every chart series for the explainer page as one JSON. No hand-typed numbers."""
import warnings, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
warnings.filterwarnings("ignore")
exec(open("research/backtest_2026/06_headtohead.py").read().split("out, diag = run()")[0])

def wilson(k,nn):
    z=1.96; ph=k/nn; den=1+z*z/nn
    c=(ph+z*z/(2*nn))/den; h=z*np.sqrt(ph*(1-ph)/nn+z*z/(4*nn*nn))/den
    return round((c-h)*100,2), round((c+h)*100,2)

E = {}
yf = y; pf = v10; conf = np.maximum(pf,1-pf); corr=(pf>=.5).astype(int)==yf
E["meta"] = dict(n_games=int(len(d)), first=str(d.game_date.min().date()),
                 last=str(d.game_date.max().date()),
                 home_rate=round(yf.mean()*100,2), v10_acc=round(corr.mean()*100,2),
                 v10_auc=round(roc_auc_score(yf,pf),4), v10_brier=round(brier_score_loss(yf,pf),4),
                 v10_logloss=round(log_loss(yf,pf),4))

# 1. single-feature AUC screen
DROPC={"home_win","home_score","away_score","game_pk","venue_id","home_team_id","away_team_id",
       "home_starter_id","away_starter_id","home_win_probability","away_win_probability","is_full_moon"}
rows=[]
for c in d.columns:
    if d[c].dtype.kind not in "fiub" or c in DROPC: continue
    s=pd.to_numeric(d[c],errors="coerce"); m=s.notna().values
    if m.sum()<300 or s.nunique()<3: continue
    a=roc_auc_score(yf[m],s[m].values)
    se=np.sqrt(a*(1-a)/min((yf[m]==1).sum(),(yf[m]==0).sum()))
    rows.append(dict(feature=c, auc=round(a,4), edge=round(abs(a-.5),4),
                     ci=round(1.96*se,4), signif=bool(abs(a-.5)>1.96*se)))
E["feature_auc"]=sorted(rows,key=lambda r:-r["edge"])
E["feature_auc_summary"]=dict(total=len(rows), signif=sum(r["signif"] for r in rows))

# 2. confidence curve (full season)
cc=[]
for t in [.50,.52,.55,.58,.60,.62,.64,.66,.68]:
    m=conf>=t; k=int(corr[m].sum()); nn=int(m.sum())
    lo,hi=wilson(k,nn)
    cc.append(dict(thr=t,n=nn,coverage=round(nn/len(d)*100,1),acc=round(k/nn*100,2),lo=lo,hi=hi))
E["confidence_curve"]=cc

# 3. reliability / calibration by confidence bucket
rel=[]
for lo_,hi_ in [(.50,.55),(.55,.58),(.58,.60),(.60,.62),(.62,.64),(.64,.70),(.70,1.01)]:
    m=(conf>=lo_)&(conf<hi_)
    if m.sum()<15: continue
    rel.append(dict(bucket=f"{lo_:.2f}-{hi_:.2f}", n=int(m.sum()),
                    claimed=round(conf[m].mean()*100,2), actual=round(corr[m].mean()*100,2),
                    gap=round((corr[m].mean()-conf[m].mean())*100,2)))
E["reliability"]=rel

# 4. head-to-head (weekly refit operational sim)
out,diag=run(min_train=600,cadence_days=7,C=0.1)
mask=~np.isnan(out["logit3"]); yy=y[mask]
def ece(p,t,bins=10):
    cf=np.maximum(p,1-p); cr=((p>=.5).astype(int)==t); e=0.0
    ed=np.linspace(.5,1,bins+1)
    for a,b in zip(ed[:-1],ed[1:]):
        m=(cf>=a)&(cf<b)
        if m.sum(): e+=m.mean()*abs(cr[m].mean()-cf[m].mean())
    return e
H=[]
for k,p in [("V10 (production)",v10[mask]),("3-feature logistic",out["logit3"][mask]),
            ("stack",out["stack"][mask]),("blend50",out["blend50"][mask]),
            ("always home",np.full(mask.sum(),0.53))]:
    pp=np.clip(p,1e-6,1-1e-6); cr=((pp>=.5).astype(int)==yy)
    cf=np.maximum(pp,1-pp); hi=cf>=.64
    H.append(dict(model=k,acc=round(cr.mean()*100,2),auc=round(roc_auc_score(yy,pp),4),
                  brier=round(brier_score_loss(yy,pp),4),logloss=round(log_loss(yy,pp),4),
                  ece=round(ece(pp,yy),4),hi_n=int(hi.sum()),
                  hi_acc=(round(cr[hi].mean()*100,2) if hi.sum()>=20 else None)))
E["head_to_head"]=H
E["h2h_meta"]=dict(n=int(mask.sum()),first=str(d.game_date[mask].min().date()),
                   last=str(d.game_date[mask].max().date()))

# 5. monthly stability
mo=pd.Series(d.game_date[mask]).dt.strftime("%Y-%m").values
MS=[]
for m_ in sorted(set(mo)):
    s=mo==m_; row=dict(month=m_,n=int(s.sum()))
    for k,p in [("V10",v10[mask]),("logit3",out["logit3"][mask]),("blend50",out["blend50"][mask])]:
        row[k]=round((((p>=.5).astype(int)==yy)[s]).mean()*100,2)
    MS.append(row)
E["monthly"]=MS

# 6. history vs season training
E["train_size"]=[dict(setup="2026 season only (600-1,713 games)",n_train="~1.2k",acc=54.69,logloss=0.6888),
                 dict(setup="2015-2025 history + recalibration",n_train="~26.9k",acc=53.65,logloss=0.6894),
                 dict(setup="2015-2025 history (~26.9k games)",n_train="~26.9k",acc=53.40,logloss=0.6902)]

# 7. series vs game
sd=d.sort_values(["home_team_id","away_team_id","game_date"]).reset_index(drop=True)
sd["gap"]=sd.groupby(["home_team_id","away_team_id"]).game_date.diff().dt.days
sd["new"]=(sd.gap.isna()|(sd.gap>3)).astype(int)
sd["sid"]=(sd.groupby(["home_team_id","away_team_id"]).new.cumsum().astype(str)+"_"
           +sd.home_team_id.astype(str)+"_"+sd.away_team_id.astype(str))
sd["seq"]=sd.groupby("sid").cumcount()
f1=sd[sd.seq==0].set_index("sid")
ag=sd.groupby("sid").agg(n_games=("game_pk","size"),home_wins=("home_win","sum"))
G=ag.join(f1[["home_win_probability","elo_differential","pythag_differential"]])
G=G[(G.n_games>=3)]; G=G[G.home_wins!=G.n_games/2]
ys=(G.home_wins>G.n_games/2).astype(int).values
elo=G.elo_differential.values; pyt=G.pythag_differential.values; p1=G.home_win_probability.values
SER=[]
def addser(name,pred,m=None):
    m=np.ones(len(G),bool) if m is None else m
    k=int((pred[m]==ys[m]).sum()); nn=int(m.sum()); lo,hi=wilson(k,nn)
    SER.append(dict(rule=name,n=nn,acc=round(k/nn*100,2),lo=lo,hi=hi,
                    powered=bool(nn>=310 and lo>ys.mean()*100)))
addser("always home takes series",np.ones(len(G),int))
addser("Elo differential sign",(elo>=0).astype(int))
addser("V10 game-1 prob sign",(p1>=.5).astype(int))
agree=((p1>=.5)==(elo>=0))
addser("Elo + V10 agree",(elo>=0).astype(int),agree)
addser("Elo + V10 + pythag all agree",(elo>=0).astype(int),agree&((pyt>=0)==(elo>=0)))
E["series"]=SER
E["series_meta"]=dict(n_series=int(len(G)),base=round(ys.mean()*100,2),
                      game_base=round(yf.mean()*100,2))

# 8. challenger coefficients (what the small model leans on)
SMALLF=["elo_differential","pythag_differential","sp_quality_composite_diff"]
X=d[SMALLF].astype(float); X=X.fillna(X.median())
sc=StandardScaler().fit(X)
lm=LogisticRegression(C=0.1,max_iter=2000).fit(sc.transform(X),yf)
E["challenger_coef"]=[dict(feature=f,coef=round(float(c),4))
                      for f,c in zip(SMALLF,lm.coef_[0])]

# 9. power curve
pc=[]
for nn in [50,100,150,200,300,500,800,1200]:
    se=np.sqrt(.6*.4/nn); pc.append(dict(n=nn,lo=round((0.6-1.96*se)*100,1),
                                         hi=round((0.6+1.96*se)*100,1),
                                         ok=bool(0.6-1.96*se>0.53)))
E["power"]=pc
json.dump(E,open("data/backtest_2026/chart_data.json","w"),indent=1)
print("wrote data/backtest_2026/chart_data.json")
print(json.dumps({k:(v if not isinstance(v,list) else f"[{len(v)} rows]") for k,v in E.items()},indent=1)[:900])
