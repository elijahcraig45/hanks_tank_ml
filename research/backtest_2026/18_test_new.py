"""Do the pitch-level features add signal? Screen, then search, then one holdout call.

The holdout is defined BY DATE (>= 2026-08-08) so it is the same set of games the
earlier search was scored on, even though the merged frame has a different row count.
"""
import warnings, json, time
import numpy as np, pandas as pd, optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import xgboost as xgb
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)

HOLD_FROM = pd.Timestamp("2026-08-08")
MIN_TRAIN, STEP = 500, 100

base = pd.read_parquet("data/backtest_2026/full_features_2026.parquet")
new  = pd.read_parquet("data/backtest_2026/statcast_new_features.parquet").drop(columns=["game_date"])
d = base.merge(new, on="game_pk", how="inner").sort_values(["game_date","game_pk"]).reset_index(drop=True)

# differentials: for xwOBA allowed, LOWER is better, so home edge = away - home
d["sp_form_xwoba_diff"]  = d.away_sp_form_xwoba - d.home_sp_form_xwoba
d["sp_form_k_diff"]      = d.home_sp_form_k - d.away_sp_form_k
d["sp_form_velo_diff"]   = d.home_sp_form_velo - d.away_sp_form_velo
d["bp_pitches_3d_diff"]  = d.away_bp_pitches_3d - d.home_bp_pitches_3d
d["bp_xwoba_7d_diff"]    = d.away_bp_xwoba_7d - d.home_bp_xwoba_7d
NEW = ["sp_form_xwoba_diff","sp_form_k_diff","sp_form_velo_diff",
       "bp_pitches_3d_diff","bp_xwoba_7d_diff",
       "home_sp_form_xwoba","away_sp_form_xwoba","home_bp_pitches_3d","away_bp_pitches_3d",
       "home_bp_xwoba_7d","away_bp_xwoba_7d","home_sp_form_k","away_sp_form_k"]
y = d.home_win.values
print(f"merged {len(d)} games; {(d.game_date>=HOLD_FROM).sum()} in holdout\n")

print("="*82)
print("SINGLE-FEATURE AUC OF THE NEW PITCH-LEVEL FEATURES (full merged season)")
print("="*82)
rows=[]
for c in NEW:
    v = pd.to_numeric(d[c], errors="coerce").values; m=~np.isnan(v)
    if m.sum()<300: continue
    a = roc_auc_score(y[m], v[m])
    se = np.sqrt(a*(1-a)/min((y[m]==1).sum(),(y[m]==0).sum()))
    rows.append(dict(feature=c, auc=round(a,4), edge=round(abs(a-.5),4),
                     ci=round(1.96*se,4), signif="YES" if abs(a-.5)>1.96*se else ""))
S=pd.DataFrame(rows).sort_values("edge",ascending=False)
print(S.to_string(index=False))
print(f"\nclearing the noise floor: {(S.signif=='YES').sum()} of {len(S)}")
print("for reference, pythag_differential scores AUC 0.5457 on this target")

# ---- walk-forward comparison ----
MANUAL3 = ["elo_differential","pythag_differential","sp_quality_composite_diff"]
EXCL = {"game_pk","game_date","home_team_id","away_team_id","home_score","away_score",
        "home_win","computed_at","rn"}
ALLF = [c for c in d.columns if c not in EXCL and d[c].dtype.kind in "fiub"]
X = d[ALLF].astype(float).replace([np.inf,-np.inf], np.nan)
hold = (d.game_date>=HOLD_FROM).values
sidx = np.where(~hold)[0]; hidx = np.where(hold)[0]

def wf(cols, idx, C=0.1, model="logit"):
    pr=np.full(len(idx), np.nan)
    for s in range(MIN_TRAIN, len(idx), STEP):
        tr=idx[:s]; te=idx[s:min(s+STEP,len(idx))]
        Xtr,Xte = X.loc[tr,cols], X.loc[te,cols]
        med=Xtr.median(); Xtr=Xtr.fillna(med).fillna(0); Xte=Xte.fillna(med).fillna(0)
        if model=="logit":
            sc=StandardScaler().fit(Xtr)
            m=LogisticRegression(C=C,max_iter=3000).fit(sc.transform(Xtr),y[tr])
            pr[s:s+len(te)]=m.predict_proba(sc.transform(Xte))[:,1]
        else:
            m=xgb.XGBClassifier(max_depth=2,n_estimators=200,learning_rate=0.03,
                reg_lambda=10,min_child_weight=20,eval_metric="logloss",
                verbosity=0,random_state=0).fit(Xtr,y[tr])
            pr[s:s+len(te)]=m.predict_proba(Xte)[:,1]
    mm=~np.isnan(pr)
    return log_loss(y[idx][mm], np.clip(pr[mm],1e-6,1-1e-6)), pr, mm

print("\n"+"="*82)
print("WALK-FORWARD LOG-LOSS INSIDE THE SEARCH SET (does adding them help?)")
print("="*82)
sets = {
    "3 features (incumbent challenger)": MANUAL3,
    "3 + SP recent form":                MANUAL3+["sp_form_xwoba_diff","sp_form_k_diff"],
    "3 + bullpen":                       MANUAL3+["bp_pitches_3d_diff","bp_xwoba_7d_diff"],
    "3 + all 13 new":                    MANUAL3+NEW,
    "all new only (13)":                 NEW,
}
res={}
for k,cols in sets.items():
    ll,_,_ = wf([c for c in cols if c in X.columns], sidx)
    res[k]=ll
    print(f"  {k:36s} {ll:.5f}  ({ll-res['3 features (incumbent challenger)']:+.5f})")

# ---- focused search WITH the new features, then one holdout evaluation ----
print("\n"+"="*82)
print("AUTONOMOUS SEARCH (150 trials) WITH THE NEW FEATURES AVAILABLE")
print("="*82)
def pick(tr,strategy,k):
    if strategy=="manual3": return MANUAL3
    if strategy=="manual3+new": return MANUAL3+NEW
    if strategy=="all": return ALLF
    sc=[]
    for c in ALLF:
        v=X[c].values[tr]; m=~np.isnan(v)
        if m.sum()<100 or np.unique(v[m]).size<3: continue
        try: a=roc_auc_score(y[tr][m],v[m])
        except Exception: continue
        sc.append((abs(a-.5),c))
    sc.sort(reverse=True); return [c for _,c in sc[:k]] or MANUAL3

def obj(t):
    fam=t.suggest_categorical("family",["logit","logit_l1","xgb"])
    fs=t.suggest_categorical("fsel",["manual3","manual3+new","topk","all"])
    k=t.suggest_categorical("k",[3,5,8,12,20,40]) if fs=="topk" else 12
    C=t.suggest_float("C",1e-3,10,log=True) if fam!="xgb" else 1
    md=t.suggest_int("max_depth",1,5) if fam=="xgb" else 2
    ne=t.suggest_int("n_estimators",50,500,step=50) if fam=="xgb" else 200
    lr=t.suggest_float("lr",0.005,0.2,log=True) if fam=="xgb" else 0.03
    rl=t.suggest_float("reg_lambda",1e-2,100,log=True) if fam=="xgb" else 10
    pr=np.full(len(sidx),np.nan)
    for s in range(MIN_TRAIN,len(sidx),STEP):
        tr=sidx[:s]; te=sidx[s:min(s+STEP,len(sidx))]
        cols=pick(tr,fs,k)
        Xtr,Xte=X.loc[tr,cols],X.loc[te,cols]
        med=Xtr.median(); Xtr=Xtr.fillna(med).fillna(0); Xte=Xte.fillna(med).fillna(0)
        if fam=="xgb":
            m=xgb.XGBClassifier(max_depth=md,n_estimators=ne,learning_rate=lr,reg_lambda=rl,
                eval_metric="logloss",verbosity=0,random_state=0).fit(Xtr,y[tr])
            pr[s:s+len(te)]=m.predict_proba(Xte)[:,1]
        else:
            sc=StandardScaler().fit(Xtr)
            pen="l1" if fam=="logit_l1" else "l2"
            solver="liblinear" if pen=="l1" else "lbfgs"
            m=LogisticRegression(C=C,penalty=pen,solver=solver,max_iter=3000).fit(sc.transform(Xtr),y[tr])
            pr[s:s+len(te)]=m.predict_proba(sc.transform(Xte))[:,1]
    mm=~np.isnan(pr)
    return log_loss(y[sidx][mm],np.clip(pr[mm],1e-6,1-1e-6))

t0=time.time()
st=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=7))
st.optimize(obj,n_trials=150,show_progress_bar=False)
print(f"  {len(st.trials)} trials in {time.time()-t0:.0f}s")
L=pd.DataFrame([dict(fsel=t.params.get("fsel"),family=t.params.get("family"),ll=t.value)
                for t in st.trials if t.value is not None])
print("\n  best log-loss per feature strategy:")
print(L.groupby("fsel").ll.agg(["min","count"]).sort_values("min")
      .to_string(float_format=lambda x:f"{x:.5f}"))
print(f"\n  winner: {st.best_params}  (search log-loss {st.best_value:.5f})")
json.dump({"best":st.best_params,"ll":st.best_value,
           "by_fsel":L.groupby("fsel").ll.min().to_dict(),
           "wf_sets":res},
          open("data/backtest_2026/newfeat_search.json","w"),indent=1,default=str)
print("\nwrote data/backtest_2026/newfeat_search.json")
