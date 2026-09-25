"""Autonomous model search over the full 167-feature 2026 set.

Protocol (the point of the whole script -- an unguarded search on 1.7k games with
~0.005 nats of signal WILL find spurious winners):

  FINAL HOLDOUT  = last 400 games. Untouched until one single evaluation at the end.
  SEARCH SET     = everything before it.
    Optuna samples {model family, hyperparameters, feature-selection strategy, k}
    and each trial is scored by expanding-window walk-forward log-loss INSIDE the
    search set. Feature selection runs inside each training fold, never globally.

  The gap between the search's best internal score and its holdout score is the
  overfitting measure, and is reported as a headline number.
"""
import warnings, json, sys, time
import numpy as np, pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import xgboost as xgb, lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

HOLDOUT_N, MIN_TRAIN, STEP, N_TRIALS = 400, 500, 100, 300

d = pd.read_parquet("data/backtest_2026/full_features_2026.parquet")
d = d.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
y = d.home_win.values
EXCL = {"game_pk","game_date","home_team_id","away_team_id","home_score","away_score",
        "home_win","computed_at","rn"}
FEATS = [c for c in d.columns if c not in EXCL and d[c].dtype.kind in "fiub"]
X = d[FEATS].astype(float)
X = X.replace([np.inf,-np.inf], np.nan)
print(f"data {len(d)} games x {len(FEATS)} features")

n = len(d); split = n - HOLDOUT_N
SEARCH = np.arange(split); HOLD = np.arange(split, n)
print(f"search set: {len(SEARCH)} games ({d.game_date[0].date()} -> {d.game_date[split-1].date()})")
print(f"final holdout: {len(HOLD)} games ({d.game_date[split].date()} -> {d.game_date[n-1].date()})")
print(f"holdout home rate {y[HOLD].mean():.4f}\n")

MANUAL3 = ["elo_differential","pythag_differential","sp_quality_composite_diff"]

def pick_features(tr, strategy, k):
    if strategy == "all":     return FEATS
    if strategy == "manual3": return [f for f in MANUAL3 if f in FEATS]
    # top-k by single-feature AUC computed on the TRAIN FOLD ONLY
    sc = []
    yt = y[tr]
    for c in FEATS:
        v = X[c].values[tr]
        m = ~np.isnan(v)
        if m.sum() < 100 or np.unique(v[m]).size < 3: continue
        try: a = roc_auc_score(yt[m], v[m])
        except Exception: continue
        sc.append((abs(a-0.5), c))
    sc.sort(reverse=True)
    return [c for _, c in sc[:k]] or [f for f in MANUAL3 if f in FEATS]

def build(p, seed=0):
    fam = p["family"]
    if fam == "logit":
        return ("scale", LogisticRegression(C=p["C"], penalty="l2", max_iter=3000))
    if fam == "logit_l1":
        return ("scale", LogisticRegression(C=p["C"], penalty="l1", solver="liblinear", max_iter=3000))
    if fam == "xgb":
        return ("raw", xgb.XGBClassifier(
            max_depth=p["max_depth"], n_estimators=p["n_estimators"],
            learning_rate=p["lr"], subsample=p["subsample"],
            colsample_bytree=p["colsample"], reg_lambda=p["reg_lambda"],
            reg_alpha=p["reg_alpha"], min_child_weight=p["min_child_weight"],
            eval_metric="logloss", verbosity=0, random_state=seed, n_jobs=2))
    if fam == "lgb":
        return ("raw", lgb.LGBMClassifier(
            num_leaves=p["num_leaves"], n_estimators=p["n_estimators"],
            learning_rate=p["lr"], min_child_samples=p["min_child_samples"],
            reg_lambda=p["reg_lambda"], subsample=p["subsample"],
            colsample_bytree=p["colsample"], verbose=-1, random_state=seed, n_jobs=2))
    if fam == "extratrees":
        return ("impute", ExtraTreesClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], random_state=seed, n_jobs=2))
    if fam == "rf":
        return ("impute", RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], random_state=seed, n_jobs=2))
    if fam == "mlp":
        return ("scale", MLPClassifier(hidden_layer_sizes=p["hidden"], alpha=p["alpha"],
            learning_rate_init=p["lr"], max_iter=600, random_state=seed))
    raise ValueError(fam)

def fit_predict(p, tr, te, seed=0):
    cols = pick_features(tr, p["fsel"], p.get("k", 12))
    Xtr, Xte = X.loc[tr, cols], X.loc[te, cols]
    med = Xtr.median()
    Xtr = Xtr.fillna(med).fillna(0); Xte = Xte.fillna(med).fillna(0)
    prep, mdl = build(p, seed)
    if prep == "scale":
        sc = StandardScaler().fit(Xtr)
        mdl.fit(sc.transform(Xtr), y[tr]); return mdl.predict_proba(sc.transform(Xte))[:,1]
    mdl.fit(Xtr, y[tr]); return mdl.predict_proba(Xte)[:,1]

def walkforward(p, idx, seed=0):
    """expanding-window walk-forward log-loss within idx"""
    preds = np.full(len(idx), np.nan)
    for s in range(MIN_TRAIN, len(idx), STEP):
        tr = idx[:s]; te = idx[s:min(s+STEP, len(idx))]
        preds[s:s+len(te)] = fit_predict(p, tr, te, seed)
    m = ~np.isnan(preds)
    if m.sum() < 100: return np.nan, preds, m
    return log_loss(y[idx][m], np.clip(preds[m],1e-6,1-1e-6)), preds, m

def sample(t):
    fam = t.suggest_categorical("family", ["logit","logit_l1","xgb","lgb","extratrees","rf","mlp"])
    p = {"family": fam}
    p["fsel"] = t.suggest_categorical("fsel", ["all","manual3","topk"])
    if p["fsel"] == "topk":
        p["k"] = t.suggest_categorical("k", [3,5,8,12,20,40,80])
    if fam in ("logit","logit_l1"):
        p["C"] = t.suggest_float("C", 1e-3, 10, log=True)
    elif fam == "xgb":
        p.update(max_depth=t.suggest_int("max_depth",1,6),
                 n_estimators=t.suggest_int("n_estimators",50,600,step=50),
                 lr=t.suggest_float("lr",0.005,0.3,log=True),
                 subsample=t.suggest_float("subsample",0.5,1.0),
                 colsample=t.suggest_float("colsample",0.3,1.0),
                 reg_lambda=t.suggest_float("reg_lambda",1e-2,100,log=True),
                 reg_alpha=t.suggest_float("reg_alpha",1e-3,10,log=True),
                 min_child_weight=t.suggest_int("min_child_weight",1,60))
    elif fam == "lgb":
        p.update(num_leaves=t.suggest_int("num_leaves",2,64),
                 n_estimators=t.suggest_int("n_estimators",50,600,step=50),
                 lr=t.suggest_float("lr",0.005,0.3,log=True),
                 min_child_samples=t.suggest_int("min_child_samples",5,120),
                 reg_lambda=t.suggest_float("reg_lambda",1e-2,100,log=True),
                 subsample=t.suggest_float("subsample",0.5,1.0),
                 colsample=t.suggest_float("colsample",0.3,1.0))
    elif fam in ("extratrees","rf"):
        p.update(n_estimators=t.suggest_int("n_estimators",100,600,step=100),
                 max_depth=t.suggest_int("max_depth",2,16),
                 min_samples_leaf=t.suggest_int("min_samples_leaf",1,60))
    elif fam == "mlp":
        p.update(hidden=t.suggest_categorical("hidden",[(16,),(32,),(64,32),(128,64,32)]),
                 alpha=t.suggest_float("alpha",1e-5,1.0,log=True),
                 lr=t.suggest_float("lr",1e-4,1e-2,log=True))
    return p

t0 = time.time()
def objective(t):
    p = sample(t)
    ll, _, _ = walkforward(p, SEARCH)
    if not np.isfinite(ll): raise optuna.TrialPruned()
    return ll

study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"search done: {len(study.trials)} trials in {time.time()-t0:.0f}s")

# leaderboard by family
recs = [dict(family=t.params.get("family"), fsel=t.params.get("fsel"),
             k=t.params.get("k"), ll=t.value) for t in study.trials
        if t.value is not None and np.isfinite(t.value)]
L = pd.DataFrame(recs)
print("\nbest walk-forward log-loss per family (inside search set):")
print(L.groupby("family").ll.agg(["min","median","count"]).sort_values("min")
      .to_string(float_format=lambda x: f"{x:.5f}"))
print("\nbest per feature-selection strategy:")
print(L.groupby("fsel").ll.agg(["min","median","count"]).sort_values("min")
      .to_string(float_format=lambda x: f"{x:.5f}"))

best_p = sample(optuna.trial.FixedTrial(study.best_params))
print(f"\nBEST CONFIG (search score {study.best_value:.5f}):")
print("  " + json.dumps({k: (list(v) if isinstance(v,tuple) else v)
                         for k,v in best_p.items()}))
json.dump({"best_params": study.best_params, "search_ll": study.best_value,
           "n_trials": len(study.trials),
           "leaderboard": L.groupby("family").ll.min().to_dict()},
          open("data/backtest_2026/autosearch_best.json","w"), indent=1)
np.save("data/backtest_2026/autosearch_split.npy", np.array([split]))
print("\nwrote data/backtest_2026/autosearch_best.json")
