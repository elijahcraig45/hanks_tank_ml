"""Does fixing the V10 feature-builder bugs change which model should serve?

Honest-data protocol (see memory notes game-predictions-backfill-contamination,
mlb-autosearch-exhausted-and-sample-cap):

  * 2026 features come from game_v10_features rows computed BEFORE first pitch
    (data/backtest_2026/full_features_2026.parquet, 1,743 regular-season games).
  * "V10 as served" is the genuine pregame prediction row for each game
    (games_2026_pregame.parquet, predicted_at < game_time_utc).
  * Season-trained candidates are scored walk-forward: expanding window, first fit on
    500 games, refit every 100. Nothing is fit on a game it is scored on.
  * Reported twice: the whole walk-forward span, and the last 400 games alone.
  * 95% CIs from 2,000 game-level bootstrap resamples; paired deltas vs V10-as-served.

Inputs that need BigQuery were pulled read-only into the scratch dir first
(team_statcast_asof.parquet, games_2026_R.parquet) -- see the sibling scripts.

Also needs <scratch_dir>/v10_prod.pkl: a read-only copy of
gs://hanks_tank_data/models/vertex/game_outcome_2026_v10/model.pkl.

Usage: python research/v10_fixes/eval_v10_fixes.py <scratch_dir>
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRATCH = Path(sys.argv[1])
DATA = Path("/Users/VTNX82W/Documents/personalDev/mlb/hanks_tank_ml/data")
BT = DATA / "backtest_2026"
MIN_TRAIN, STEP, HOLDOUT_N, B = 500, 100, 400, 2000
THRESH = 0.64
rng = np.random.default_rng(0)
import pyarrow.parquet as pq


def rp(path):
    """read_parquet that tolerates BigQuery's db-dtypes metadata without the package"""
    return pq.read_table(path).to_pandas(ignore_metadata=True, date_as_object=True)

# ----------------------------------------------------------------------------- data
IN = SCRATCH / "fresh"          # written by fetch_inputs.py
d = rp(IN / "full_features_2026.parquet")
d["game_date"] = pd.to_datetime(d["game_date"])
d = d.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
y = d["home_win"].to_numpy().astype(int)
n = len(d)

served = rp(IN / "served_2026.parquet")[["game_pk", "home_win_probability", "model_version"]]
d = d.merge(served.rename(columns={"home_win_probability": "p_served"}), on="game_pk", how="left")
d = d[d.p_served.notna()].reset_index(drop=True)
y = d["home_win"].to_numpy().astype(int)
n = len(d)

# true season-to-date games played (regular season only, deduped game_pk)
g = rp(IN / "games_2026_R.parquet")
g["game_date"] = pd.to_datetime(g["game_date"])
long = pd.concat([g[["game_date", "home_team_id"]].rename(columns={"home_team_id": "t"}),
                  g[["game_date", "away_team_id"]].rename(columns={"away_team_id": "t"})])
def games_before(team, date):
    s = long.loc[long.t == team, "game_date"]
    return int((s < date).sum())
cache = {}
gp = []
for t_h, t_a, dt in zip(d.home_team_id, d.away_team_id, d.game_date):
    for t in (t_h, t_a):
        if (t, dt) not in cache:
            cache[(t, dt)] = games_before(t, dt)
    gp.append((cache[(t_h, dt)] + cache[(t_a, dt)]) / 2)
d["season_game_number_fixed"] = np.array(gp)
d["season_pct_fixed"] = np.clip(d.season_game_number_fixed / 162.0, 0, 1)

# real team Statcast metrics, season-to-date strictly before the game date
ts = rp(IN / "team_statcast_asof.parquet")
ts["game_date"] = pd.to_datetime(ts["game_date"])
ts["woba"] = ts.woba_num / ts.woba_den
ts["ev"] = ts.ev_sum / ts.bbe_n
ts["hh"] = ts.hh_n / ts.bbe_n
ts["brl"] = ts.brl_n / ts.bbe_n
ts["whiff"] = ts.whiffs / ts.swings
ts["fbv"] = ts.fbv_sum / ts.fbv_n
ip = ts.outs / 3.0
lg = ts.groupby("game_date")[["hr", "fb", "k", "bb", "hbp", "outs"]].transform("sum")
lg_hrfb = lg.hr / lg.fb
# xFIP = (13*FB*lgHR/FB + 3*(BB+HBP) - 2*K)/IP + constant (constant cancels in diffs)
ts["xfip"] = (13 * ts.fb * lg_hrfb + 3 * (ts.bb + ts.hbp) - 2 * ts.k) / ip + 3.1
for m in ["woba", "ev", "hh", "brl", "whiff", "fbv", "xfip"]:
    ts.loc[ts.woba_den < 300, m] = np.nan  # too few PAs early season
wide = ts.pivot_table(index="game_pk", columns="side",
                      values=["woba", "ev", "hh", "brl", "whiff", "fbv", "xfip"])
wide.columns = [f"{s}_{m}_sc" for m, s in wide.columns]
d = d.merge(wide.reset_index(), on="game_pk", how="left")
for m in ["woba", "ev", "hh", "brl", "whiff", "fbv"]:
    d[f"{m}_sc_diff"] = d[f"home_{m}_sc"] - d[f"away_{m}_sc"]
d["xfip_sc_diff"] = d["away_xfip_sc"] - d["home_xfip_sc"]  # + = home pitching better

# ----------------------------------------------------------------------------- model zoo
prod = pickle.load(open(SCRATCH / "v10_prod.pkl", "rb"))
PF, FILL, M = prod["features"], prod["fill_values"], prod["model"]

def prod_matrix(frame, fixed):
    X = pd.DataFrame({f: frame[f].astype(float) if f in frame else FILL[f] for f in PF})
    if fixed:
        X["season_pct_complete"] = frame["season_pct_fixed"].to_numpy()
        # training encodes day_of_week 1..7 with Sunday = 1; the live builder wrote
        # pandas dayofweek 0..6 with Monday = 0
        X["day_of_week"] = ((pd.to_datetime(frame["game_date"]).dt.dayofweek + 1) % 7) + 1
        # training: home - away; live V8 builder wrote away - home
        X["luck_differential"] = frame["home_luck_factor"] - frame["away_luck_factor"]
    return X.fillna(pd.Series(FILL))

p = {}
p["V10 as served"] = d["p_served"].to_numpy(dtype=float)
p["V10 re-scored (stored inputs)"] = M.predict_proba(prod_matrix(d, False))[:, 1]
p["V10 fixed inputs (no retrain)"] = M.predict_proba(prod_matrix(d, True))[:, 1]

# V10 retrained offline with the fixes, same window + params as train_v10_models.py
tr = pd.read_parquet(DATA / "training" / "train_v8_2015_2024.parquet")
mh = pq.read_table(DATA / "training" / "matchup_features_historical.parquet").to_pandas(
    ignore_metadata=True, date_as_object=True)
mcols = [c for c in PF if c in mh.columns]
tr = tr.merge(mh[["game_pk"] + mcols].drop_duplicates("game_pk"), on="game_pk", how="left")
tr["lineup_confirmed"] = 1
tr["lineup_woba_differential"] = tr.home_lineup_woba_vs_hand - tr.away_lineup_woba_vs_hand
tr["lineup_k_pct_differential"] = tr.away_lineup_k_pct_vs_hand - tr.home_lineup_k_pct_vs_hand
tr["h2h_woba_differential"] = tr.home_h2h_woba - tr.away_h2h_woba
dead = [f for f in PF if f in tr and tr[f].nunique() <= 1]
RF = [f for f in PF if f in tr and f not in dead]
Xtr = tr[RF].astype(float)
med = Xtr.median()
rt = xgb.XGBClassifier(n_estimators=450, max_depth=4, learning_rate=0.035, subsample=0.82,
                       colsample_bytree=0.82, min_child_weight=4, reg_alpha=0.05,
                       reg_lambda=1.5, random_state=42, eval_metric="logloss", verbosity=0)
rt.fit(Xtr.fillna(med), tr["home_won"].astype(int))
Xl = prod_matrix(d, True)[RF]
p["V10 retrained 2015-24, fixed"] = rt.predict_proba(Xl.fillna(med))[:, 1]

# season-trained, walk-forward candidates
def walkforward(Xf, make, scale):
    out = np.full(n, np.nan)
    for s in range(MIN_TRAIN, n, STEP):
        trn, te = np.arange(s), np.arange(s, min(s + STEP, n))
        A, Bt = Xf.iloc[trn], Xf.iloc[te]
        mdn = A.median()
        A, Bt = A.fillna(mdn).fillna(0), Bt.fillna(mdn).fillna(0)
        mdl = make()
        if scale:
            sc = StandardScaler().fit(A)
            A, Bt = sc.transform(A), sc.transform(Bt)
        mdl.fit(A, y[trn])
        out[te] = mdl.predict_proba(Bt)[:, 1]
    return out

d["luck_differential_fixed"] = d.home_luck_factor - d.away_luck_factor
FIX = {"luck_differential": "luck_differential_fixed"}
diff_cols = [c for c in d.columns
             if ("diff" in c or "advantage" in c) and not c.endswith("_fixed")
             and c not in ("home_win",) and d[c].dtype.kind in "fi"]
diff_cols = [FIX.get(c, c) for c in diff_cols] + ["home_park_factor", "season_pct_fixed"]
# drop placeholders/aliases and anything constant, then greedy-dedupe |r| > 0.95
bad = {"fg_xfip_differential", "fg_woba_differential"}  # ERA / OBP aliases
cand = [c for c in dict.fromkeys(diff_cols) if c not in bad and d[c].nunique() > 1]
# greedy: keep a column only if it is not near-duplicate (|r| > 0.95) of a kept one
# AND it raises the matrix rank (catches exact linear combos like a - b)
keep = []
Z = d[cand].astype(float)
Z = (Z.fillna(Z.median()) - Z.mean()) / Z.std()
C = Z.corr().abs()
for c in cand:
    if any(C.loc[c, k] > 0.95 for k in keep):
        continue
    if np.linalg.matrix_rank(Z[keep + [c]].values, tol=1e-6) == len(keep) + 1:
        keep.append(c)
Xd = d[keep].astype(float)
rank = np.linalg.matrix_rank(Z[keep].values, tol=1e-6)
V10P = dict(n_estimators=450, max_depth=4, learning_rate=0.035, subsample=0.82,
            colsample_bytree=0.82, min_child_weight=4, reg_alpha=0.05, reg_lambda=1.5,
            random_state=42, eval_metric="logloss", verbosity=0)
p["V10 params, season-fit, deduped diffs"] = walkforward(
    Xd, lambda: xgb.XGBClassifier(**V10P), False)
M3 = ["elo_differential", "pythag_differential", "sp_quality_composite_diff"]
p["3-feat L1 logistic"] = walkforward(
    d[M3].astype(float),
    lambda: LogisticRegression(C=0.5568, penalty="l1", solver="liblinear", max_iter=3000), True)
base = np.full(n, np.nan)
for s in range(MIN_TRAIN, n, STEP):
    base[s:min(s + STEP, n)] = y[:s].mean()
p["always home (base rate)"] = base

# does real team Statcast data add anything on top of the 3-feature model?
SC = ["woba_sc_diff", "ev_sc_diff", "hh_sc_diff", "brl_sc_diff", "whiff_sc_diff",
      "fbv_sc_diff", "xfip_sc_diff"]
p["3-feat + real team Statcast (7)"] = walkforward(
    d[M3 + SC].astype(float),
    lambda: LogisticRegression(C=0.5568, penalty="l1", solver="liblinear", max_iter=3000), True)

# ----------------------------------------------------------------------------- scoring
def ll_vec(pp, t):
    pp = np.clip(pp, 1e-6, 1 - 1e-6)
    return -(t * np.log(pp) + (1 - t) * np.log(1 - pp))

def ci(stat_fn, idx):
    bs = [stat_fn(rng.choice(idx, len(idx), replace=True)) for _ in range(B)]
    return np.nanpercentile(bs, [2.5, 97.5])

def score(idx, label):
    rows = []
    ref = p["V10 as served"]
    for name, pp in p.items():
        m = idx[~np.isnan(pp[idx]) & ~np.isnan(ref[idx])]
        L = ll_vec(pp[m], y[m]); A = ((pp[m] >= .5) == y[m]).astype(float)
        conf = np.maximum(pp[m], 1 - pp[m]) >= THRESH
        dl = L - ll_vec(ref[m], y[m])
        ll_ci = ci(lambda b: L[np.searchsorted(m, b)].mean(), m)
        acc_ci = ci(lambda b: A[np.searchsorted(m, b)].mean(), m)
        d_ci = ci(lambda b: dl[np.searchsorted(m, b)].mean(), m)
        if conf.sum() >= 5:
            cm = m[conf]; Ac = A[conf]
            c_ci = ci(lambda b: Ac[np.searchsorted(cm, b)].mean(), cm)
            c_acc = f"{Ac.mean()*100:.1f} [{c_ci[0]*100:.1f}, {c_ci[1]*100:.1f}]"
        else:
            c_acc = "-"
        rows.append(dict(model=name, n=len(m), logloss=f"{L.mean():.4f} [{ll_ci[0]:.4f}, {ll_ci[1]:.4f}]",
                         d_ll_vs_served=f"{dl.mean():+.4f} [{d_ci[0]:+.4f}, {d_ci[1]:+.4f}]",
                         acc=f"{A.mean()*100:.1f} [{acc_ci[0]*100:.1f}, {acc_ci[1]*100:.1f}]",
                         n_ge64=int(conf.sum()), acc_ge64=c_acc))
    R = pd.DataFrame(rows)
    print(f"\n=== {label}: {len(idx)} games, {d.game_date[idx[0]].date()} -> {d.game_date[idx[-1]].date()} ===")
    print(R.to_string(index=False))
    return R

def sharp_end(idx, label):
    """Equal-count sharp end: each model's N most confident picks, N = V10's count >= .64."""
    ref = p["V10 as served"]
    N = int((np.maximum(ref[idx], 1 - ref[idx]) >= THRESH).sum())
    print(f"\n--- sharp end, {label}: each model's top {N} most-confident picks ---")
    out = {}
    for name, pp in p.items():
        if name.startswith("always"):
            continue
        m = idx[~np.isnan(pp[idx])]
        top = m[np.argsort(-np.abs(pp[m] - .5))[:N]]
        A = ((pp[top] >= .5) == y[top]).astype(float)
        bs = [A[rng.integers(0, len(A), len(A))].mean() for _ in range(B)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[name] = (A.mean(), lo, hi)
        print(f"  {name:40s} {A.mean()*100:5.1f}%  [{lo*100:.1f}, {hi*100:.1f}]  (min conf {np.abs(pp[top]-.5).min()+.5:.3f})")
    return N, out

WF = np.arange(MIN_TRAIN, n)
HO = np.arange(n - HOLDOUT_N, n)
FRESH = np.where(d.game_date > "2026-09-07")[0]   # after every earlier study's data
print("served model_version in walk-forward span:",
      d.loc[WF, "model_version"].value_counts().to_dict())
print(f"games {n}; walk-forward span {len(WF)}; holdout {len(HO)}")
print(f"deduped differential set: {len(keep)} features, rank {rank}: {keep}")
print(f"retrain: {len(RF)} features, dropped dead-in-training: {dead}")
print("corr(served, re-scored stored inputs) =",
      round(np.corrcoef(p['V10 as served'], p['V10 re-scored (stored inputs)'])[0, 1], 4))
R1 = score(WF, "WALK-FORWARD")
R2 = score(HO, "HOLDOUT (last 400)")
R3 = score(FRESH, "FRESH (after 2026-09-07, untouched by earlier studies)")
S1 = sharp_end(WF, "walk-forward")
S2 = sharp_end(HO, "holdout")
S3 = sharp_end(FRESH, "fresh")
# single-feature AUC of the real Statcast team diffs, for the record
from sklearn.metrics import roc_auc_score
print("\nsingle-feature AUC (all 1,743 games, NaN dropped):")
for c in SC + ["season_pct_fixed", "fg_xfip_differential", "fg_woba_differential", "pythag_differential"]:
    v = d[c].to_numpy(dtype=float); mm = ~np.isnan(v)
    print(f"  {c:28s} {roc_auc_score(y[mm], v[mm]):.4f}  (n={mm.sum()})")
pd.concat([R1.assign(split="walkforward"), R2.assign(split="holdout"), R3.assign(split="fresh")]).to_csv(
    SCRATCH / "eval_results.csv", index=False)
json.dump({"sharp_wf": {"N": S1[0], **{k: list(v) for k, v in S1[1].items()}},
           "sharp_ho": {"N": S2[0], **{k: list(v) for k, v in S2[1].items()}},
           "sharp_fresh": {"N": S3[0], **{k: list(v) for k, v in S3[1].items()}},
           "dedup_features": keep, "dedup_rank": int(rank)},
          open(SCRATCH / "eval_sharp.json", "w"), indent=1, default=float)
pd.DataFrame({k: v for k, v in p.items()}).assign(game_pk=d.game_pk, y=y, game_date=d.game_date).to_parquet(
    SCRATCH / "eval_preds.parquet")
