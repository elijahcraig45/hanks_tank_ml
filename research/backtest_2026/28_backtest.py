"""Phase 4 gate: full 2026 game-level backtest of the PA simulator.

Weekly refit -- rates are trained only on PAs strictly before each week, so no game is
ever predicted by a model that saw it. 1000 episodes per game. Scored against V10 as
served, the 3-feature logistic, and always-home, on BOTH windows:

    search window   2026-04-07 -> 2026-08-07
    untouched hold  2026-08-08 -> 2026-09-07

A candidate must win in BOTH to count -- the gate every earlier mechanism failed.
"""
import sys, warnings, time, json
import numpy as np, pandas as pd
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from pa_sim.predict import SimEngine

HOLD_FROM = pd.Timestamp("2026-08-08")
N_EP = 1000
alpha = float(np.load("data/backtest_2026/pa_alpha.npy")[0])

pa = pd.read_parquet("data/backtest_2026/pa_2015_2026.parquet"); pa["ev"] = pa["ev"].astype(str)
hand = pa.groupby("pitcher", observed=True)["p_throws"].agg(lambda s: s.mode().iloc[0]).to_dict()
lu = pd.read_parquet("data/backtest_2026/lineups_2026.parquet")
ff = pd.read_parquet("data/backtest_2026/full_features_2026.parquet")
park = dict(zip(ff.game_pk, ff.home_park_factor.fillna(1.0)))
pred_v10 = pd.read_parquet("data/backtest_2026/games_2026_pregame.parquet")

# assemble one row per game
games = []
for gid, g in lu.groupby("game_pk", sort=False):
    h = g[g.team_type == "home"].sort_values("batting_order")
    a = g[g.team_type == "away"].sort_values("batting_order")
    if len(h) != 9 or len(a) != 9: continue
    r0 = g.iloc[0]
    if pd.isna(r0.home_starter_id) or pd.isna(r0.away_starter_id): continue
    if h.player_id.isna().any() or a.player_id.isna().any(): continue
    games.append(dict(game_pk=gid, game_date=r0.game_date,
                      home_order=[int(x) for x in h.player_id], away_order=[int(x) for x in a.player_id],
                      home_starter=int(r0.home_starter_id), away_starter=int(r0.away_starter_id),
                      home_team=r0.home_team_name, away_team=r0.away_team_name,
                      park=float(park.get(gid, 1.0)), home_win=int(r0.home_win)))
G = pd.DataFrame(games).sort_values(["game_date", "game_pk"]).reset_index(drop=True)
# statcast team abbreviations for the bullpen lookup
abbr = pa.groupby("home_team", observed=True).size().index.tolist()
name2abbr = {}
for tm in ff.itertuples():
    pass
sc = pa[["game_pk", "home_team", "away_team"]].drop_duplicates("game_pk")
G = G.merge(sc, on="game_pk", how="left", suffixes=("", "_ab"))
G["home_ab"] = G.home_team_ab if "home_team_ab" in G else G.home_team
G["away_ab"] = G.away_team_ab if "away_team_ab" in G else G.away_team
print(f"{len(G)} games assembled; bullpen team keys matched for "
      f"{G.home_ab.notna().mean()*100:.1f}%")

# ---- weekly refit + simulate ----
G["week"] = G.game_date.dt.to_period("W").dt.start_time
sim_p = np.full(len(G), np.nan)
eng = SimEngine(n_episodes=N_EP, alpha=alpha)
t0 = time.time()
for wk, idx in G.groupby("week").groups.items():
    idx = np.asarray(idx)
    try:
        eng.fit(pa, wk)
    except ValueError:
        continue
    for i in idx:
        r = G.loc[i]
        res = eng.predict_game(r.home_order, r.away_order, r.home_starter, r.away_starter,
                               r.home_ab, r.away_ab, park_factor=r.park,
                               home_starter_hand=hand.get(r.home_starter),
                               away_starter_hand=hand.get(r.away_starter),
                               n_episodes=N_EP, seed=int(r.game_pk) % 10_000)
        sim_p[i] = res["home_win_prob"]
print(f"simulated {np.isfinite(sim_p).sum()} games x {N_EP} episodes in {time.time()-t0:.0f}s")
G["sim"] = sim_p

# ---- comparison models ----
G = G.merge(pred_v10[["game_pk", "home_win_probability"]], on="game_pk", how="left")
F3 = ["elo_differential", "pythag_differential", "sp_quality_composite_diff"]
G = G.merge(ff[["game_pk"] + F3], on="game_pk", how="left")
y = G.home_win.values
X = G[F3].astype(float); X = X.fillna(X.median())
logit = np.full(len(G), np.nan)
for s in range(400, len(G), 100):
    tr, te = np.arange(0, s), np.arange(s, min(s+100, len(G)))
    scl = StandardScaler().fit(X.iloc[tr])
    m = LogisticRegression(C=0.1, max_iter=3000).fit(scl.transform(X.iloc[tr]), y[tr])
    logit[te] = m.predict_proba(scl.transform(X.iloc[te]))[:, 1]
G["logit3"] = logit

# calibrated simulator: Platt on the training window only
cal = np.full(len(G), np.nan)
lg = lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
for s in range(400, len(G), 100):
    tr, te = np.arange(0, s), np.arange(s, min(s+100, len(G)))
    ok = np.isfinite(G.sim.values[tr])
    if ok.sum() < 200: continue
    c = LogisticRegression(C=1e6, max_iter=3000).fit(lg(G.sim.values[tr][ok]).reshape(-1,1), y[tr][ok])
    cal[te] = c.predict_proba(lg(G.sim.values[te]).reshape(-1,1))[:,1]
G["sim_cal"] = cal

CAND = {"PA simulator (raw)": "sim", "PA simulator (calibrated)": "sim_cal",
        "3-feature logistic": "logit3", "V10 (production)": "home_win_probability"}
def block(mask, tag):
    print(f"\n{'='*84}\n{tag}  (n={int(mask.sum())})\n{'='*84}")
    print(f"{'model':32}{'n':>6}{'acc':>8}{'auc':>9}{'brier':>9}{'logloss':>10}")
    out = {}
    for nm, col in CAND.items():
        p = G[col].values
        m = mask & np.isfinite(p)
        pp = np.clip(p[m].astype(float),1e-6,1-1e-6); t = y[m]
        out[nm] = dict(acc=((pp>=.5).astype(int)==t).mean()*100, auc=roc_auc_score(t,pp),
                       brier=brier_score_loss(t,pp), ll=log_loss(t,pp), n=int(m.sum()))
        o=out[nm]; print(f"{nm:32}{o['n']:>6}{o['acc']:>8.2f}{o['auc']:>9.4f}{o['brier']:>9.4f}{o['ll']:>10.5f}")
    b = y[mask].mean()
    print(f"{'   always home':32}{int(mask.sum()):>6}{max(b,1-b)*100:>8.2f}{0.5:>9.4f}"
          f"{np.mean((b-y[mask])**2):>9.4f}{log_loss(y[mask],np.full(int(mask.sum()),b)):>10.5f}")
    return out

srch = (G.game_date < HOLD_FROM).values & np.isfinite(G.sim.values)
hold = (G.game_date >= HOLD_FROM).values & np.isfinite(G.sim.values)
rs = block(srch, "SEARCH WINDOW  2026-04-07 -> 2026-08-07")
rh = block(hold, "UNTOUCHED HOLDOUT  2026-08-08 -> 2026-09-07")

print(f"\n{'='*84}\nPHASE 4 GATE -- log-loss gain vs V10, required in BOTH windows\n{'='*84}")
print(f"{'model':32}{'search':>11}{'holdout':>11}   verdict")
for nm in CAND:
    if nm.startswith("V10"): continue
    gs = rs["V10 (production)"]["ll"] - rs[nm]["ll"]
    gh = rh["V10 (production)"]["ll"] - rh[nm]["ll"]
    v = "BETTER IN BOTH" if (gs>0 and gh>0) else ("worse in both" if (gs<0 and gh<0) else "inconsistent")
    print(f"{nm:32}{gs:>+11.5f}{gh:>+11.5f}   {v}")
G[["game_pk","game_date","home_win","sim","sim_cal","logit3","home_win_probability"]].to_parquet(
    "data/backtest_2026/sim_backtest.parquet", index=False)
json.dump({"search":{k:{kk:vv for kk,vv in v.items()} for k,v in rs.items()},
           "holdout":{k:{kk:vv for kk,vv in v.items()} for k,v in rh.items()},
           "alpha":alpha,"n_episodes":N_EP},
          open("data/backtest_2026/sim_backtest.json","w"), indent=1)
print("\nwrote data/backtest_2026/sim_backtest.{parquet,json}")
