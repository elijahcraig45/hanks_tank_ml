"""Phase 2 gate: calibrate the run environment, then check predicted vs actual runs.

Validated on 2025 (held out of rate training, which stops at 2024) using lineups
reconstructed from what actually happened -- this isolates the run model from the
lineup-forecasting problem. ~4,800 team-game run totals, far more power than wins.
"""
import sys, warnings, time
import numpy as np, pandas as pd
from scipy.optimize import brentq
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from pa_sim.predict import SimEngine
from pa_sim.markov import game_runs
from pa_sim.game import expected_runs, apply_scale
from pa_sim.rates import log5
from google.cloud import bigquery

pa = pd.read_parquet("data/backtest_2026/pa_2015_2026.parquet")
pa["ev"] = pa["ev"].astype(str)
eng = SimEngine(n_episodes=1000).fit(pa, pd.Timestamp("2025-01-01"))
print(f"rates fitted through 2024 ({(pa.game_date < pd.Timestamp('2025-01-01')).sum():,} PAs)")

# ---- observed 2025 run environment ----
bq = bigquery.Client(project="hankstank")
gh = bq.query("""SELECT game_pk, home_score, away_score
                 FROM `hankstank.mlb_historical_data.games_historical`
                 WHERE game_type='R' AND EXTRACT(YEAR FROM game_date)=2025
                   AND home_score IS NOT NULL""").to_dataframe()
target = float(np.concatenate([gh.home_score, gh.away_score]).mean())
print(f"2025 observed: {len(gh)} games, {target:.3f} runs/team/game")

# ---- solve alpha analytically on a league-average matchup ----
L = eng.league
def analytic_runs(a):
    order = apply_scale(log5(np.tile(L, (9, 1)), L[None, :], L[None, :]), a)
    return expected_runs(game_runs([order] * 9))
alpha = float(brentq(lambda a: analytic_runs(a) - target, 0.8, 2.0, xtol=1e-4))
eng.alpha = alpha
print(f"solved alpha = {alpha:.4f}  (analytic league-average side -> {analytic_runs(alpha):.3f} runs)")

# ---- reconstruct 2025 lineups from actual PA order ----
p25 = pa[pa.game_year == 2025]
rows = []
for gid, g in p25.groupby("game_pk", sort=False):
    rec = {}
    ok = True
    for is_home in (True, False):
        sub = g[g.home_batting == is_home]
        order = sub.batter.drop_duplicates().tolist()[:9]
        st = sub[sub.pitcher_is_starter].pitcher
        if len(order) < 9 or st.empty:
            ok = False; break
        tag = "home" if is_home else "away"
        rec[f"{tag}_order"] = order
        rec[f"opp_starter_for_{tag}"] = int(st.mode().iloc[0])
        rec[f"{tag}_team"] = sub.iloc[0].batting_team
    if ok:
        rec["game_pk"] = gid
        rows.append(rec)
G = pd.DataFrame(rows).merge(gh, on="game_pk", how="inner")
print(f"reconstructed {len(G)} 2025 games with full lineups")

# ---- simulate ----
t0 = time.time()
ph, pa_ = [], []
for r in G.itertuples():
    res = eng.predict_game(r.home_order, r.away_order,
                           r.opp_starter_for_away,   # pitcher the AWAY side faced = home starter
                           r.opp_starter_for_home,
                           r.home_team, r.away_team, park_factor=1.0, n_episodes=400)
    ph.append(res["mean_home"]); pa_.append(res["mean_away"])
ph, pa_ = np.array(ph), np.array(pa_)
print(f"simulated {len(G)} games x 400 episodes in {time.time()-t0:.0f}s")

pred = np.concatenate([ph, pa_])
act = np.concatenate([G.home_score.values, G.away_score.values]).astype(float)
print(f"\n{'='*76}\nPHASE 2 GATE -- predicted vs actual team runs (n={len(pred)})\n{'='*76}")
print(f"  mean predicted {pred.mean():.3f}   mean actual {act.mean():.3f}   bias {pred.mean()-act.mean():+.3f}")
print(f"  sd predicted   {pred.std():.3f}   sd actual   {act.std():.3f}")
r = float(np.corrcoef(pred, act)[0, 1])
print(f"  correlation r = {r:.4f}")
mae, mae0 = np.abs(pred-act).mean(), np.abs(act.mean()-act).mean()
print(f"  MAE {mae:.3f}  vs {mae0:.3f} predicting the mean   ({(mae0-mae)/mae0*100:+.2f}%)")
# does the run model separate games at all?
q = pd.qcut(pred, 5, labels=False, duplicates="drop")
print("\n  actual runs by predicted quintile (monotone = the model separates games):")
for i in range(q.max()+1):
    m = q == i
    print(f"    Q{i+1}: predicted {pred[m].mean():.2f}  actual {act[m].mean():.2f}  n={m.sum()}")
ok = abs(pred.mean()-act.mean()) < 0.25 and r > 0.08
print(f"\n  -> PHASE 2 {'PASS' if ok else 'FAIL'}")
np.save("data/backtest_2026/pa_alpha.npy", np.array([alpha]))
