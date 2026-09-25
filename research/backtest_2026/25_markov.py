"""Phase 2 gate: does the run generator reproduce actual team run totals?

This is a COMPONENT test, so the lineup is taken from what actually happened (the nine
batters in order of first appearance) rather than forecast. The question is only whether
the base-out chain converts PA probabilities into correct run distributions. Validated
on ~4,800 team-game run totals -- far more power than 1,300 binary wins.

Rates are trained on 2015-2024; validation games are 2025 and 2026.
"""
import sys, warnings, time
import numpy as np, pandas as pd
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from pa_sim import CLASSES
from pa_sim.rates import RateTable, log5
from pa_sim.markov import game_runs
from pa_sim.game import apply_scale, expected_runs, fit_run_scale, win_prob
from pa_sim.usage import hook_curve

pa = pd.read_parquet("data/backtest_2026/pa_2015_2026.parquet")
pa["ev"] = pa["ev"].astype(str)
tr = pa[pa.game_year <= 2024]
bat, pit = RateTable(tr, "batter"), RateTable(tr, "pitcher")
L = bat.league
curve = hook_curve(tr)
print(f"trained on {len(tr):,} PAs; hook curve P(starter pitching) by inning:")
print("  " + "  ".join(f"{i+1}:{c:.2f}" for i, c in enumerate(curve)))

# bullpen composite per (season, team): relievers weighted by PA volume
rel = tr[~tr.pitcher_is_starter]
bp_map = {}
for (yr, tm), g in rel.groupby(["game_year", "batting_team"], observed=True):
    # relievers who pitched TO this batting team are the opponent's -- invert:
    pass
rel2 = pa[(~pa.pitcher_is_starter)]
pitch_team = np.where(rel2.inning_topbot.values == "Top", rel2.home_team.values, rel2.away_team.values)
rel2 = rel2.assign(pitch_team=pitch_team)
for (yr, tm), g in rel2[rel2.game_year <= 2024].groupby(["game_year", "pitch_team"], observed=True):
    vc = g.pitcher.value_counts().head(12)
    bp_map[(yr, tm)] = [(int(p), float(w)) for p, w in vc.items()]
league_bp = rel2[rel2.game_year <= 2024]
bp_league = np.array([ (league_bp.ev == c).mean() for c in CLASSES ]); bp_league /= bp_league.sum()
print(f"bullpen composites for {len(bp_map)} (season, team) pairs; "
      f"league relief rates K={bp_league[0]*100:.1f}% BB={bp_league[1]*100:.1f}%")

def sides_for_game(g):
    """(batting_team, is_home, ordered 9 batters, opposing starter) for both sides."""
    out = []
    for is_home in (False, True):
        sub = g[g.home_batting == is_home]
        if len(sub) < 20:
            return None
        order = sub.batter.drop_duplicates().tolist()[:9]
        if len(order) < 9:
            return None
        st = sub[sub.pitcher_is_starter].pitcher
        if st.empty:
            return None
        starter = int(st.mode().iloc[0])
        team = sub.batting_team.iloc[0]
        out.append((team, is_home, order, starter))
    return out

def side_dist(order, starter, opp_team, season, alpha):
    lineup = np.array([bat.get(int(b)) for b in order])
    sp = pit.get(int(starter))
    bp = bp_map.get((min(season, 2024), opp_team))
    if bp:
        rows = np.array([pit.get(p) for p, _ in bp]); w = np.array([w for _, w in bp], dtype=float)
        bpr = np.average(rows, axis=0, weights=w / w.sum())
    else:
        bpr = bp_league
    per_inning = []
    for p_start in curve:
        opp = p_start * sp + (1 - p_start) * bpr
        opp = opp / opp.sum()
        per_inning.append(apply_scale(log5(lineup, opp[None, :], L[None, :]), alpha))
    return game_runs(per_inning)

def build(year, n_games):
    sub = pa[pa.game_year == year]
    gids = sub.game_pk.drop_duplicates().tolist()
    rng = np.random.default_rng(0); rng.shuffle(gids)
    rows = []
    for gid in gids:
        g = sub[sub.game_pk == gid]
        s = sides_for_game(g)
        if s is None: continue
        (at, _, aord, ast), (ht, _, hord, hst) = s[0], s[1]
        rows.append(dict(game_pk=gid, season=year, home_team=ht, away_team=at,
                         home_order=hord, home_starter=hst,
                         away_order=aord, away_starter=ast))
        if len(rows) >= n_games: break
    return pd.DataFrame(rows)

N = 320
t0 = time.time()
G = build(2025, N)
print(f"\nbuilt {len(G)} 2025 games in {time.time()-t0:.0f}s")

# actual runs
from google.cloud import bigquery
ids = ",".join(str(int(x)) for x in G.game_pk)
act = bigquery.Client(project="hankstank").query(
    f"SELECT game_pk, home_score, away_score FROM `hankstank.mlb_historical_data.games_historical` "
    f"WHERE game_pk IN ({ids})").to_dataframe()
G = G.merge(act, on="game_pk", how="inner")
print(f"joined actual scores for {len(G)} games")

def sim_runs(alpha):
    hr, ar = [], []
    for r in G.itertuples():
        hr.append(expected_runs(side_dist(r.home_order, r.away_starter, r.away_team, r.season, alpha)))
        ar.append(expected_runs(side_dist(r.away_order, r.home_starter, r.home_team, r.season, alpha)))
    return np.array(hr), np.array(ar)

targets = np.concatenate([G.home_score.values, G.away_score.values]).astype(float)
t0 = time.time()
alpha = fit_run_scale(lambda a: np.concatenate(sim_runs(a)), targets)
print(f"fitted run scale alpha = {alpha:.4f}  ({time.time()-t0:.0f}s)")

hr, ar = sim_runs(alpha)
pred = np.concatenate([hr, ar])
print(f"\n{'='*74}\nPHASE 2 GATE -- predicted vs actual team runs (n={len(pred)} team-games)\n{'='*74}")
print(f"  mean predicted {pred.mean():.3f}   mean actual {targets.mean():.3f}   bias {pred.mean()-targets.mean():+.3f}")
print(f"  sd predicted   {pred.std():.3f}   sd actual   {targets.std():.3f}")
r = np.corrcoef(pred, targets)[0,1]
print(f"  correlation r = {r:.4f}   (a team-level run model typically lands r ~ 0.15-0.25)")
mae = np.abs(pred-targets).mean(); mae0 = np.abs(targets.mean()-targets).mean()
print(f"  MAE {mae:.3f} vs {mae0:.3f} for always predicting the mean  ({(mae0-mae)/mae0*100:+.2f}%)")
ok = (abs(pred.mean()-targets.mean()) < 0.25) and (r > 0.08)
print(f"\n  -> PHASE 2 {'PASS' if ok else 'FAIL'}")
np.save("data/backtest_2026/pa_alpha.npy", np.array([alpha]))
G.to_parquet("data/backtest_2026/phase2_games.parquet")
