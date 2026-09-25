"""Experiment 2 -- injuries / IL burden. New information, already in our warehouse.

PREDICTION REGISTERED: this should help more than weather did, because unlike weather it
is ASYMMETRIC -- it hits one team and not the other. But the effect should still be small,
because the market reads the same IL reports, and rolling team stats already partly absorb
an absence (a team playing badly without its ace looks like a worse team).

Method: reconstruct each team's IL roster as of every game date by walking the transaction
log forward, then weight each absent player by how much he actually plays -- an unweighted
count treats a lost ace and a lost 26th man identically.
"""
import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from edge.benchmark import score
from google.cloud import bigquery

bq = bigquery.Client(project="hankstank")
tx = bq.query("""
WITH mlb AS (SELECT DISTINCT team_id FROM `hankstank.mlb_2026_season.teams`)
SELECT PARSE_DATE('%Y-%m-%d', SUBSTR(t.date,1,10)) AS tdate,
       t.person_id, t.description,
       COALESCE(t.to_team_id, t.from_team_id) AS team_id
FROM `hankstank.mlb_historical_data.transactions_historical` t
WHERE SUBSTR(t.date,1,4) BETWEEN '2014' AND '2021'
  AND (t.to_team_id IN (SELECT team_id FROM mlb) OR t.from_team_id IN (SELECT team_id FROM mlb))
  AND (LOWER(t.description) LIKE '%disabled list%'
       OR LOWER(t.description) LIKE '%injured list%')
""").to_dataframe()
tx["tdate"] = pd.to_datetime(tx.tdate)
dl = tx.description.str.lower()
tx["is_return"] = dl.str.contains("activated|reinstated", regex=True)
tx["is_place"] = dl.str.contains("placed|transferred", regex=True) & ~tx.is_return
tx = tx[(tx.is_place | tx.is_return) & tx.person_id.notna() & tx.team_id.notna()].copy()
tx["person_id"] = tx.person_id.astype(int); tx["team_id"] = tx.team_id.astype(int)
print(f"IL transactions 2014-2021 (MLB only): {len(tx)}  "
      f"({int(tx.is_place.sum())} placements, {int(tx.is_return.sum())} returns)")

pa = pd.read_parquet("data/backtest_2026/pa_2015_2026.parquet")
bv = pa.groupby(["game_year","batter"], observed=True).size().rename("vol").reset_index() \
       .rename(columns={"batter":"pid"})
pv = pa.groupby(["game_year","pitcher"], observed=True).size().rename("vol").reset_index() \
       .rename(columns={"pitcher":"pid"})
vol = pd.concat([bv, pv], ignore_index=True).groupby(["game_year","pid"], observed=True) \
        .vol.max().reset_index()
volmap = {(int(r.game_year)+1, int(r.pid)): float(r.vol) for r in vol.itertuples()}
print(f"prior-season workload known for {len(volmap)} (season, player) pairs")

arena = pd.read_parquet("data/odds/arena_weather.parquet")
arena["game_date"] = pd.to_datetime(arena.game_date)
gdates = set(arena.game_date)

# --- build BOUNDED IL stints, not a cumulative counter ---
# The transaction log is unbalanced: 6,012 placements vs 4,081 returns, because some
# activations are filed under other transaction types. Walking it forward without a cap
# makes players accumulate on the IL forever -- the first version of this feature climbed
# from 4.3 players out in Apr-2015 to 20.9 by Aug-2018, i.e. it measured season length.
# Each placement therefore closes at the next observed return for that player, or after
# MEDIAN_STINT days, whichever comes first.
MEDIAN_STINT = pd.Timedelta(days=30)
MAX_STINT = pd.Timedelta(days=120)

intervals = []
for (team, pid), g in tx.sort_values("tdate").groupby(["team_id", "person_id"], sort=False):
    open_at = None
    for r in g.itertuples():
        if r.is_place:
            if open_at is not None:                      # placement with no return seen
                intervals.append((team, pid, open_at, min(open_at + MEDIAN_STINT, r.tdate)))
            open_at = r.tdate
        else:                                            # a return closes the stint
            if open_at is not None:
                end = min(r.tdate, open_at + MAX_STINT)
                intervals.append((team, pid, open_at, end))
                open_at = None
    if open_at is not None:
        intervals.append((team, pid, open_at, open_at + MEDIAN_STINT))
IV = pd.DataFrame(intervals, columns=["team_id", "pid", "start", "end"])
print(f"IL stints reconstructed: {len(IV)}  median length "
      f"{(IV.end - IV.start).median().days} days")

gd = np.array(sorted(gdates))
rows = []
for team, g in IV.groupby("team_id"):
    st = g.start.values.astype("datetime64[ns]")
    en = g.end.values.astype("datetime64[ns]")
    pw = np.array([volmap.get((pd.Timestamp(s).year, int(p)), 0.0)
                   for s, p in zip(g.start.values, g.pid.values)])
    for dt in gd:
        live = (st <= dt) & (dt < en)
        if live.any() or True:
            rows.append(dict(game_date=pd.Timestamp(dt), team_id=team,
                             il_count=int(live.sum()), il_weighted=float(pw[live].sum())))
IL = pd.DataFrame(rows)
print(f"IL snapshots: {len(IL)} (team,date) rows; mean out {IL.il_count.mean():.2f} players, "
      f"mean weighted {IL.il_weighted.mean():.0f} PA/BF")
mm = IL.copy(); mm["mo"] = mm.game_date.dt.to_period("M")
chk = mm.groupby("mo").il_count.mean()
print("  sanity, mean players out by month (should oscillate, not climb): "
      + ", ".join(f"{k}:{v:.1f}" for k, v in list(chk.items())[:6]) + " ... "
      + ", ".join(f"{k}:{v:.1f}" for k, v in list(chk.items())[-3:]))

a = arena.merge(IL.rename(columns={"team_id":"home_team_id","il_count":"home_il_count",
                                   "il_weighted":"home_il_wt"}),
                on=["game_date","home_team_id"], how="left")
a = a.merge(IL.rename(columns={"team_id":"away_team_id","il_count":"away_il_count",
                               "il_weighted":"away_il_wt"}),
            on=["game_date","away_team_id"], how="left")
for c in ("home_il_count","away_il_count","home_il_wt","away_il_wt"):
    a[c] = a[c].fillna(0.0)
a["il_count_diff"] = a.away_il_count - a.home_il_count       # + = home healthier
a["il_wt_diff"] = a.away_il_wt - a.home_il_wt
a["il_wt_diff_log"] = np.sign(a.il_wt_diff) * np.log1p(np.abs(a.il_wt_diff))

BASE = [c for c in ["elo_differential","pythag_differential","win_pct_diff",
                    "run_diff_differential","home_pythag_last30","away_pythag_last30",
                    "era_proxy_differential","streak_differential","luck_differential"]
        if c in a.columns]
IL_RAW = ["home_il_count","away_il_count","il_count_diff"]
IL_WT = ["il_wt_diff_log","home_il_wt","away_il_wt"]

y = a.home_won.values.astype(int)
tr = (a.game_date < pd.Timestamp("2020-01-01")).values
te = ~tr
half = np.zeros(int(te.sum()), bool); half[:int(te.sum()*0.5)] = True

def run(cols, name):
    X = a[cols].astype(float); X = X.fillna(X[tr].median()).fillna(0)
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=0.1, max_iter=4000).fit(sc.transform(X[tr]), y[tr])
    p = m.predict_proba(sc.transform(X))[:, 1]
    return (score(y[te], p[te], a.market_home_prob.values[te], a.home_moneyline.values[te],
                  a.away_moneyline.values[te], name=name, train_mask=half, verbose=False),
            m, cols)

res = [run(BASE, "A. baseline"), run(IL_RAW+IL_WT, "B. IL ONLY (control)"),
       run(BASE+IL_RAW, "C. baseline + IL counts"),
       run(BASE+IL_RAW+IL_WT, "D. baseline + IL + workload-weighted")]

print(f"\n{'='*92}\nEXPERIMENT 2 -- INJURIES  (test 2020-21, n={int(te.sum())})\n{'='*92}")
print(f"  {'model':42}{'acc':>8}{'AUC':>9}{'log-loss':>11}{'gap':>12}")
for r,_,_ in res:
    print(f"  {r['name']:42}{r['acc']:>7.2f}%{r['auc']:>9.4f}{r['logloss']:>11.5f}{r['gap_nats']:>+12.5f}")
print(f"  {'closing line':42}{res[0][0]['market_acc']:>7.2f}%{res[0][0]['market_auc']:>9.4f}"
      f"{res[0][0]['market_logloss']:>11.5f}{0.0:>+12.5f}")
print(f"\n  orthogonal edge over the line:")
for r,_,_ in res:
    if "orth_gain_nats" in r:
        print(f"    {r['name']:42}{r['orth_gain_nats']:>+10.5f}  P(helps)={r['orth_p_better']:.3f}")
print(f"\n  IL coefficients in model D:")
rD, mD, cD = res[-1]
for n_, co in sorted(zip(cD, mD.coef_[0]), key=lambda kv:-abs(kv[1])):
    if n_ in IL_RAW+IL_WT: print(f"    {n_:22}{co:+.4f}")
g = res[0][0]["logloss"] - res[3][0]["logloss"]
print(f"\n  VERDICT: IL features changed log-loss by {g:+.5f} nats "
      f"({'HELPS' if g>0.001 else 'no material help'})")
a.to_parquet("data/odds/arena_il.parquet", index=False)
