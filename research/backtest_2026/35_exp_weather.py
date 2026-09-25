"""Experiment 1 -- weather. Does new physical information close the gap to the line?

PREDICTION REGISTERED BEFORE RUNNING: weather will add ~nothing to winner prediction.
Both teams play in the same weather, so it moves the run environment symmetrically; it
should be a TOTALS signal, not a moneyline one. It can only affect the winner through
interaction with which team benefits from a high- or low-scoring game.

Testing that prediction is the point. A confirmed null is a real result: it tells you
where to stop looking.

Feature engineering is mechanism-driven, not dumped in raw:
  wind_out   -- signed projection along the batter's line (+out / -in / 0 crosswind).
                This is the component that actually carries fly balls.
  temp       -- warmer air is less dense; the ball carries further.
  dome       -- indoors, all of the above is meaningless and gets masked to zero.
"""
import sys, warnings, re
import numpy as np, pandas as pd
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from edge.benchmark import score

a = pd.read_parquet("data/odds/arena.parquet")
w = pd.read_parquet("data/odds/weather_2015_2021.parquet")
a["game_date"] = pd.to_datetime(a.game_date)
d = a.merge(w.drop(columns=["game_date"]), on="game_pk", how="left")
print(f"arena {len(a)} games; weather joined on {d.temp.notna().mean()*100:.1f}%")

# --- mechanism-driven encoding ---
d["temp_f"] = pd.to_numeric(d.temp, errors="coerce")
def parse_wind(s):
    """'12 mph, In From RF' -> (12, 'In From RF'). Returns (speed, signed_out, cross)."""
    if not isinstance(s, str):
        return np.nan, np.nan, np.nan
    m = re.match(r"\s*(\d+)\s*mph", s)
    spd = float(m.group(1)) if m else np.nan
    t = s.lower()
    if "out to" in t:      return spd, +spd, 0.0
    if "in from" in t:     return spd, -spd, 0.0
    if " to " in t:        return spd, 0.0, spd      # L To R / R To L = crosswind
    if "varies" in t or "none" in t: return spd, 0.0, 0.0
    return spd, 0.0, 0.0
pw = d.wind.apply(parse_wind)
d["wind_speed"] = [x[0] for x in pw]
d["wind_out"]   = [x[1] for x in pw]
d["wind_cross"] = [x[2] for x in pw]
d["is_dome"] = d.roof.isin(["Dome", "Closed"]).astype(int)
# indoors the weather channel carries no information -> mask it, don't let the model
# learn a spurious "domes are different" effect through the weather columns
for c in ("temp_f", "wind_out", "wind_cross", "wind_speed"):
    d.loc[d.is_dome == 1, c] = np.nan
d["is_rain"] = d.condition.fillna("").str.contains("Rain|Drizzle", case=False).astype(int)
d["is_clear"] = d.condition.fillna("").str.contains("Sunny|Clear", case=False).astype(int)

# a single scalar for "how much offence will this environment allow"
d["run_env"] = (0.030 * (d.temp_f.fillna(72) - 72) + 0.035 * d.wind_out.fillna(0)
                - 0.010 * d.wind_cross.fillna(0))
# mechanism interaction: a high-scoring environment should favour whichever side has
# the bigger offence-minus-pitching edge
off_edge = d.pythag_differential.fillna(0)
d["runenv_x_offedge"] = d.run_env * off_edge

BASE = [c for c in ["elo_differential","pythag_differential","win_pct_diff",
                    "run_diff_differential","home_pythag_last30","away_pythag_last30",
                    "era_proxy_differential","streak_differential","luck_differential"]
        if c in d.columns]
WX_RAW = ["temp_f","wind_out","wind_cross","is_dome","is_rain","is_clear"]
WX_MECH = ["run_env","runenv_x_offedge"]

y = d.home_won.values.astype(int)
tr = (d.game_date < pd.Timestamp("2020-01-01")).values
te = ~tr
half = np.zeros(int(te.sum()), bool); half[:int(te.sum()*0.5)] = True

def run(cols, name):
    X = d[cols].astype(float)
    X = X.fillna(X[tr].median()).fillna(0)
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=0.1, max_iter=4000).fit(sc.transform(X[tr]), y[tr])
    p = m.predict_proba(sc.transform(X))[:, 1]
    r = score(y[te], p[te], d.market_home_prob.values[te],
              d.home_moneyline.values[te], d.away_moneyline.values[te],
              name=name, train_mask=half, verbose=False)
    return r, m, cols

results = []
for cols, nm in [(BASE, "A. baseline (team strength)"),
                 (WX_RAW, "B. weather ONLY (control - expect ~0.50 AUC)"),
                 (BASE + WX_RAW, "C. baseline + weather"),
                 (BASE + WX_RAW + WX_MECH, "D. baseline + weather + mechanism interaction")]:
    r, m, c = run(cols, nm)
    results.append((r, m, c))

print(f"\n{'='*94}")
print(f"EXPERIMENT 1 -- WEATHER   (train 2015-19, test 2020-21, n={int(te.sum())})")
print(f"{'='*94}")
print(f"  {'model':46}{'acc':>8}{'AUC':>9}{'log-loss':>11}{'gap to line':>13}")
for r, _, _ in results:
    print(f"  {r['name']:46}{r['acc']:>7.2f}%{r['auc']:>9.4f}{r['logloss']:>11.5f}"
          f"{r['gap_nats']:>+13.5f}")
print(f"  {'closing line':46}{results[0][0]['market_acc']:>7.2f}%"
      f"{results[0][0]['market_auc']:>9.4f}{results[0][0]['market_logloss']:>11.5f}{0.0:>+13.5f}")

print(f"\n  orthogonal edge over the line:")
for r, _, _ in results:
    if "orth_gain_nats" in r:
        print(f"    {r['name']:46}{r['orth_gain_nats']:>+10.5f} nats  "
              f"P(helps)={r['orth_p_better']:.3f}")

print(f"\n  weather coefficients in model D (standardised):")
rD, mD, cD = results[-1]
for n_, co in sorted(zip(cD, mD.coef_[0]), key=lambda kv: -abs(kv[1])):
    if n_ in WX_RAW + WX_MECH:
        print(f"    {n_:24}{co:+.4f}")

print(f"\n  VERDICT vs the registered prediction:")
gain = results[2][0]["logloss"] - results[0][0]["logloss"]
print(f"    weather added {-gain:+.5f} nats to winner prediction "
      f"({'confirms' if abs(gain) < 0.002 else 'CONTRADICTS'} the null prediction)")
d.to_parquet("data/odds/arena_weather.parquet", index=False)

# --- and now the part the prediction says SHOULD work: total runs ---
print(f"\n{'='*94}\nTHE OTHER HALF -- does weather predict TOTAL RUNS?\n{'='*94}")
from google.cloud import bigquery
sc_ = bigquery.Client(project="hankstank").query("""
  SELECT game_pk, home_score + away_score AS total_runs
  FROM `hankstank.mlb_historical_data.games_historical`
  WHERE game_type='R' AND home_score IS NOT NULL""").to_dataframe()
dd = d.merge(sc_, on="game_pk", how="inner")
out = dd[dd.is_dome == 0].dropna(subset=["temp_f", "wind_out", "total_runs"])
print(f"  outdoor games with weather and a final score: {len(out)}")
for var, lab in [("temp_f", "temperature (F)"), ("wind_out", "wind out (mph, signed)"),
                 ("run_env", "run_env composite")]:
    q = pd.qcut(out[var], 5, labels=False, duplicates="drop")
    means = [out.total_runs[q == i].mean() for i in range(int(q.max()) + 1)]
    r_ = np.corrcoef(out[var], out.total_runs)[0, 1]
    print(f"  {lab:26} r={r_:+.4f}   quintile mean total runs: "
          + " -> ".join(f"{m:.2f}" for m in means))
