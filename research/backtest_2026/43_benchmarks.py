"""Benchmarks every simulator variant is scored against (local files only).

  market    closing moneyline (kaggle vig-free, falling back to SBR), closing total +
            over price, closing run line (SBR archive 2015-2021, joined on date + home
            team + final score)
  strength  walk-forward team-strength logistic: Elo (from game results) + season-to-date
            pythag, fit only on games before each season
  totals    walk-forward Poisson GLM on team RS/RA per game to date + venue runs factor,
            negative-binomial dispersion fit on training seasons
  2026      V10 as served (pregame) and the 3-feature logistic inputs

Output: data/backtest_2026/rich/bench.parquet
"""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor

R = "data/backtest_2026/rich/"
G = pd.read_parquet(R + "games.parquet")
meta = pd.read_parquet(R + "sched_meta.parquet")[["game_pk", "venue_id", "sched_inn", "temp", "condition", "roof",
                                                  "games_in_series", "series_game", "home_id", "away_id"]]
G = G.merge(meta, on="game_pk", how="left")
G["dkey"] = G.game_date.dt.strftime("%Y%m%d").astype(int)

# ---------------- market: SBR archive ----------------
NICK = {"Angels": "LAA", "Astros": "HOU", "Athletics": "ATH", "Blue Jays": "TOR", "Braves": "ATL",
        "Brewers": "MIL", "CUB": "CHC", "Cardinals": "STL", "Cubs": "CHC", "Diamondbacks": "AZ",
        "Dodgers": "LAD", "Indians": "CLE", "KAN": "KC", "LOS": "LOS", "Mariners": "SEA", "Marlins": "MIA",
        "Mets": "NYM", "Nationals": "WSH", "Orioles": "BAL", "Phillies": "PHI", "Pirates": "PIT",
        "Rangers": "TEX", "Red Sox": "BOS", "Reds": "CIN", "Rockies": "COL", "SDG": "SD", "SFO": "SF",
        "TAM": "TB", "Tigers": "DET", "Twins": "MIN", "White Sox": "CWS", "Yankees": "NYY"}
s = pd.DataFrame(json.load(open("data/odds/sbr_mlb_archive_10Y.json")))
# The scraped SBR archive mis-pairs rows: each record's home_* fields describe one team and
# its away_* fields a team from a DIFFERENT game. Per-team fields (final, ML, run line) are
# right for their own team; the record's over/under belongs to the game of the team in the
# record's home_* fields, which is that game's VISITOR (verified: line~total r=0.218 vs 0.039).
s = s[(s.season >= 2015) & s.date.notna()].copy()
def imp(o):
    o = np.asarray(o, float)
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))
tm = []
for r in s.itertuples():
    for side in ("home", "away"):
        tm.append(dict(d=int(r.date), team=NICK.get(getattr(r, side + "_team")), side=side,
                       fin=pd.to_numeric(getattr(r, side + "_final"), errors="coerce"),
                       ml=pd.to_numeric(getattr(r, side + "_close_ml"), errors="coerce"),
                       sp=pd.to_numeric(getattr(r, side + "_close_spread"), errors="coerce"),
                       spo=pd.to_numeric(getattr(r, side + "_close_spread_odds"), errors="coerce"),
                       ou=pd.to_numeric(r.close_over_under, errors="coerce"),
                       ouo=pd.to_numeric(r.close_over_under_odds, errors="coerce")))
tm = pd.DataFrame(tm)
tm = tm[tm.fin.notna() & tm.team.notna()]
def attach(gs_team, gs_runs, pref):
    out = []
    for t in (["LAD", "LAA"],):
        pass
    x = tm.copy()
    x = pd.concat([x[x.team != "LOS"], x[x.team == "LOS"].assign(team="LAD"), x[x.team == "LOS"].assign(team="LAA")])
    m = x.merge(G[["game_pk", "dkey", gs_team, gs_runs]].rename(columns={"dkey": "d", gs_team: "team", gs_runs: "fin"}),
                on=["d", "team", "fin"]).drop_duplicates("game_pk", keep=False)
    return m.set_index("game_pk")
H_ = attach("home_team", "h_runs", "h"); A_ = attach("away_team", "a_runs", "a")
sb = pd.DataFrame(index=G.game_pk.unique())
ph, pa_ = imp(H_.ml).astype(float), imp(A_.ml).astype(float)
sb["ml_h"] = pd.Series(ph, index=H_.index); sb["ml_a"] = pd.Series(pa_, index=A_.index)
sb["sbr_p"] = sb.ml_h / (sb.ml_h + sb.ml_a)
Ahome = A_[A_.side == "home"]                     # visitor rows carrying the game's total
sb["total_line"] = Ahome.ou.where((Ahome.ou > 4) & (Ahome.ou < 16))
oo = Ahome.ouo
sb["over_p"] = pd.Series(np.where((oo.abs() >= 100) & (oo.abs() <= 160), imp(oo) / 1.045, np.nan), index=Ahome.index)
sb["rl_home_spread"] = H_.sp
rp = pd.Series(imp(H_.spo), index=H_.index).where((H_.spo.abs() >= 100) & (H_.spo.abs() <= 300) & (H_.sp.abs() == 1.5))
ra_ = pd.Series(imp(A_.spo), index=A_.index).where((A_.spo.abs() >= 100) & (A_.spo.abs() <= 300))
sb["rl_home_p"] = rp / (rp + ra_)
sb = sb.drop(columns=["ml_h", "ml_a"]).reset_index().rename(columns={"index": "game_pk"})

G = G.merge(sb, on="game_pk", how="left")
k = pd.read_parquet("data/odds/arena.parquet")[["game_pk", "market_home_prob"]]
G = G.merge(k, on="game_pk", how="left")
G["mkt_p"] = G.market_home_prob.fillna(G.sbr_p)
print("market coverage by year:\n", G.groupby("year")[["mkt_p", "total_line", "over_p", "rl_home_p"]]
      .apply(lambda x: x.notna().mean()).round(3).to_string())
print("ML agreement kaggle vs sbr r =", G[["market_home_prob", "sbr_p"]].corr().iloc[0, 1].round(4))

# ---------------- team strength: Elo + pythag ----------------
G = G.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
elo = {}; K = 4.0; HFA = 24.0
e_diff = np.zeros(len(G)); last_year = {}
rs, ra, gp = {}, {}, {}
py = np.zeros(len(G)); tot_feats = np.zeros((len(G), 5))
lg_rpg = {}
for i, r in enumerate(G.itertuples()):
    y = r.year
    for t in (r.home_team, r.away_team):
        if last_year.get(t) != y:              # regress 1/3 toward mean each season
            elo[t] = 1500 + (elo.get(t, 1500) - 1500) * (2 / 3); last_year[t] = y
            rs[(t, y)] = 0.0; ra[(t, y)] = 0.0; gp[(t, y)] = 0
    eh, ea = elo[r.home_team], elo[r.away_team]
    e_diff[i] = eh - ea + HFA
    def pyth(t):
        n = gp[(t, y)]; a, b = rs[(t, y)] + 4.5 * 10, ra[(t, y)] + 4.5 * 10
        return a ** 1.83 / (a ** 1.83 + b ** 1.83)
    py[i] = pyth(r.home_team) - pyth(r.away_team)
    def pg(d, t):
        return (d[(t, y)] + 10 * 4.5) / (gp[(t, y)] + 10)
    tot_feats[i] = [pg(rs, r.home_team), pg(ra, r.home_team), pg(rs, r.away_team), pg(ra, r.away_team), 0]
    # update
    hw = r.h_runs > r.a_runs
    pe = 1 / (1 + 10 ** (-(e_diff[i]) / 400))
    mov = np.log1p(abs(r.h_runs - r.a_runs))
    elo[r.home_team] += K * mov * (hw - pe); elo[r.away_team] -= K * mov * (hw - pe)
    rs[(r.home_team, y)] += r.h_runs; ra[(r.home_team, y)] += r.a_runs; gp[(r.home_team, y)] += 1
    rs[(r.away_team, y)] += r.a_runs; ra[(r.away_team, y)] += r.h_runs; gp[(r.away_team, y)] += 1
G["elo_d"] = e_diff; G["pyth_d"] = py
G["y"] = (G.h_runs > G.a_runs).astype(int)
G["strength_p"] = np.nan
for y in sorted(G.year.unique()):
    tr = (G.year < y) & (G.year >= y - 3); te = G.year == y
    if tr.sum() < 1000: continue
    X = G[["elo_d", "pyth_d"]].values
    m = LogisticRegression(C=1.0).fit(X[tr], G.y[tr])
    G.loc[te, "strength_p"] = m.predict_proba(X[te])[:, 1]

# ---------------- totals baseline: Poisson GLM + venue factor, NB dispersion ----------------
G["tot"] = G.h_runs + G.a_runs
G[["h_rs", "h_ra", "a_rs", "a_ra"]] = tot_feats[:, :4]
vf = np.ones(len(G))
for y in sorted(G.year.unique()):
    tr = (G.year < y) & (G.year >= y - 3)
    v = G[tr].groupby("venue_id").tot.agg(["sum", "count"])
    lg = G[tr].tot.mean() if tr.sum() else 8.8
    f = ((v["sum"] + 300 * lg) / (v["count"] + 300)) / lg
    vf[G.year == y] = G.loc[G.year == y, "venue_id"].map(f).fillna(1.0).values
G["venue_rf"] = vf
G["tot_mu_base"] = np.nan; G["nb_r_base"] = np.nan
for y in sorted(G.year.unique()):
    tr = ((G.year < y) & (G.year >= y - 3)).values; te = (G.year == y).values
    if tr.sum() < 1000: continue
    X = np.log(G[["h_rs", "h_ra", "a_rs", "a_ra", "venue_rf"]].values)
    m = PoissonRegressor(alpha=1e-4, max_iter=500).fit(X[tr], G.tot[tr])
    mu_tr = m.predict(X[tr])
    # NB2 dispersion by moments: Var = mu + mu^2/r
    r_nb = float(np.sum(mu_tr ** 2) / max(np.sum((G.tot[tr] - mu_tr) ** 2 - mu_tr), 1e-6))
    G.loc[te, "tot_mu_base"] = m.predict(X[te]); G.loc[te, "nb_r_base"] = r_nb

# ---------------- 2026: V10 pregame + logit3 inputs ----------------
v10a = pd.read_parquet("data/backtest_2026/games_2026_pregame.parquet")[["game_pk", "home_win_probability"]]
v10b = pd.read_parquet(R + "lineups_post0907.parquet")[["game_pk", "home_win_probability"]].drop_duplicates("game_pk")
v10 = pd.concat([v10a, v10b]).drop_duplicates("game_pk").rename(columns={"home_win_probability": "v10_p"})
f3 = pd.read_parquet(R + "feats_2026_all.parquet").drop(columns=["game_date"])
G = G.merge(v10, on="game_pk", how="left").merge(f3, on="game_pk", how="left")
G.to_parquet(R + "bench.parquet", index=False)
print(G.shape)
print(G.groupby("year")[["strength_p", "tot_mu_base", "v10_p", "elo_differential"]].apply(lambda x: x.notna().mean()).round(3).to_string())
