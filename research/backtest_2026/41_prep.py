"""Build the unified tables the v2 simulator research runs on (local files only).

Outputs (data/backtest_2026/rich/):
  pa_all.parquet     one row per PA 2015-2026, 9-class outcome (pa_sim.v2.CLASSES)
  trans.parquet      empirical base-out transitions (2015-2025, from Savant state)
  hook.parquet       starter-removal opportunities (2015-2025) for the hook hazard
  games.parquet      one row per game: lineups, starters, final + F5 score, innings
"""
import numpy as np, pandas as pd

R = "data/backtest_2026/rich/"
h = pd.read_parquet(R + "pa_hist.parquet")
c = pd.read_parquet(R + "pa_2026.parquet")

DROP = {"truncated_pa", "ejection", "game_advisory"}
h = h[~h.events.isin(DROP)].copy()
c = c[~c.events.isin(DROP)].copy()

# K BB 1B 2B 3B HR E GO AO
def classify(df, air):
    ev = df.events.values
    out = np.full(len(df), -1, np.int8)
    m = lambda *e: np.isin(ev, e)
    out[m("strikeout", "strikeout_double_play")] = 0
    out[m("walk", "intent_walk", "hit_by_pitch", "catcher_interf")] = 1
    out[m("single")] = 2; out[m("double")] = 3; out[m("triple")] = 4; out[m("home_run")] = 5
    out[m("field_error")] = 6
    ip = m("field_out", "force_out", "sac_fly", "sac_bunt", "fielders_choice", "fielders_choice_out",
           "grounded_into_double_play", "double_play", "sac_fly_double_play", "triple_play",
           "sac_bunt_double_play")
    ground = m("grounded_into_double_play", "force_out", "fielders_choice", "fielders_choice_out",
               "sac_bunt", "sac_bunt_double_play")
    airy = m("sac_fly", "sac_fly_double_play")
    is_air = np.where(ground, False, np.where(airy, True, air))
    out[ip & ~is_air] = 7
    out[ip & is_air] = 8
    return out

h_air = h.bb_type.isin(["fly_ball", "line_drive", "popup"]).values
h_air = np.where(h.bb_type.isna(), (h.launch_angle.fillna(20) >= 10).values, h_air)
h["cls"] = classify(h, h_air)
c["cls"] = classify(c, (c.launch_angle.fillna(20) >= 10).values)
h = h[h.cls >= 0]; c = c[c.cls >= 0]
print("class shares hist", np.bincount(h.cls, minlength=9) / len(h))
print("class shares 2026", np.bincount(c.cls, minlength=9) / len(c))

# ---- starters and TTO, hist (true order) ----
h = h.sort_values(["game_pk", "at_bat_number"]).reset_index(drop=True)
h["top"] = h.inning_topbot.eq("Top")
first = h.groupby(["game_pk", "top"]).pitcher.transform("first")
h["is_starter"] = h.pitcher.eq(first)
# ---- 2026: starter = pitcher who appeared in inning 1 for that defense ----
c["top"] = c.inning_topbot.eq("Top")
fi = c[c.inning == 1].groupby(["game_pk", "top"]).pitcher.agg(lambda s: s.value_counts().index[0])
c = c.join(fi.rename("sp"), on=["game_pk", "top"])
c["is_starter"] = c.pitcher.eq(c.sp)

cols = ["game_pk", "game_date", "game_year", "batter", "pitcher", "stand", "p_throws", "inning", "top",
        "home_team", "away_team", "cls", "n_pitches", "launch_speed", "launch_angle", "xwoba",
        "is_starter"]
pa = pd.concat([h[cols + ["n_thruorder_pitcher", "at_bat_number"]], c[cols]], ignore_index=True)
pa["stand"] = (pa.stand == "R").astype(np.int8); pa["p_throws"] = (pa.p_throws == "R").astype(np.int8)
for k in ("batter", "pitcher", "game_pk"): pa[k] = pa[k].astype(np.int64)
pa["inning"] = pa.inning.astype(np.int16); pa["game_year"] = pa.game_year.astype(np.int16)
pa = pa.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
pa.to_parquet(R + "pa_all.parquet", index=False)
print("pa_all", pa.shape)

# ---- empirical transitions from Savant state (hist only) ----
h["bases"] = (h.r1.astype(int) + 2 * h.r2.astype(int) + 4 * h.r3.astype(int)).astype(np.int8)
h["half"] = h.game_pk.astype(str) + h.inning.astype(str) + h.top.astype(str)
g = h.groupby(["game_pk", "inning", "top"], sort=False)
nxt_b = g.bases.shift(-1); nxt_o = g.outs_when_up.shift(-1); nxt_s = g.bat_score.shift(-1)
last = nxt_b.isna()
runs = np.where(last, h.post_bat_score - h.bat_score, nxt_s - h.bat_score)
no = np.where(last, 3, nxt_o)
nb = np.where(last, 0, nxt_b)
# walk-off / game-ending last PAs never reach 3 outs -- drop them from the chain
maxinn = h.groupby("game_pk").inning.transform("max")
walk = last & ~h.top & (h.inning >= 9) & (h.inning == maxinn) & (h.post_home_score > h.post_away_score)
ok = ~walk & (runs >= 0) & (runs <= 4) & (no >= h.outs_when_up)
T = pd.DataFrame(dict(year=h.game_year.values, bases=h.bases.values, outs=h.outs_when_up.values,
                      cls=h.cls.values, nb=nb, no=no, runs=runs))[ok.values]
T = T.astype(np.int16)
T.to_parquet(R + "trans.parquet", index=False)
print("trans", T.shape)

# ---- hook opportunities: starter PAs in order, y=1 when the starter is gone ----
h["def_top"] = h.top
h["cum_p"] = h.groupby(["game_pk", "top"]).n_pitches.cumsum() - h.n_pitches
h["bf"] = h.groupby(["game_pk", "top"]).cumcount()
h["ra"] = h.bat_score - h.groupby(["game_pk", "top"]).bat_score.transform("first")
h["inn_start"] = h.groupby(["game_pk", "inning", "top"]).cumcount().eq(0)
st = h.is_starter.values
prev_st = h.groupby(["game_pk", "top"]).is_starter.shift(1).fillna(True).astype(bool).values
gone_before = (~h.is_starter).groupby([h.game_pk, h.top]).cumsum().values - (~st).astype(int)
opp = prev_st & (gone_before == 0)
hk = h.loc[opp, ["game_pk", "game_date", "game_year", "top", "pitcher", "inning", "outs_when_up",
                 "cum_p", "bf", "ra", "inn_start", "bases"]].copy()
hk["sp"] = first[opp].values
hk["y"] = (~h.is_starter[opp]).astype(np.int8).values
hk.to_parquet(R + "hook.parquet", index=False)
print("hook", hk.shape, hk.y.mean())

# ---- games ----
def lineup(df):
    seen = []
    for b in df.batter.values:
        if b not in seen:
            seen.append(b)
            if len(seen) == 9: break
    return seen

rows = []
fin = h.groupby("game_pk").agg(hr=("post_home_score", "max"), ar=("post_away_score", "max"),
                               inn=("inning", "max"), date=("game_date", "first"),
                               year=("game_year", "first"), hteam=("home_team", "first"),
                               ateam=("away_team", "first"))
f5 = h[h.inning <= 5].groupby("game_pk").agg(h5=("post_home_score", "max"), a5=("post_away_score", "max"),
                                             i5=("inning", "max"))
lu = {k: lineup(v) for k, v in h.groupby(["game_pk", "top"], sort=False)}
sp = first.groupby([h.game_pk, h.top]).first().to_dict()
for gid, r in fin.iterrows():
    hl, al = lu.get((gid, False), []), lu.get((gid, True), [])
    if len(hl) != 9 or len(al) != 9: continue
    f = f5.loc[gid] if gid in f5.index else None
    has5 = f is not None and f.i5 == 5 and r.inn >= 6
    rows.append(dict(game_pk=gid, game_date=r.date, year=int(r.year), home_team=r.hteam, away_team=r.ateam,
                     h_lineup=hl, a_lineup=al, h_sp=int(sp[(gid, True)]), a_sp=int(sp[(gid, False)]),
                     h_runs=int(r.hr), a_runs=int(r.ar), h_f5=int(f.h5) if has5 else -1,
                     a_f5=int(f.a5) if has5 else -1, innings=int(r.inn), src="hist"))

# 2026: pregame lineup snapshots (Apr-Sep7 + post Sep7)
l1 = pd.read_parquet("data/backtest_2026/lineups_2026.parquet")
l2 = pd.read_parquet(R + "lineups_post0907.parquet")
lu26 = pd.concat([l1, l2], ignore_index=True).drop_duplicates(["game_pk", "team_type", "batting_order"])
g26 = pd.read_parquet(R + "games_2026.parquet").set_index("game_pk")
c5 = c[c.inning == 6].sort_values(["game_pk"]).groupby("game_pk").agg(h5=("home_score", "min"), a5=("away_score", "min"))
cmeta = c.groupby("game_pk").agg(hteam=("home_team", "first"), ateam=("away_team", "first"), inn=("inning", "max"))
for gid, gg in lu26.groupby("game_pk"):
    hh = gg[gg.team_type == "home"].sort_values("batting_order"); aa = gg[gg.team_type == "away"].sort_values("batting_order")
    r0 = gg.iloc[0]
    if len(hh) != 9 or len(aa) != 9 or gid not in cmeta.index or gid not in g26.index: continue
    if pd.isna(r0.home_starter_id) or pd.isna(r0.away_starter_id): continue
    if hh.player_id.isna().any() or aa.player_id.isna().any(): continue
    gr = g26.loc[gid]; m = cmeta.loc[gid]
    has5 = gid in c5.index
    rows.append(dict(game_pk=int(gid), game_date=pd.Timestamp(gr.game_date), year=2026,
                     home_team=m.hteam, away_team=m.ateam,
                     h_lineup=[int(x) for x in hh.player_id], a_lineup=[int(x) for x in aa.player_id],
                     h_sp=int(r0.home_starter_id), a_sp=int(r0.away_starter_id),
                     h_runs=int(gr.home_score), a_runs=int(gr.away_score),
                     h_f5=int(c5.loc[gid, "h5"]) if has5 else -1, a_f5=int(c5.loc[gid, "a5"]) if has5 else -1,
                     innings=int(m.inn), src="lineups"))
G = pd.DataFrame(rows).sort_values(["game_date", "game_pk"]).reset_index(drop=True)
G.to_parquet(R + "games.parquet", index=False)
print("games", G.shape); print(G.groupby("year").size().to_string())
