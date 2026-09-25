"""Walk-forward backtest of one v2 simulator variant over one season.

    python3 research/backtest_2026/42_run_v2.py VARIANT YEAR [N_EPISODES]

Weekly refit: rates use only PAs strictly before each week's Monday. League constants
that the model borrows (TTO multipliers, temperature slopes, x-outcome table) are
estimated from 2015 ONLY, a season that is never scored. Transitions / hook hazard use
the 2-3 seasons strictly before the scored season.

Writes data/backtest_2026/rich/runs/{VARIANT}_{YEAR}.npz
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
from pa_sim import v2
from pa_sim.v2 import Config, NC

R = "data/backtest_2026/rich/"
OUT = R + "runs/"; os.makedirs(OUT, exist_ok=True)

V1LIKE = dict(tau=np.inf, platoon="bat", home=False, park="scalar", tto=False, hook="curve",
              bullpen="all", trans="fixed", k_hand_mult=3.0, alpha=1.155)
FULL = dict()   # Config defaults: tau 365, both platoon, home, class park, tto, hazard hook, recent pen, empirical
VARIANTS = {
    # --- reference points
    "v1like":      V1LIKE,
    "v1_emp":      {**V1LIKE, "trans": "empirical", "alpha": 1.0},
    # --- defect fixes on top of v1like+empirical transitions
    "d_tau730":    {**V1LIKE, "trans": "empirical", "alpha": 1.0, "tau": 730.0},
    "d_tau365":    {**V1LIKE, "trans": "empirical", "alpha": 1.0, "tau": 365.0},
    "d_tau180":    {**V1LIKE, "trans": "empirical", "alpha": 1.0, "tau": 180.0},
    "d_pen30":     {**V1LIKE, "trans": "empirical", "alpha": 1.0, "tau": 365.0, "bullpen": "recent", "bp_days": 30},
    # --- full v2 and one-at-a-time ablations from it
    "full":        FULL,
    "full_noHome": {"home": False},
    "full_noTTO":  {"tto": False},
    "full_curve":  {"hook": "curve"},
    "full_parkS":  {"park": "scalar"},
    "full_parkN":  {"park": "none"},
    "full_batpl":  {"platoon": "bat"},
    "full_x50":    {"xw": 0.5},
    "full_x100":   {"xw": 1.0},
    "full_wx":     {"weather": True},
    "full_wxc":    {"weather": True},                 # centred version of full_wx (bug fix)
    "full_x50_wxc": {"xw": 0.5, "weather": True},
    "full_pn":     {"park_neutral": True},
    "full_tau730": {"tau": 730.0},
    "full_tau180": {"tau": 180.0},
    "full_pen14":  {"bp_days": 14},
    "full_penall": {"bullpen": "all"},
}


def est_tto(pa15):
    """Class multipliers for a starter's 1st/2nd/3rd+ time through, vs log5 expectation."""
    L = np.bincount(pa15.cls, minlength=NC) / len(pa15)
    def rates(key):
        c = pa15.groupby(key).cls.value_counts().unstack(fill_value=0).reindex(columns=range(NC), fill_value=0)
        n = c.sum(1).values[:, None]
        return pd.DataFrame((c.values + 200 * L) / (n + 200), index=c.index)
    b, p = rates("batter"), rates("pitcher")
    s = pa15[pa15.is_starter & (pa15.n_thruorder_pitcher > 0)]
    e = v2.log5(b.loc[s.batter].values, p.loc[s.pitcher].values, L[None, :])
    t = np.minimum(s.n_thruorder_pitcher.values, 3) - 1
    oh = np.eye(NC)[s.cls.values]
    m = np.ones((3, NC))
    w = np.bincount(t, minlength=3).astype(float)
    for k in range(3):
        m[k] = oh[t == k].sum(0) / e[t == k].sum(0)
    avg = (m * w[:, None]).sum(0) / w.sum()
    return m / avg


def est_temp(pa15, meta):
    """log-rate slope per class per degree F, outdoor games, venue fixed effects (2015)."""
    x = pa15.merge(meta[["game_pk", "venue_id", "temp", "condition", "roof"]], on="game_pk")
    x = x[(x.roof != "Dome") & ~x.condition.isin(["Dome", "Roof Closed"])]
    x["temp"] = pd.to_numeric(x.temp, errors="coerce"); x = x[x.temp.notna()]
    x["dt"] = x.temp - x.groupby("venue_id").temp.transform("mean")
    oh = np.eye(NC)[x.cls.values]
    vm = pd.DataFrame(oh).groupby(x.venue_id.values).transform("mean").values
    sl = np.zeros(NC)
    bins = pd.cut(x.dt, [-60, -15, -7, 0, 7, 15, 60])
    for j in range(NC):
        g = pd.DataFrame(dict(o=oh[:, j], e=vm[:, j], dt=x.dt.values, b=bins.values)).groupby("b")
        agg = g.agg(o=("o", "sum"), e=("e", "sum"), dt=("dt", "mean"), n=("o", "size"))
        lr = np.log(agg.o / agg.e); w = agg.n
        sl[j] = np.sum(w * (agg.dt - np.average(agg.dt, weights=w)) * lr) / np.sum(w * (agg.dt - np.average(agg.dt, weights=w)) ** 2)
    return sl


def load():
    pa = pd.read_parquet(R + "pa_all.parquet")
    meta = pd.read_parquet(R + "sched_meta.parquet")
    games = pd.read_parquet(R + "games.parquet").merge(
        meta[["game_pk", "venue_id", "sched_inn", "temp", "condition", "roof"]], on="game_pk", how="left")
    return pa, meta, games


def main(variant, year, n_ep=2000, pa=None, meta=None, games=None, cfg_over=None, tag=None,
         lineup_mode="actual"):
    t0 = time.time()
    if pa is None:
        pa, meta, games = load()
    pa15 = pa[pa.game_year == 2015]
    venue_of = dict(zip(meta.game_pk, meta.venue_id))
    tmeta = meta.copy(); tmeta["temp"] = pd.to_numeric(tmeta.temp, errors="coerce")
    closed = tmeta.condition.isin(["Dome", "Roof Closed"]) | (tmeta.roof == "Dome")
    temp_of = dict(zip(tmeta.game_pk, np.where(closed, np.nan, tmeta.temp)))
    kw = dict(VARIANTS[variant]) if cfg_over is None else cfg_over
    cfg = Config(**kw, tag=variant)
    xtab = v2.build_xtable(pa15) if cfg.xw > 0 else None
    D = v2.Data(pa[pa.game_date < pd.Timestamp(f"{year}-12-31")], venue_of, temp_of, xtab)
    vh = games.groupby("venue_id").home_team.agg(lambda s: s.value_counts().index[0]).to_dict()
    hook = pd.read_parquet(R + "hook.parquet")
    hook = hook.merge(D.starts[["game_pk", "pitcher", "exp_p"]].rename(columns={"pitcher": "sp"}),
                      on=["game_pk", "sp"], how="left")
    hook["exp_p"] = hook.exp_p.fillna(88.0)
    trans = pd.read_parquet(R + "trans.parquet")
    tto = est_tto(pa15) if cfg.tto else None
    tsl = est_temp(pa15, meta) if cfg.weather else None
    eng = v2.Engine(D, cfg, trans, hook, vh, tto, tsl)
    if cfg.weather and kw.get("temp_centered"):
        pass
    if cfg.weather and variant.endswith("wxc"):
        # centre the temperature effect on the 2015 outdoor mean so it moves totals
        # up in heat and down in cold without shifting the season average
        eng.temp_center = float(np.nanmean([t for g_, t in temp_of.items() if np.isfinite(t)]))
    hand = pa.groupby("pitcher").p_throws.agg(lambda s: int(s.mean() >= 0.5)).to_dict()
    st = pd.DataFrame(dict(b=D.bi, t=D.throws, s=D.stand)).groupby(["b", "t"]).s.mean()
    stand_tab = {k: int(v >= 0.5) for k, v in st.items()}
    eng.prepare(hand, stand_tab)

    g = games[games.year == year].copy()
    if lineup_mode == "prev":
        # lineup unknown at prediction time: use the team's previous game's lineup
        g = g.sort_values(["game_date", "game_pk"])
        last = {}
        hl, al = [], []
        for r in g.itertuples():
            hl.append(last.get(r.home_team, list(r.h_lineup))); al.append(last.get(r.away_team, list(r.a_lineup)))
            last[r.home_team] = list(r.h_lineup); last[r.away_team] = list(r.a_lineup)
        g["h_lineup"] = hl; g["a_lineup"] = al
    g["week"] = g.game_date.dt.to_period("W").dt.start_time
    outs = []
    for wk, gw in g.groupby("week"):
        eng.fit(wk)
        specs = [v2.GameSpec(list(r.h_lineup), list(r.a_lineup), r.h_sp, r.a_sp, r.home_team, r.away_team,
                             int(r.venue_id) if pd.notna(r.venue_id) else -1,
                             int(r.sched_inn) if pd.notna(r.sched_inn) else 9, year >= 2020,
                             temp_of.get(int(r.game_pk), np.nan))
                 for r in gw.itertuples()]
        res = v2.simulate(eng, specs, n=n_ep, seed=int(wk.value // 10**9) % 100000)
        sm = v2.summarize(res, len(specs))
        sm["game_pk"] = gw.game_pk.values
        outs.append(sm)
    keys = outs[0].keys()
    cat = {k: np.concatenate([o[k] for o in outs]) for k in keys}
    np.savez_compressed(OUT + f"{tag or variant}_{year}.npz", **cat)
    print(f"{tag or variant} {year}: {len(cat['game_pk'])} games, {time.time()-t0:.0f}s", flush=True)
    return cat


if __name__ == "__main__":
    lm = sys.argv[4] if len(sys.argv) > 4 else "actual"
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 2000,
         lineup_mode=lm, tag=(sys.argv[1] + "_prevlu") if lm == "prev" else None)
