"""Unit-vs-unit matchup modeling vs the plain margin ridge. READ-ONLY (local caches only).

usage: .venv-nfl/bin/python matchup_eval.py nfl|cfb

Protocol reused from football_eval.py: same games, same (season, week) blocks, same
2-season decayed training window, same ridge tuning grid/period, same test folds.
Unit ratings are opponent-adjusted weighted ridges on per-game EPA/play (NFL, nflverse
pbp aggregated by nfl_pbp_agg.py) or PPA/play (CFB, CFBD /stats/game/advanced).
Stage 2 is a logistic regression trained only on earlier seasons' pre-game features.
"""
import sys, json, glob, warnings, logging
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
SP = Path(__file__).resolve().parent
REPO = Path("/Users/VTNX82W/Documents/personalDev/machineLEARNING/hanks_tank_ml")
sys.path.insert(0, str(SP))
from football_eval import ridge_margins, fit_sigma  # exact baseline code  # noqa

SPORT = sys.argv[1] if len(sys.argv) > 1 else "nfl"
WPS = 30


# ---------------------------------------------------------------- data
def load_nfl():
    sys.path.insert(0, str(REPO / "src/nfl"))
    from data import completed_games
    games = completed_games()
    rg = games[["game_id", "season", "week", "home_team", "away_team", "result", "location"]].copy()
    rg["margin"] = rg["result"]; rg["neutral"] = (rg["location"] != "Home").astype(int)
    rg = rg.reset_index(drop=True)
    tg = pd.concat(pd.read_parquet(f) for f in glob.glob(str(SP / "nflagg/tg_*.parquet")))
    # pbp uses LA/LV/LAC for every year; schedules use STL/OAK/SD in old seasons.
    sch = pd.read_parquet(REPO / "data/nfl/raw/schedules.parquet").set_index("game_id")
    tg = tg[tg.game_id.isin(sch.index)].copy()
    def fix(gid, t, o):
        h, a = sch.at[gid, "home_team"], sch.at[gid, "away_team"]
        if t in (h, a): return t
        old = {"LV": "OAK", "LAC": "SD", "LA": "STL"}.get(t)
        if old in (h, a): return old
        if o in (h, a): return a if o == h else h
        raise ValueError((gid, t, o, h, a))
    tg["off"] = [fix(g_, t, o) for g_, t, o in zip(tg.game_id, tg.posteam, tg.defteam)]
    tg["dfn"] = [fix(g_, t, o) for g_, t, o in zip(tg.game_id, tg.defteam, tg.posteam)]
    assert (tg.off != tg.dfn).all()
    tg["rush_epa"] = tg.rush_epa_sum / tg.rush_n.clip(lower=1)
    tg["pass_epa"] = tg.pass_epa_sum / tg.pass_n.clip(lower=1)
    tg["tot_epa"] = (tg.rush_epa_sum + tg.pass_epa_sum) / tg.plays
    tg["rr_num"], tg["rr_den"] = tg.neu_rush_n, tg.neu_n        # neutral-script run rate
    return rg, tg[["game_id", "off", "dfn", "rush_n", "pass_n", "plays", "rush_epa", "pass_epa",
                   "tot_epa", "rr_num", "rr_den"]], None


def load_cfb():
    sys.path.insert(0, str(REPO / "src/cfb"))
    import pipeline as cp
    raw = pd.read_parquet(REPO / "data/cfb/raw/cfb_games_v2.parquet")
    raw = raw[raw.home_won.notna()]
    div_of = cp.resolve_team_divisions(cp.normalize_games(raw))
    rg = raw[["game_id", "season", "week", "home_team", "away_team", "result", "neutral_site", "division"]].copy()
    rg["margin"] = rg.result.clip(-45, 45)
    rg["neutral"] = rg.neutral_site.astype(int)
    rg["hd"] = (rg.home_team.map(div_of) == "fbs").astype(int) - (rg.away_team.map(div_of) == "fbs").astype(int)
    rg = rg.reset_index(drop=True)
    gi = rg.set_index("game_id")
    recs = []
    for y in range(2021, 2026):
        for x in json.load(open(SP / f"cfbd/adv_{y}.json")):
            recs.append(dict(game_id=str(x["gameId"]), season=y, team=x["team"], opp=x["opponent"],
                             o=x["offense"], d=x["defense"]))
    df = pd.DataFrame(recs)
    df = df[df.game_id.isin(gi.index)]
    # per-season CFBD name -> ESPN abbreviation, by intersecting each name's game pairs
    name_map = {}
    for (s, t), grp in df.groupby(["season", "team"]):
        sets = [set(gi.loc[g, ["home_team", "away_team"]]) for g in grp.game_id]
        common = set.intersection(*sets)
        if len(common) == 1:
            name_map[(s, t)] = common.pop()
    df["off"] = [name_map.get((s, t)) for s, t in zip(df.season, df.team)]
    df = df[df.off.notna()]
    df["dfn"] = [(set(gi.loc[g, ["home_team", "away_team"]]) - {o}).pop()
                 if len(set(gi.loc[g, ["home_team", "away_team"]]) - {o}) == 1 else None
                 for g, o in zip(df.game_id, df.off)]
    df = df[df.dfn.notna()]
    def n_of(u):  # plays = totalPPA / ppa
        return (u["totalPPA"] / u["ppa"]) if u and u.get("ppa") else np.nan
    df["rush_n"] = [n_of(o.get("rushingPlays")) for o in df.o]
    df["pass_n"] = [n_of(o.get("passingPlays")) for o in df.o]
    df["plays"] = [o.get("plays") for o in df.o]
    df["rush_epa"] = [o["rushingPlays"]["ppa"] if o.get("rushingPlays") else np.nan for o in df.o]
    df["pass_epa"] = [o["passingPlays"]["ppa"] if o.get("passingPlays") else np.nan for o in df.o]
    df["tot_epa"] = [o.get("ppa") for o in df.o]
    df = df.dropna(subset=["rush_n", "pass_n", "rush_epa", "pass_epa", "tot_epa"])
    df = df[(df.rush_n > 0) & (df.pass_n > 0)]
    df["rr_num"], df["rr_den"] = df.rush_n, df.rush_n + df.pass_n    # raw run share (no script adj)
    df = df.drop_duplicates(["game_id", "off"])
    return rg, df[["game_id", "off", "dfn", "rush_n", "pass_n", "plays", "rush_epa", "pass_epa",
                   "tot_epa", "rr_num", "rr_den"]], "hd"


# ---------------------------------------------------------------- unit ratings
def unit_features(rg, tg, blocks_mask, cfg, div_col):
    """For each (season, week) block, fit opponent-adjusted unit ridges on strictly earlier
    team-games (2-season window, exp decay) and emit pre-game features for the block."""
    g = rg.copy(); g["t"] = g.season * WPS + g.week
    tg = tg.merge(g[["game_id", "t", "home_team", "away_team", "neutral"]], on="game_id")
    tg["side"] = np.where(tg.neutral == 1, 0, np.where(tg.off == tg.home_team, 1, -1))
    teams = pd.Index(sorted(set(g.home_team) | set(g.away_team) | set(tg.off) | set(tg.dfn)))
    T = len(teams)
    oi, di = teams.get_indexer(tg.off), teams.get_indexer(tg.dfn)
    n = len(tg)
    X = np.zeros((n, 2 * T + 1))
    X[np.arange(n), oi] = 1; X[np.arange(n), T + di] = 1; X[:, 2 * T] = tg.side.values
    tvals = tg.t.values
    out = []
    for (s, w), idx in g[blocks_mask(g)].groupby(["season", "week"]).groups.items():
        t_now = s * WPS + w
        win = (tvals < t_now) & (tvals >= t_now - 2 * WPS)
        if win.sum() < 100:
            continue
        rows = {}
        for unit, ncol in (("rush", "rush_n"), ("pass", "pass_n"), ("tot", "plays")):
            a, tau = cfg[unit]
            wts = np.exp(-(t_now - tvals[win]) / tau) * tg[ncol].values[win] / tg[ncol].mean()
            m = Ridge(alpha=a).fit(X[win], tg[f"{unit}_epa"].values[win], sample_weight=wts)
            rows[unit] = (m.coef_[:T], m.coef_[T:2 * T], m.coef_[2 * T], m.intercept_)
        # run rate: decayed, shrunk to league mean with k pseudo-plays
        wts = np.exp(-(t_now - tvals[win]) / cfg["rr_tau"])
        num = np.bincount(oi[win], wts * tg.rr_num.values[win], T)
        den = np.bincount(oi[win], wts * tg.rr_den.values[win], T)
        lg = num.sum() / den.sum(); k = cfg["rr_k"]
        rr = (num + k * lg) / (den + k)
        gg = g.loc[idx]
        h, a_ = teams.get_indexer(gg.home_team), teams.get_indexer(gg.away_team)
        side = 1 - gg.neutral.values
        f = pd.DataFrame({"game_id": gg.game_id.values})
        for unit, (O, D, hfa, c) in rows.items():
            f[f"{unit}_h"] = c + O[h] + D[a_] + hfa * side
            f[f"{unit}_a"] = c + O[a_] + D[h] - hfa * side
            f[f"O{unit}_h"], f[f"O{unit}_a"] = O[h], O[a_]
            f[f"D{unit}_h"], f[f"D{unit}_a"] = D[h], D[a_]
        f["rr_h"], f["rr_a"], f["rr_lg"] = rr[h], rr[a_], lg
        out.append(f)
    return pd.concat(out, ignore_index=True)


def add_derived(F):
    F = F.copy()
    for s, o in (("h", "a"), ("a", "h")):
        F[f"mix_{s}"] = F[f"rr_{s}"] * F[f"rush_{s}"] + (1 - F[f"rr_{s}"]) * F[f"pass_{s}"]
        F[f"rrdev_{s}"] = F[f"rr_{s}"] - F.rr_lg
        # "run-heavy offense vs a defense relatively weak against the run"
        F[f"style_{s}"] = F[f"rrdev_{s}"] * (F[f"Drush_{o}"] - F[f"Dpass_{o}"])
        F[f"xr_{s}"] = F[f"Orush_{s}"] * F[f"Drush_{o}"]
        F[f"xp_{s}"] = F[f"Opass_{s}"] * F[f"Dpass_{o}"]
    F["tot_d"] = F.tot_h - F.tot_a
    F["mix_d"] = F.mix_h - F.mix_a
    return F


def tune_units(rg, tg, tune_mask, div_col, grid_a, grid_tau):
    """Pick alpha/tau per unit by next-block unit prediction MSE on the tuning period."""
    cfg, best = {}, {}
    for a in grid_a:
        for tau in grid_tau:
            c = {u: (a, tau) for u in ("rush", "pass", "tot")}; c.update(rr_tau=tau, rr_k=100)
            F = unit_features(rg, tg, tune_mask, c, div_col)
            m = tg.merge(rg[["game_id", "home_team"]], on="game_id").merge(F, on="game_id")
            for unit in ("rush", "pass", "tot"):
                pred = np.where(m.off == m.home_team, m[f"{unit}_h"], m[f"{unit}_a"])
                ncol = {"rush": "rush_n", "pass": "pass_n", "tot": "plays"}[unit]
                mse = np.average((m[f"{unit}_epa"] - pred) ** 2, weights=m[ncol])
                if unit not in best or mse < best[unit][0]:
                    best[unit] = (mse, a, tau)
    for unit in ("rush", "pass", "tot"):
        cfg[unit] = best[unit][1:]
        print(f"  unit {unit}: alpha={best[unit][1]} tau={best[unit][2]} wMSE={best[unit][0]:.5f}")
    cfg["rr_tau"] = cfg["tot"][1]; cfg["rr_k"] = 100
    return cfg


# ---------------------------------------------------------------- stage 2 + reporting
def ll_i(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(d, reps=2000):
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    if SPORT == "nfl":
        rg, tg, div_col = load_nfl()
        tune_mask = lambda g: g.season.between(2010, 2016)
        test_seasons = list(range(2017, 2026)); feat_mask = lambda g: g.season.between(2010, 2025)
        stage2_first = 2010; eval_div = None
        ga, gt = [3, 10, 30, 100], [6, 10, 16, 25, 40]
        r_grid = ([1, 3, 10, 30], [6, 10, 16, 25])
    else:
        rg, tg, div_col = load_cfb()
        tune_mask = lambda g: (g.season == 2022) & (g.division == "fbs")
        test_seasons = [2023, 2024, 2025]
        feat_mask = lambda g: g.season.between(2022, 2025) & (g.division == "fbs")
        stage2_first = 2022; eval_div = "fbs"
        ga, gt = [3, 10, 30, 100], [4, 8, 16, 30, 60]
        r_grid = ([1, 3, 10, 30], [4, 8, 16, 30])
    y_all = (rg.result > 0).astype(int).values
    print(f"{SPORT}: {len(rg)} games, {len(tg)} team-game unit rows")

    # A) baseline, exactly as football_eval.py
    best = None
    for alpha in r_grid[0]:
        for tau in r_grid[1]:
            mp = ridge_margins(rg, tune_mask, alpha, tau, div_col=div_col)
            ok = ~np.isnan(mp) & tune_mask(rg).values
            sig = fit_sigma(mp[ok], y_all[ok])
            ll = log_loss(y_all[ok], np.clip(norm.cdf(mp[ok] / sig), 1e-6, 1 - 1e-6))
            if best is None or ll < best[0]:
                best = (ll, alpha, tau, sig)
    print(f"ridge tuned: alpha={best[1]} tau={best[2]} sigma={best[3]:.2f}")
    mp = ridge_margins(rg, feat_mask, best[1], best[2], div_col=div_col)
    rg["ridge_margin"] = mp; rg["A"] = norm.cdf(mp / best[3])

    print("tuning unit ridges on the ridge tuning period...")
    cfg = tune_units(rg, tg, tune_mask, div_col, ga, gt)
    F = add_derived(unit_features(rg, tg, feat_mask, cfg, div_col))
    D = rg.merge(F, on="game_id")
    D["home"] = 1 - D.neutral
    D["y"] = (D.result > 0).astype(int)
    D = D.dropna(subset=["ridge_margin"])
    extra = [div_col] if div_col else []
    B0 = ["tot_h", "tot_a", "home"] + extra
    B = ["rush_h", "pass_h", "rush_a", "pass_a", "home"] + extra
    specs = {
        "A_lr (ridge margin, LR)": ["ridge_margin"],
        "B0 scalar EPA units": B0,
        "B split rush/pass additive": B,
        "C1 mix-weighted": ["mix_h", "mix_a", "home"] + extra,
        "C2 B + mix + run rates": B + ["mix_h", "mix_a", "rr_h", "rr_a"],
        "C3 B + products + style": B + ["xr_h", "xr_a", "xp_h", "xp_a", "style_h", "style_a"],
        "D0 ridge + scalar EPA": ["ridge_margin", "tot_d"],
        "D ridge + split (B)": ["ridge_margin"] + B,
        "D2 ridge + C3": ["ridge_margin"] + B + ["xr_h", "xr_a", "xp_h", "xp_a", "style_h", "style_a"],
        "D3 ridge + style only": ["ridge_margin", "style_h", "style_a"],
    }
    preds = []
    for s in test_seasons:
        tr = D[(D.season >= stage2_first) & (D.season < s)]
        te = D[D.season == s]
        if eval_div:
            tr = tr[tr.division == eval_div]; te = te[te.division == eval_div]
        p = te[["game_id", "season", "y", "A"]].copy()
        for name, cols in specs.items():
            for C, tag in ((1.0, ""), (0.05, " [C=.05]")):
                if tag and not name.startswith(("C3", "D2", "D3")):
                    continue
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
                m.fit(tr[cols].values, tr.y.values)
                p[name + tag] = m.predict_proba(te[cols].values)[:, 1]
        preds.append(p)
    P = pd.concat(preds).dropna()
    models = ["A"] + [c for c in P.columns if c not in ("game_id", "season", "y", "A")]
    wf_label = "wf 2017-24" if SPORT == "nfl" else "wf 2023-24"
    print(f"\n{SPORT.upper()} results (diff = model - A log loss; negative is better)")
    for period, mask in ((wf_label, P.season < 2025), ("holdout 2025", P.season == 2025)):
        b = P[mask]; y = b.y.values
        print(f"\n{period} n={len(b)}")
        print(f"  {'model':34s} {'acc':>6} {'ll':>7} {'d_vs_A':>8}  95% CI")
        la = ll_i(y, b.A.values)
        for mname in models:
            li = ll_i(y, b[mname].values); d = li - la
            lo, hi = boot(d) if mname != "A" else (0, 0)
            print(f"  {mname:34s} {np.mean((b[mname]>.5)==y):6.3f} {li.mean():7.4f} {d.mean():+8.4f}  ({lo:+.4f},{hi:+.4f})")
    print("\nper-fold log loss (A / B0 / B / C3 / D / D2):")
    for s, b in P.groupby("season"):
        y = b.y.values
        print(f"  {s}: n={len(b):4d} " + " ".join(f"{ll_i(y, b[m].values).mean():.4f}" for m in
              ["A", "B0 scalar EPA units", "B split rush/pass additive", "C3 B + products + style",
               "D ridge + split (B)", "D2 ridge + C3"]))

    # ---- style spread diagnostics (pre-game values in test seasons)
    Dt = D[D.season.isin(test_seasons)]
    if eval_div: Dt = Dt[Dt.division == eval_div]
    rr = pd.concat([Dt.rr_h, Dt.rr_a]); Or = pd.concat([Dt.Orush_h, Dt.Orush_a]); Op = pd.concat([Dt.Opass_h, Dt.Opass_a])
    Dr = pd.concat([Dt.Drush_h, Dt.Drush_a]); Dp = pd.concat([Dt.Dpass_h, Dt.Dpass_a])
    print("\nstyle spread (pre-game, test seasons, per team-game):")
    print(f"  run rate: mean {rr.mean():.3f} sd {rr.std():.3f} p10-p90 {rr.quantile(.1):.3f}-{rr.quantile(.9):.3f}")
    print(f"  offense rating sd: rush {Or.std():.4f} pass {Op.std():.4f}  corr(rush,pass) {np.corrcoef(Or,Op)[0,1]:.2f}")
    print(f"  defense rating sd: rush {Dr.std():.4f} pass {Dp.std():.4f}  corr(rush,pass) {np.corrcoef(Dr,Dp)[0,1]:.2f}")
    print(f"  style term sd {pd.concat([Dt.style_h, Dt.style_a]).std():.5f}; "
          f"mix vs scalar per-play gap sd {pd.concat([Dt.mix_h-Dt.tot_h, Dt.mix_a-Dt.tot_a]).std():.4f}")
    # how many points is one sd of the matchup-only signal? scale via margin ~ tot_d regression
    k = np.polyfit(Dt.tot_d, Dt.result, 1)[0]
    print(f"  margin per unit of net EPA/play (OLS): {k:.1f} pts; mix-minus-scalar net gap sd "
          f"= {k*((Dt.mix_d)-(Dt.tot_d)).std():.2f} pts")

    # ---- unit-level test of the hypothesis: does the offense's actual per-play output depend on
    # style x opponent weakness beyond additive (offense + defense) expectations?
    U = tg.merge(D[["game_id", "season", "home_team"] + [c for c in D.columns if c[-2:] in ("_h", "_a")]], on="game_id")
    U = U[U.season.isin(test_seasons)]
    hs = U.off == U.home_team
    pick = lambda c: np.where(hs, U[f"{c}_h"], U[f"{c}_a"])
    U["add_tot"] = pick("tot"); U["mixp"] = pick("mix"); U["style"] = pick("style")
    U["resid"] = U.tot_epa - U.add_tot
    w = U.plays.values
    mse_add = np.average((U.tot_epa - U.add_tot) ** 2, weights=w)
    mse_mix = np.average((U.tot_epa - U.mixp) ** 2, weights=w)
    Xs = np.c_[np.ones(len(U)), U["style"].values]
    beta, *_ = np.linalg.lstsq(Xs * np.sqrt(w)[:, None], U.resid.values * np.sqrt(w), rcond=None)
    r = U.resid.values - Xs @ beta
    se = np.sqrt(np.average(r ** 2, weights=w) / (np.var(U["style"]) * len(U)))
    print(f"\nunit level (test seasons, {len(U)} team-games): wMSE additive-scalar {mse_add:.5f}  "
          f"mix-weighted split {mse_mix:.5f}  ({(1-mse_mix/mse_add)*100:+.2f}% )")
    print(f"  resid ~ style: beta={beta[1]:.3f} (se~{se:.3f}, t={beta[1]/se:.2f})")
    P.to_parquet(SP / f"matchup_preds_{SPORT}.parquet")


if __name__ == "__main__":
    main()
