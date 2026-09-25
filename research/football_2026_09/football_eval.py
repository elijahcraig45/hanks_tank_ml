"""Read-only football model review: walk-forward comparison + collinearity."""
import sys, logging, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss, brier_score_loss

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)
ROOT = Path("/Users/VTNX82W/Documents/personalDev/machineLEARNING/hanks_tank_ml/src")
sys.path.insert(0, str(ROOT / "nfl"))
from data import completed_games  # noqa
from features import build_features, feature_columns  # noqa
from models import build_lr, build_xgb  # noqa
import polars as pl
from epa import EPA_CACHE

SPORT = sys.argv[1] if len(sys.argv) > 1 else "nfl"


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(n=len(y), acc=np.mean((p > .5) == y), ll=log_loss(y, p, labels=[0, 1]),
                brier=brier_score_loss(y, p))


# ---------------- margin ridge rating ----------------
def ridge_margins(games, test_mask_fn, alpha, tau, div_col=None, weeks_per_season=30):
    """For each (season, week) block of test games, fit weighted ridge on margin using
    only games strictly before that block; return predicted home margin."""
    g = games.copy()
    g["t"] = g["season"] * weeks_per_season + g["week"]
    teams = pd.Index(sorted(set(g.home_team) | set(g.away_team)))
    hi = teams.get_indexer(g.home_team); ai = teams.get_indexer(g.away_team)
    n, T = len(g), len(teams)
    X = np.zeros((n, T + 1 + (1 if div_col else 0)))
    X[np.arange(n), hi] = 1; X[np.arange(n), ai] = -1
    X[:, T] = (1 - g["neutral"].values) * 10.0  # HFA, lightly penalized
    if div_col:
        X[:, T + 1] = g[div_col].values * 10.0
    y = g["margin"].values
    pred = np.full(n, np.nan)
    for (s, w), idx in g[test_mask_fn(g)].groupby(["season", "week"]).groups.items():
        t_now = s * weeks_per_season + w
        tr = (g["t"].values < t_now) & (g["t"].values >= t_now - 2 * weeks_per_season) & g["margin"].notna().values
        if tr.sum() < 50:
            continue
        wts = np.exp(-(t_now - g["t"].values[tr]) / tau)
        m = Ridge(alpha=alpha, fit_intercept=False).fit(X[tr], y[tr], sample_weight=wts)
        pred[g.index.get_indexer(idx)] = X[g.index.get_indexer(idx)] @ m.coef_
    return pred


def fit_sigma(margin_pred, y):
    ok = ~np.isnan(margin_pred)
    f = lambda s: log_loss(y[ok], np.clip(norm.cdf(margin_pred[ok] / s), 1e-6, 1 - 1e-6))
    return minimize_scalar(f, bounds=(5, 40), method="bounded").x


def vif_report(X: pd.DataFrame, label):
    X = X.loc[:, X.std() > 0]
    C = np.corrcoef(X.values, rowvar=False)
    inv = np.linalg.pinv(C)
    vif = pd.Series(np.diag(inv), index=X.columns)
    # exact linear dependence -> near-singular
    eig = np.linalg.eigvalsh(C)
    cum = np.cumsum(sorted(eig, reverse=True)) / eig.sum()
    print(f"\n== {label}: {X.shape[1]} features, {len(X)} rows")
    print(f"condition number {np.sqrt(eig.max()/max(eig.min(),1e-12)):.3g}; "
          f"PCs for 90%/95% variance: {np.searchsorted(cum,.9)+1}/{np.searchsorted(cum,.95)+1}; "
          f"eigenvalues<1e-6: {(eig<1e-6).sum()}")
    print(f"VIF>10: {(vif>10).sum()}  VIF>100: {(vif>100).sum()}  VIF>1e6 (exact dup): {(vif>1e6).sum()}")
    print("top VIF:", vif.sort_values(ascending=False).head(12).round(1).to_dict())
    R = pd.DataFrame(C, index=X.columns, columns=X.columns)
    pairs = []
    cols = list(X.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(C[i, j]) >= 0.9:
                pairs.append((cols[i], cols[j], round(C[i, j], 3)))
    print(f"pairs |r|>=0.9: {len(pairs)}")
    for p in sorted(pairs, key=lambda x: -abs(x[2]))[:25]:
        print("  ", p)
    # clusters at |r|>=0.8
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = 1 - np.abs(R.values); np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), "average")
    lab = fcluster(Z, 0.2, "distance")
    cl = pd.Series(lab, index=cols)
    big = [list(v.index) for _, v in cl.groupby(cl) if len(v) > 1]
    print(f"clusters at avg |r|>=0.8: {len(set(lab))} groups from {len(cols)} features")
    for b in sorted(big, key=len, reverse=True)[:10]:
        print("   ", b)
    elo_r = {c: round(R.loc["elo_differential", c], 2) for c in
             ["pythag_differential", "point_diff_differential", "win_pct_diff", "net_epa_8g",
              "points_allowed_differential", "streak_differential"] if c in R}
    print("corr with elo_differential:", elo_r)


def run_nfl():
    games = completed_games()
    epa = pl.read_parquet(EPA_CACHE).to_pandas()
    ep_seasons = sorted(epa.season.unique())
    g_epa = games[games.season.isin(ep_seasons)]
    F_epa = build_features(g_epa, epa=epa)            # the backtested/intended model
    F_noepa = build_features(games, epa=None)         # what production actually runs
    cols_epa = feature_columns(F_epa); cols_no = feature_columns(F_noepa)

    vif_report(F_epa[F_epa.season < 2025][cols_epa].fillna(0), "NFL pure feature set (with EPA)")

    folds = list(range(2017, 2026))  # 2017-2024 walk-forward + 2025 holdout
    res = []
    def wf(F, cols, fn, name):
        out = []
        for s in folds:
            tr, te = F[F.season < s], F[F.season == s]
            m = fn(); m.fit(tr[cols].fillna(0).values, tr.home_won.values)
            p = m.predict_proba(te[cols].fillna(0).values)[:, 1]
            out.append(pd.DataFrame({"game_id": te.game_id.values, "season": s, name: p}))
        return pd.concat(out)
    base = F_epa[F_epa.season.isin(folds)][["game_id", "season", "home_won", "elo_home_win_prob", "spread_line"]].copy()
    small = ["elo_differential", "net_epa_8g", "rest_advantage", "is_neutral_site"]
    for F, cols, fn, name in [
        (F_epa, cols_epa, build_xgb, "xgb_epa (intended)"),
        (F_epa, cols_epa, lambda: build_lr(0.1), "lr_epa"),
        (F_noepa, cols_no, build_xgb, "xgb_noepa (PRODUCTION)"),
        (F_epa, small, lambda: build_lr(1.0), "lr_4feat"),
    ]:
        base = base.merge(wf(F, cols, fn, name)[["game_id", name]], on="game_id", how="left")

    # margin ridge
    rg = games[["game_id", "season", "week", "home_team", "away_team", "result", "location"]].copy()
    rg["margin"] = rg["result"]; rg["neutral"] = (rg["location"] != "Home").astype(int)
    rg = rg.reset_index(drop=True)
    y_all = (rg.result > 0).astype(int).values
    # tune on 2010-2016 only
    best = None
    for alpha in [1, 3, 10, 30]:
        for tau in [6, 10, 16, 25]:
            mp = ridge_margins(rg, lambda g: g.season.between(2010, 2016), alpha, tau)
            ok = ~np.isnan(mp) & rg.season.between(2010, 2016).values
            sig = fit_sigma(mp[ok], y_all[ok])
            ll = log_loss(y_all[ok], np.clip(norm.cdf(mp[ok] / sig), 1e-6, 1 - 1e-6))
            if best is None or ll < best[0]:
                best = (ll, alpha, tau, sig)
    print(f"\nridge tuned on 2010-16: alpha={best[1]} tau={best[2]} sigma={best[3]:.2f} ll={best[0]:.4f}")
    mp = ridge_margins(rg, lambda g: g.season.isin(folds), best[1], best[2])
    rg["ridge_margin"] = mp
    rg["ridge"] = norm.cdf(mp / best[3])
    base = base.merge(rg[["game_id", "ridge", "ridge_margin"]], on="game_id", how="left")
    base["vegas_spread"] = norm.cdf(base.spread_line / 13.45)
    # ridge margin MAE vs spread MAE
    mm = base.merge(rg[["game_id", "result"]], on="game_id")
    print(f"margin MAE 2017-25: ridge {np.nanmean(abs(mm.ridge_margin-mm.result)):.2f}  "
          f"vegas {np.nanmean(abs(mm.spread_line-mm.result)):.2f}  "
          f"ridge ATS (pick side vs spread, pushes excluded): "
          f"{np.mean(((mm.ridge_margin>mm.spread_line)==(mm.result>mm.spread_line))[mm.result!=mm.spread_line]):.3f}")

    report(base, ["elo_home_win_prob", "xgb_epa (intended)", "lr_epa", "xgb_noepa (PRODUCTION)",
                  "lr_4feat", "ridge", "vegas_spread"], "NFL")


def report(base, models, label):
    for period, mask in [("walk-forward 2017-24" if label == "NFL" else "walk-fwd 2023-24", base.season < 2025),
                         ("holdout 2025", base.season == 2025)]:
        b = base[mask].dropna(subset=models)
        print(f"\n{label} {period}  (n={len(b)})")
        print(f"  {'model':26s} {'acc':>6} {'logloss':>8} {'brier':>7}")
        for m in models:
            r = metrics(b.home_won.values, b[m].values)
            print(f"  {m:26s} {r['acc']:6.3f} {r['ll']:8.4f} {r['brier']:7.4f}")
        if "ridge" in b:
            # paired bootstrap of ll difference vs production/xgb
            for ref in [m for m in models if m not in ("ridge","vegas_spread")]:
                y = b.home_won.values
                def ll_i(p): p = np.clip(p, 1e-6, 1 - 1e-6); return -(y*np.log(p)+(1-y)*np.log(1-p))
                d = ll_i(b["ridge"].values) - ll_i(b[ref].values)
                rng = np.random.default_rng(0)
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                print(f"  ridge - {ref}: {d.mean():+.4f} ll (95% CI {np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f})")


def run_cfb():
    sys.path.insert(0, str(ROOT / "cfb"))
    import pipeline as cp
    raw = pd.read_parquet(ROOT.parent / "data/cfb/raw/cfb_games_v2.parquet")
    raw = raw[raw.home_won.notna()]
    feats = cp.build(raw)
    cols = cp.cfb_feature_columns(feats)
    fbs = feats[feats.division == "fbs"]
    vif_report(fbs[fbs.season < 2025][cols].fillna(0), "CFB FBS feature set")

    # production-equivalent weekly backfill, XGB
    outs = []
    for s in (2023, 2024, 2025):
        r = cp.backfill_division(feats, "fbs", s)
        outs.append(r[["game_id", "season", "home_won", "home_win_probability", "elo_home_win_prob"]])
    base = pd.concat(outs).rename(columns={"home_win_probability": "xgb (PRODUCTION)"})

    # margin ridge, single pool, with division covariate
    div_of = cp.resolve_team_divisions(cp.normalize_games(raw))
    rg = raw[["game_id", "season", "week", "home_team", "away_team", "result", "neutral_site", "division"]].copy()
    rg["margin"] = rg.result.clip(-45, 45)  # cap blowouts (analog of MOV damping)
    rg["neutral"] = rg.neutral_site.astype(int)
    rg["hd"] = (rg.home_team.map(div_of) == "fbs").astype(int) - (rg.away_team.map(div_of) == "fbs").astype(int)
    rg = rg.reset_index(drop=True)
    y_all = (rg.result > 0).astype(int).values
    best = None
    tune = lambda g: (g.season == 2022) & (g.division == "fbs")
    for alpha in [1, 3, 10, 30]:
        for tau in [4, 8, 16, 30]:
            mp = ridge_margins(rg, tune, alpha, tau, div_col="hd")
            ok = ~np.isnan(mp) & tune(rg).values
            sig = fit_sigma(mp[ok], y_all[ok])
            ll = log_loss(y_all[ok], np.clip(norm.cdf(mp[ok] / sig), 1e-6, 1 - 1e-6))
            if best is None or ll < best[0]:
                best = (ll, alpha, tau, sig)
    print(f"\nCFB ridge tuned on 2022 FBS: alpha={best[1]} tau={best[2]} sigma={best[3]:.2f} ll={best[0]:.4f}")
    mp = ridge_margins(rg, lambda g: g.season.isin([2023, 2024, 2025]) & (g.division == "fbs"),
                       best[1], best[2], div_col="hd")
    rg["ridge"] = norm.cdf(mp / best[3])
    base = base.merge(rg[["game_id", "ridge"]], on="game_id", how="left")
    report(base, ["elo_home_win_prob", "xgb (PRODUCTION)", "ridge"], "CFB-FBS")
    print("\nnote: ridge covers", base.ridge.notna().mean().round(3), "of FBS test games")


if __name__ == "__main__":
    run_nfl() if SPORT == "nfl" else run_cfb()
