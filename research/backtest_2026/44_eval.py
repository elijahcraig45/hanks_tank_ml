"""Score v2 simulator runs against outcomes and benchmarks, with day-block bootstrap CIs.

    python3 research/backtest_2026/44_eval.py VARIANT[,VARIANT...] YEARS  [--ref VARIANT]

YEARS like 2016-2019. Calibration (Platt on logit p) is always fit on EARLIER seasons of
the same variant only, so calibrated metrics start at the second season listed.
Metrics are per-game losses so every comparison is paired, and the bootstrap resamples
whole dates (games on one date share weather/scheduling shocks).
"""
import sys, numpy as np, pandas as pd, warnings
from scipy.stats import nbinom, poisson
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
R = "data/backtest_2026/rich/"
EPS = 1e-4


def lg(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(delta, dates, n=2000, seed=0):
    """Mean of per-game delta with a date-block bootstrap 95% CI and P(delta>0)."""
    d = pd.DataFrame(dict(x=delta, d=dates)).dropna()
    grp = d.groupby("d").x.agg(["sum", "count"])
    s, c = grp["sum"].values, grp["count"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n, len(s)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())


def crps_discrete(pmf, y):
    """CRPS for an integer outcome: sum_k (F(k) - 1{y<=k})^2."""
    F = np.cumsum(pmf, 1)
    k = np.arange(pmf.shape[1])[None, :]
    return ((F - (k >= y[:, None])) ** 2).sum(1)


def load_runs(variant, years):
    out = []
    for y in years:
        z = np.load(R + f"runs/{variant}_{y}.npz")
        d = {k: z[k] for k in z.files}
        d["year"] = np.full(len(d["game_pk"]), y)
        out.append(d)
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def frame(variant, years):
    z = load_runs(variant, years)
    B = pd.read_parquet(R + "bench.parquet").set_index("game_pk")
    df = B.loc[z["game_pk"]].reset_index()
    df["sim_p"] = z["p_home"]
    return df, z


def platt_by_season(df, col, years):
    out = np.full(len(df), np.nan)
    for y in years:
        tr = (df.year < y).values; te = (df.year == y).values
        if tr.sum() < 500: continue
        m = LogisticRegression(C=1e4).fit(lg(df[col].values[tr]).reshape(-1, 1), df.y.values[tr])
        out[te] = m.predict_proba(lg(df[col].values[te]).reshape(-1, 1))[:, 1]
    return out


def stack_by_season(df, cols, years):
    out = np.full(len(df), np.nan)
    for y in years:
        tr = (df.year < y).values & df[cols].notna().all(1).values
        te = (df.year == y).values & df[cols].notna().all(1).values
        if tr.sum() < 500 or te.sum() == 0: continue
        X = np.column_stack([lg(df[c].values) for c in cols])
        m = LogisticRegression(C=1e4).fit(X[tr], df.y.values[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def win_metrics(df, years):
    df = df.copy()
    df["sim_cal"] = platt_by_season(df, "sim_p", years)
    if df.mkt_p.notna().any():
        df["stack_mkt"] = stack_by_season(df, ["mkt_p", "sim_p"], years)
        df["mkt_only"] = stack_by_season(df, ["mkt_p"], years)
    return df


def totals_metrics(df, z):
    """Per-game log score and CRPS for the sim's total-runs pmf and the baselines."""
    y = df.tot.values.astype(int)
    th = z["tot_hist"]; K = th.shape[1]
    yc = np.clip(y, 0, K - 1)
    pmf_s = th * 0.999 + 0.001 / K                           # floor so log score is finite
    df["ls_sim"] = -np.log(pmf_s[np.arange(len(y)), yc])
    df["crps_sim"] = crps_discrete(th, y)
    df["mean_sim"] = (th * np.arange(K)).sum(1)
    k = np.arange(K)
    def nb_pmf(mu, r):
        p = r / (r + mu)
        pm = nbinom.pmf(k[None, :], r[:, None], p[:, None]); return pm / pm.sum(1, keepdims=True)
    if df.tot_mu_base.notna().all():
        pm = nb_pmf(df.tot_mu_base.values, df.nb_r_base.values)
        df["ls_base"] = -np.log(pm[np.arange(len(y)), yc] * 0.999 + 0.001 / K)
        df["crps_base"] = crps_discrete(pm, y)
    # market-line NB: mean = a + b*line fit on earlier seasons in this frame
    df["ls_mkt"] = np.nan; df["crps_mkt"] = np.nan; df["mu_mkt"] = np.nan
    for yr in sorted(df.year.unique()):
        tr = (df.year < yr) & df.total_line.notna(); te = (df.year == yr) & df.total_line.notna()
        if tr.sum() < 500 or te.sum() == 0: continue
        b = np.polyfit(df.total_line[tr], df.tot[tr], 1)
        mu_tr = np.polyval(b, df.total_line[tr])
        r = float(np.sum(mu_tr ** 2) / max(np.sum((df.tot[tr] - mu_tr) ** 2 - mu_tr), 1e-6))
        mu = np.polyval(b, df.total_line[te].values)
        pm = nb_pmf(mu, np.full(te.sum(), r))
        yy = np.clip(df.tot[te].values.astype(int), 0, K - 1)
        df.loc[te, "ls_mkt"] = -np.log(pm[np.arange(te.sum()), yy] * 0.999 + 0.001 / K)
        df.loc[te, "crps_mkt"] = crps_discrete(pm, df.tot[te].values.astype(int))
        df.loc[te, "mu_mkt"] = mu
    # over/under vs the closing total (pushes dropped)
    L = df.total_line.values
    ok = np.isfinite(L) & (df.tot.values != L)
    F = np.cumsum(th, 1)
    Li = np.where(np.isfinite(L), np.floor(L).astype(int), 0).clip(0, K - 1)
    p_le = F[np.arange(len(L)), Li]
    push = np.where(np.isfinite(L) & (L == np.floor(L)), th[np.arange(len(L)), Li], 0.0)
    p_over = (1 - p_le) / np.clip(1 - push, 1e-6, None)
    df["sim_over"] = np.where(ok, p_over, np.nan)
    df["over_y"] = np.where(ok, (df.tot.values > L).astype(float), np.nan)
    return df


def f5_metrics(df, z):
    f = z["f5"]                                      # [away lead, tie, home lead]
    ok = (df.h_f5.values >= 0)
    lab = np.where(df.h_f5 > df.a_f5, 2, np.where(df.h_f5 == df.a_f5, 1, 0))
    df["f5_lab"] = np.where(ok, lab, -1)
    pf = np.clip(f, 1e-4, 1); pf = pf / pf.sum(1, keepdims=True)
    df["f5_ls_sim"] = np.where(ok, -np.log(pf[np.arange(len(df)), np.clip(lab, 0, 2)]), np.nan)
    # baseline: multinomial logistic on logit(market full-game p), fit on earlier seasons
    df["f5_ls_mkt"] = np.nan; df["f5_ls_clim"] = np.nan; df["f5_ls_simcal"] = np.nan
    for yr in sorted(df.year.unique()):
        tr = ((df.year < yr) & ok).values; te = ((df.year == yr) & ok).values
        if tr.sum() < 500: continue
        fr = np.bincount(lab[tr], minlength=3) / tr.sum()
        df.loc[te, "f5_ls_clim"] = -np.log(fr[lab[te]])
        X = lg(np.where(df.mkt_p.notna(), df.mkt_p, df.strength_p).astype(float)).reshape(-1, 1)
        okm = np.isfinite(X[:, 0])
        m = LogisticRegression(C=1e4, max_iter=2000).fit(X[tr & okm], lab[tr & okm])
        pr = m.predict_proba(X[te & okm])
        df.loc[te & okm, "f5_ls_mkt"] = -np.log(pr[np.arange(pr.shape[0]), lab[te & okm]])
        Xs = np.log(pf)
        m2 = LogisticRegression(C=1e4, max_iter=2000).fit(Xs[tr], lab[tr])
        pr2 = m2.predict_proba(Xs[te])
        df.loc[te, "f5_ls_simcal"] = -np.log(pr2[np.arange(pr2.shape[0]), lab[te]])
    return df


def summarize_variant(variant, years, ref=None, verbose=True):
    df, z = frame(variant, years)
    df = win_metrics(df, years)
    df = totals_metrics(df, z)
    df = f5_metrics(df, z)
    return df, z


def report(variant, years):
    df, z = summarize_variant(variant, years)
    y = df.y.values
    cal = np.isfinite(df.sim_cal)
    r = {}
    r["n"] = len(df); r["n_cal"] = int(cal.sum())
    r["ll_raw"] = ll(y, df.sim_p).mean()
    r["ll_cal"] = ll(y[cal], df.sim_cal[cal]).mean()
    m = cal & df.mkt_p.notna().values
    if m.any():
        r["ll_mkt"] = ll(y[m], df.mkt_p[m]).mean(); r["ll_cal_m"] = ll(y[m], df.sim_cal[m]).mean()
    r["ll_strength"] = ll(y[cal], df.strength_p[cal]).mean()
    r["bias_tot"] = (df.mean_sim - df.tot).mean()
    r["crps_sim"] = df.crps_sim.mean(); r["ls_sim"] = df.ls_sim.mean()
    return r, df


if __name__ == "__main__":
    vs = sys.argv[1].split(","); a, b = sys.argv[2].split("-"); years = list(range(int(a), int(b) + 1))
    rows = []
    for v in vs:
        r, _ = report(v, years); r["variant"] = v; rows.append(r)
    print(pd.DataFrame(rows).set_index("variant").round(5).to_string())


def tilt(pmf, target):
    """Exponentially tilt each row of pmf (p_k * e^{theta k}) so its mean equals target."""
    from scipy.optimize import brentq
    k = np.arange(pmf.shape[1]); out = np.empty_like(pmf)
    for i in range(len(pmf)):
        p = pmf[i] + 1e-12
        if not np.isfinite(target[i]):
            out[i] = pmf[i]; continue
        f = lambda th: (p * np.exp(th * k) * k).sum() / (p * np.exp(th * k)).sum() - target[i]
        try:
            th = brentq(f, -1.5, 1.5)
        except ValueError:
            th = 0.0
        q = p * np.exp(th * k); out[i] = q / q.sum()
    return out


def totals_calibrated(df, z):
    """Sim total pmf tilted (a) to remove the bias seen in EARLIER seasons, (b) to the
    market-implied mean (a + b*line fit on earlier seasons). Adds per-game CRPS/log score."""
    th = z["tot_hist"]; K = th.shape[1]; y = df.tot.values.astype(int); yc = np.clip(y, 0, K - 1)
    for c in ("crps_simcal", "ls_simcal", "crps_simmkt", "ls_simmkt"):
        df[c] = np.nan
    for yr in sorted(df.year.unique()):
        tr = (df.year < yr).values; te = (df.year == yr).values
        if tr.sum() < 500: continue
        bias = float((df.mean_sim[tr] - df.tot[tr]).mean())
        q = tilt(th[te], df.mean_sim.values[te] - bias)
        df.loc[te, "crps_simcal"] = crps_discrete(q, y[te])
        df.loc[te, "ls_simcal"] = -np.log(q[np.arange(te.sum()), yc[te]] * 0.999 + 0.001 / K)
        if "mu_mkt" in df and df.mu_mkt[te].notna().any():
            tm = te & df.mu_mkt.notna().values
            q2 = tilt(th[tm], df.mu_mkt.values[tm])
            df.loc[tm, "crps_simmkt"] = crps_discrete(q2, y[tm])
            df.loc[tm, "ls_simmkt"] = -np.log(q2[np.arange(tm.sum()), yc[tm]] * 0.999 + 0.001 / K)
    return df
