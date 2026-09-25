"""Player-prop and series scoring for v2 simulator runs.

Props (sim pmf vs naive baselines, per game):
  starter strikeouts   baseline Poisson(K% x expected BF); K% = trailing-365-day rate shrunk
                       to league (150 BF), expected BF = shrunk mean of last 10 starts
  batter hits >= 1     baseline 1-(1-h)^n, h = trailing-365 H/PA shrunk (200 PA),
  batter HR   >= 1     n = league mean PAs for that lineup slot (prior season)
  batter K    >= 1
Series (>=3 games, all games inside one weekly fit so game 1's result cannot leak):
  sim     P(home wins majority) from each game's sim p, assuming independence
  market  logistic on logit(game-1 closing p), fit on earlier seasons
"""
import sys, importlib, numpy as np, pandas as pd, warnings
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
sys.path.insert(0, "research/backtest_2026")
E = importlib.import_module("44_eval")
R = "data/backtest_2026/rich/"


def _trailing(pa, key, num_mask, den_mask=None):
    """Per (player, date): trailing-365-day sums strictly before that date."""
    d = pa.assign(num=num_mask.astype(float), den=(np.ones(len(pa)) if den_mask is None else den_mask.astype(float)))
    daily = d.groupby([key, "game_date"])[["num", "den"]].sum().reset_index().sort_values([key, "game_date"])
    out = []
    for pid, g in daily.groupby(key, sort=False):
        t = g.game_date.values.astype("datetime64[D]").astype(np.int64)
        cn, cd = np.cumsum(g.num.values), np.cumsum(g.den.values)
        lo = np.searchsorted(t, t - 365, side="left")          # window start
        # strictly before the date: exclude today's row (index i) -> use i-1
        i = np.arange(len(t))
        n_before = np.where(i > 0, cn[i - 1], 0) - np.where(lo > 0, cn[lo - 1], 0)
        d_before = np.where(i > 0, cd[i - 1], 0) - np.where(lo > 0, cd[lo - 1], 0)
        out.append(pd.DataFrame({key: pid, "game_date": g.game_date.values, "num": n_before, "den": d_before}))
    return pd.concat(out, ignore_index=True)


def prop_frame(years):
    pa = pd.read_parquet(R + "pa_all.parquet", columns=["game_pk", "game_date", "game_year", "batter", "pitcher",
                                                        "cls", "is_starter", "top"])
    pa = pa[pa.game_year.between(min(years) - 1, max(years))]
    G = pd.read_parquet(R + "games.parquet")
    G = G[G.year.isin(years)]
    # actual starter Ks and BF
    sp = pa[pa.is_starter].groupby(["game_pk", "pitcher"]).agg(k=("cls", lambda s: (s == 0).sum()), bf=("cls", "size")).reset_index()
    # baselines
    lgK = (pa.cls == 0).mean()
    kr = _trailing(pa, "pitcher", pa.cls == 0)
    kr["kpct"] = (kr.num + 150 * lgK) / (kr.den + 150)
    st = sp.merge(pa.groupby("game_pk").game_date.first().reset_index(), on="game_pk").sort_values("game_date")
    st["bf_prev"] = st.groupby("pitcher").bf.transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    st["n_prev"] = st.groupby("pitcher").bf.transform(lambda x: x.shift(1).rolling(10, min_periods=1).count())
    lgbf = st.bf.mean()
    st["ebf"] = (st.bf_prev.fillna(lgbf) * st.n_prev.fillna(0) + 3 * lgbf) / (st.n_prev.fillna(0) + 3)
    st = st.merge(kr[["pitcher", "game_date", "kpct"]], on=["pitcher", "game_date"], how="left")
    st["kpct"] = st.kpct.fillna(lgK)
    st["k_mu_base"] = st.kpct * st.ebf
    # batters
    hit = pa.cls.isin([2, 3, 4, 5])
    lgH, lgHR, lgK_b = hit.mean(), (pa.cls == 5).mean(), (pa.cls == 0).mean()
    th = _trailing(pa, "batter", hit); thr = _trailing(pa, "batter", pa.cls == 5); tk = _trailing(pa, "batter", pa.cls == 0)
    bt = th.rename(columns={"num": "h", "den": "n"}).merge(thr[["batter", "game_date", "num"]].rename(columns={"num": "hr"}),
                                                           on=["batter", "game_date"]).merge(
        tk[["batter", "game_date", "num"]].rename(columns={"num": "kk"}), on=["batter", "game_date"])
    bt["h_rate"] = (bt.h + 200 * lgH) / (bt.n + 200); bt["hr_rate"] = (bt.hr + 200 * lgHR) / (bt.n + 200)
    bt["k_rate"] = (bt.kk + 200 * lgK_b) / (bt.n + 200)
    bg = pa.groupby(["game_pk", "batter"]).agg(H=("cls", lambda s: s.isin([2, 3, 4, 5]).sum()),
                                               HR=("cls", lambda s: (s == 5).sum()),
                                               K=("cls", lambda s: (s == 0).sum()), PA=("cls", "size")).reset_index()
    return G, st, bt, bg


def score_props(variant, years):
    G, st, bt, bg = prop_frame(years)
    z = E.load_runs(variant, years)
    idx = {g: i for i, g in enumerate(z["game_pk"])}
    rows_k, rows_b = [], []
    stx = st.set_index(["game_pk", "pitcher"])
    btx = bt.set_index(["batter", "game_date"])
    bgx = bg.set_index(["game_pk", "batter"])
    # league PA per slot from the first season listed minus one (prior info)
    slot_pa = np.array([4.65, 4.55, 4.45, 4.35, 4.25, 4.12, 4.0, 3.88, 3.75])
    for r in G.itertuples():
        i = idx.get(r.game_pk)
        if i is None: continue
        for side, spid in ((0, r.a_sp), (1, r.h_sp)):       # spk[:,0] = away SP
            if (r.game_pk, spid) not in stx.index: continue
            a = stx.loc[(r.game_pk, spid)]
            pm = z["spk"][i, side]
            k = int(min(a.k, 20))
            rows_k.append(dict(game_pk=r.game_pk, date=r.game_date, year=r.year, k=a.k,
                               ls_sim=-np.log(pm[k] * 0.999 + 0.001 / 21), mu_sim=(pm * np.arange(21)).sum(),
                               crps_sim=E.crps_discrete(pm[None], np.array([k]))[0],
                               ls_base=-np.log(poisson.pmf(k, a.k_mu_base) * 0.999 + 0.001 / 21), mu_base=a.k_mu_base,
                               crps_base=E.crps_discrete(poisson.pmf(np.arange(21), a.k_mu_base)[None], np.array([k]))[0]))
        for side, lu in ((0, r.a_lineup), (1, r.h_lineup)):   # bh side = batting side
            for s, b in enumerate(lu):
                if (r.game_pk, b) not in bgx.index: continue
                a = bgx.loc[(r.game_pk, b)]
                try:
                    q = btx.loc[(b, r.game_date)]
                    hr_, hh_, kk_ = q.hr_rate, q.h_rate, q.k_rate
                except KeyError:
                    hr_, hh_, kk_ = np.nan, np.nan, np.nan
                n = slot_pa[s]
                rows_b.append(dict(game_pk=r.game_pk, date=r.game_date, year=r.year, slot=s,
                                   y_h=int(a.H >= 1), y_hr=int(a.HR >= 1), y_k=int(a.K >= 1),
                                   p_h_sim=1 - z["bh"][i, side, s, 0], p_hr_sim=1 - z["bhr"][i, side, s, 0],
                                   p_k_sim=1 - z["bk"][i, side, s, 0],
                                   p_h_base=1 - (1 - hh_) ** n, p_hr_base=1 - (1 - hr_) ** n, p_k_base=1 - (1 - kk_) ** n))
    K_ = pd.DataFrame(rows_k); B_ = pd.DataFrame(rows_b)
    return K_, B_


def series_frame(variant, years):
    df, z = E.frame(variant, years)
    df["week"] = df.game_date.dt.to_period("W").dt.start_time
    df = df.sort_values(["game_date", "game_pk"])
    # consecutive games between the same two teams at the same park = a series
    key = df.home_team + "|" + df.away_team
    df["prev_gap"] = df.groupby(key).game_date.diff().dt.days
    df["sid"] = (df.prev_gap.isna() | (df.prev_gap > 1)).groupby(key).cumsum()
    df["skey"] = key + "|" + df.sid.astype(str)
    rows = []
    for k, g in df.groupby("skey"):
        if len(g) < 3 or g.week.nunique() > 1: continue
        hw = g.y.sum(); aw = len(g) - hw
        if hw == aw: continue
        p = g.sim_p.values
        # P(home wins majority), independent games
        dist = np.array([1.0])
        for pi in p:
            dist = np.convolve(dist, [1 - pi, pi])
        need = len(g) // 2 + 1
        ps = dist[need:].sum() / (dist[need:].sum() + dist[:len(g) - need + 1].sum() if len(g) % 2 == 0 else 1)
        if len(g) % 2 == 0:
            ps = dist[need:].sum() / (dist[need:].sum() + dist[:len(g) // 2].sum())
        g1 = g.iloc[0]
        rows.append(dict(skey=k, year=g1.year, date=g1.game_date, n=len(g), y=int(hw > aw), sim_series=ps,
                         sim_g1=g1.sim_p, mkt_g1=g1.mkt_p, strength_g1=g1.strength_p, v10_g1=g1.v10_p))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    v = sys.argv[1]; a, b = sys.argv[2].split("-"); years = list(range(int(a), int(b) + 1))
    K_, B_ = score_props(v, years)
    print("starter K  n=%d" % len(K_))
    for m in ("ls", "crps"):
        d = K_[f"{m}_base"] - K_[f"{m}_sim"]
        print(f"  {m}: sim {K_[f'{m}_sim'].mean():.4f} base {K_[f'{m}_base'].mean():.4f}  gain", E.boot(d.values, K_.date.values))
    print("  MAE mean: sim %.3f base %.3f" % ((K_.mu_sim - K_.k).abs().mean(), (K_.mu_base - K_.k).abs().mean()))
    for t in ("h", "hr", "k"):
        ok = B_[f"p_{t}_base"].notna()
        ls = E.ll(B_[f"y_{t}"][ok].values, B_[f"p_{t}_sim"][ok].values); lb = E.ll(B_[f"y_{t}"][ok].values, B_[f"p_{t}_base"][ok].values)
        print(f"batter {t}>=1 n={ok.sum()} sim {ls.mean():.5f} base {lb.mean():.5f} gain", E.boot(lb - ls, B_.date[ok].values))
    S = series_frame(v, years)
    print("series n=%d" % len(S), S.y.mean())
    print(" sim series ll %.5f acc %.4f" % (E.ll(S.y, S.sim_series).mean(), ((S.sim_series > .5) == S.y).mean()))
