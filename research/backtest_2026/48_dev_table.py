"""Dev-window (2016-2019) comparison table for every v2 variant.

    python3 research/backtest_2026/48_dev_table.py [REF] [YEARS]

Winner metrics:   raw log loss 2016-19; calibrated (Platt fit on earlier seasons) 2017-19;
                  market stack gain = ll(market-only refit) - ll(market+sim refit), 2017-19.
Totals metrics:   CRPS and log score of the sim's total-runs pmf, mean bias; O/U log loss
                  vs the closing total (sim P(over) Platt-calibrated on earlier seasons).
F5:               3-way log score, sim recalibrated (multinomial on log pmf, earlier seasons).
All deltas are paired vs REF with date-block bootstrap 95% CIs.
"""
import sys, json, importlib, numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
sys.path.insert(0, "research/backtest_2026")
E = importlib.import_module("44_eval")
run = importlib.import_module("42_run_v2")
REF = sys.argv[1] if len(sys.argv) > 1 else "full"
a, b = (sys.argv[2] if len(sys.argv) > 2 else "2016-2019").split("-")
YEARS = list(range(int(a), int(b) + 1))


def per_game(v):
    df, z = E.summarize_variant(v, YEARS)
    y = df.y.values
    df["L_raw"] = E.ll(y, df.sim_p.values)
    df["L_cal"] = np.where(np.isfinite(df.sim_cal), E.ll(y, np.nan_to_num(df.sim_cal.values, nan=.5)), np.nan)
    df["L_mkt"] = np.where(df.mkt_only.notna(), E.ll(y, np.nan_to_num(df.mkt_only.values, nan=.5)), np.nan)
    df["L_stk"] = np.where(df.stack_mkt.notna(), E.ll(y, np.nan_to_num(df.stack_mkt.values, nan=.5)), np.nan)
    df["L_str"] = np.where(df.strength_p.notna(), E.ll(y, np.nan_to_num(df.strength_p.values, nan=.5)), np.nan)
    # O/U calibrated on earlier seasons
    df["ou_cal"] = np.nan
    for yr in YEARS:
        tr = (df.year < yr) & df.sim_over.notna(); te = (df.year == yr) & df.sim_over.notna()
        if tr.sum() < 500: continue
        m = LogisticRegression(C=1e4).fit(E.lg(df.sim_over[tr].values).reshape(-1, 1), df.over_y[tr])
        df.loc[te, "ou_cal"] = m.predict_proba(E.lg(df.sim_over[te].values).reshape(-1, 1))[:, 1]
    ok = df.ou_cal.notna() & df.over_p.notna()
    df["L_ou_sim"] = np.where(ok, E.ll(df.over_y.fillna(0).values, df.ou_cal.fillna(.5).values), np.nan)
    df["L_ou_mkt"] = np.where(ok, E.ll(df.over_y.fillna(0).values, df.over_p.fillna(.5).values), np.nan)
    return df


if __name__ == "__main__":
    names = list(run.VARIANTS)
    frames = {}
    for v in names:
        try:
            frames[v] = per_game(v)
        except FileNotFoundError:
            print("missing", v)
    ref = frames[REF]
    rows = []
    for v, df in frames.items():
        d = df.dates = df.game_date.dt.strftime("%Y%m%d").values
        r = dict(variant=v, n=len(df), ll_raw=df.L_raw.mean(), ll_cal=np.nanmean(df.L_cal),
                 stack_gain=np.nanmean(df.L_mkt - df.L_stk), crps=df.crps_sim.mean(), ls_tot=df.ls_sim.mean(),
                 bias=(df.mean_sim - df.tot).mean(), ou_ll=np.nanmean(df.L_ou_sim), f5_ls=np.nanmean(df.f5_ls_simcal))
        if v != REF:
            r["d_cal"], r["d_cal_lo"], r["d_cal_hi"], _ = E.boot((ref.L_cal - df.L_cal).values, d)
            r["d_crps"], r["d_crps_lo"], r["d_crps_hi"], _ = E.boot((ref.crps_sim - df.crps_sim).values, d)
        rows.append(r)
    T = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 250)
    print(T.round(5).to_string())
    x = ref
    print("\nbenchmarks on the same games (2017-19 where calibrated):")
    m = x.L_cal.notna() & x.L_mkt.notna()
    print(f"  market ll {x.L_mkt[m].mean():.5f}   strength ll {x.L_str[m].mean():.5f}   {REF} cal {x.L_cal[m].mean():.5f}")
    print(f"  totals: CRPS sim {x.crps_sim.mean():.4f} base {x.crps_base.mean():.4f} mkt-line {np.nanmean(x.crps_mkt):.4f}")
    print(f"          logscore sim {x.ls_sim.mean():.4f} base {x.ls_base.mean():.4f} mkt-line {np.nanmean(x.ls_mkt):.4f}")
    print(f"  O/U ll: sim-cal {np.nanmean(x.L_ou_sim):.5f} market {np.nanmean(x.L_ou_mkt):.5f}")
    print(f"  F5 3-way: sim-cal {np.nanmean(x.f5_ls_simcal):.5f} mkt-derived {np.nanmean(x.f5_ls_mkt):.5f} clim {np.nanmean(x.f5_ls_clim):.5f}")
    T.to_csv("data/backtest_2026/rich/dev_table.csv")
