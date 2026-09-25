"""Export the stored research backtests as JSON for the site's Models pages.

    python3 research/models_page/export_backtests.py [OUT_DIR]

Reads per-game research frames that were produced by the experiment scripts (nothing
is refit here, nothing touches BigQuery) and scores every model the same way the
backend scores live predictions:

  log loss, accuracy, Brier       per model, with 95% bootstrap CIs
  spread / total MAE              where a model produces a margin or a total
  paired deltas                   model vs reference on the SAME games, CI and P(better)
  calibration                     reliability bins (quantile bins of predicted p)

Bootstraps resample whole blocks (a date for MLB, a season-week for football), because
games on one day or in one week share conditions; per-game resampling would give CIs
that are too narrow.

Inputs (absolute, because the frames live outside git):
  MLB     hanks_tank_ml/data/backtest_2026/rich/test_frame.parquet   (51_test_report.py)
  NFL     scratchpad drive_sim/eval_games_{wf,holdout}.parquet        (drive_sim/eval_sim.py)
          scratchpad cmp/bt_nfl.parquet                               (FPI backtest)
  CFB     scratchpad cmp/bt_cfb.parquet                               (FPI backtest)

Output: {mlb,nfl,cfb}.json, copied into the backend at src/data/model-backtests/.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ML_DATA = os.environ.get("ML_DATA", "/Users/VTNX82W/Documents/personalDev/mlb/hanks_tank_ml/data")
SCRATCH = os.environ.get(
    "RESEARCH_SCRATCH",
    "/private/tmp/claude-501/-Users-VTNX82W-Documents-personalDev-mlb/"
    "c6d2d5d7-e0c5-4f76-86af-92e5653efdcd/scratchpad")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out")
REPS = 2000
EPS = 1e-6
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- metrics

def _ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _block_index(blocks):
    """Row indices grouped by block, for a block bootstrap."""
    _, inv = np.unique(blocks, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return [order[s:s + c] for s, c in zip(starts, counts)]


def _boot_means(values: dict, blocks, reps=REPS, seed=0):
    """Bootstrap the mean of each per-game array in `values` with shared resamples."""
    groups = _block_index(blocks)
    k = len(groups)
    sums = {n: np.array([v[g].sum() for g in groups]) for n, v in values.items()}
    sizes = np.array([len(g) for g in groups], float)
    rng = np.random.default_rng(seed)
    out = {n: np.empty(reps) for n in values}
    for r in range(reps):
        pick = rng.integers(0, k, k)
        tot = sizes[pick].sum()
        for n in values:
            out[n][r] = sums[n][pick].sum() / tot
    return out


def _ci(point, draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"value": round(float(point), 5), "lo": round(float(lo), 5), "hi": round(float(hi), 5)}


def calibration(y, p, max_bins=10, min_per_bin=30):
    """Reliability bins: quantile bins of the predicted probability.

    Quantile rather than fixed-width bins, because baseball probabilities sit almost
    entirely inside 0.35-0.70 and ten fixed bins would leave most of them empty.
    """
    n = len(p)
    bins = int(max(1, min(max_bins, n // min_per_bin)))
    order = np.argsort(p, kind="stable")
    out = []
    for chunk in np.array_split(order, bins):
        if not len(chunk):
            continue
        yy, pp = y[chunk], p[chunk]
        m = float(yy.mean())
        se = float(np.sqrt(max(m * (1 - m), 1e-9) / len(chunk)))
        out.append({"p": round(float(pp.mean()), 4), "y": round(m, 4), "n": int(len(chunk)),
                    "y_lo": round(max(0.0, m - 1.96 * se), 4), "y_hi": round(min(1.0, m + 1.96 * se), 4)})
    return out


def score_window(df, y_col, models, blocks_col, ref=None, margin=None, total=None,
                 key="", label="", note="", source=""):
    """Score `models` ({key: (label, prob_col)}) on the rows where ALL of them exist."""
    cols = [c for _, c in models.values()]
    mask = df[cols + [y_col]].notna().all(axis=1)
    d = df[mask].reset_index(drop=True)
    y = d[y_col].astype(float).values
    blocks = d[blocks_col].astype(str).values
    per = {}
    for mk, (_, col) in models.items():
        p = d[col].astype(float).values
        per[mk] = {"ll": _ll(y, p), "acc": ((p > 0.5) == (y == 1)).astype(float),
                   "brier": (p - y) ** 2}
    flat = {f"{m}|{s}": v for m, d_ in per.items() for s, v in d_.items()}
    mae_vals = {}
    for kind, spec in (("spread", margin), ("total", total)):
        if not spec:
            continue
        truth = d[spec["truth"]].astype(float).values
        for mk, col in spec["models"].items():
            if col in d and d[col].notna().all():
                mae_vals[f"{mk}|{kind}_mae"] = np.abs(d[col].astype(float).values - truth)
    flat.update(mae_vals)
    if ref:
        for mk in models:
            if mk != ref:
                flat[f"{mk}|d_ll"] = per[ref]["ll"] - per[mk]["ll"]   # + = model better
    draws = _boot_means(flat, blocks)

    rows = []
    for mk, (mlabel, col) in models.items():
        p = d[col].astype(float).values
        r = {"key": mk, "label": mlabel, "n": int(len(d)),
             "log_loss": _ci(per[mk]["ll"].mean(), draws[f"{mk}|ll"]),
             "accuracy": _ci(per[mk]["acc"].mean(), draws[f"{mk}|acc"]),
             "brier": _ci(per[mk]["brier"].mean(), draws[f"{mk}|brier"]),
             "calibration": calibration(y, p)}
        for kind in ("spread", "total"):
            k = f"{mk}|{kind}_mae"
            if k in flat:
                r[f"{kind}_mae"] = _ci(flat[k].mean(), draws[k])
        rows.append(r)
    deltas = []
    if ref:
        for mk in models:
            if mk == ref:
                continue
            v = flat[f"{mk}|d_ll"]
            dr = draws[f"{mk}|d_ll"]
            deltas.append({"model": mk, "reference": ref,
                           "log_loss_gain": _ci(v.mean(), dr),
                           "p_better": round(float((dr > 0).mean()), 3)})
    return {"key": key, "label": label, "n": int(len(d)), "note": note, "source": source,
            "block": blocks_col, "reference": ref, "models": rows, "deltas": deltas}


# --------------------------------------------------------------------------- MLB

def mlb():
    f = pd.read_parquet(os.path.join(ML_DATA, "backtest_2026/rich/test_frame.parquet"))
    f["elo_p"] = 1 / (1 + 10 ** (-f.elo_d / 400))     # research Elo; elo_d includes HFA
    f["home_rate"] = 0.53
    src = ("hanks_tank_ml research/backtest_2026/51_test_report.py -> "
           "data/backtest_2026/rich/test_frame.parquet (frozen PA-sim config full_x50, 2026-09-25)")
    core = {
        "sim_blend": ("PA sim + strength blend", "stack_str"),
        "strength": ("Team strength (Elo + pythag)", "strength_p"),
        "pa_sim": ("PA simulator alone (calibrated)", "sim_cal"),
        "elo": ("Elo", "elo_p"),
        "home_rate": ("Always 53% home", "home_rate"),
    }
    test = f[f.year.between(2020, 2026)]
    windows = [
        score_window(test, "y", core, "d", ref="strength", key="2020-26",
                     label="Test seasons 2020-2026", source=src,
                     note="Frozen simulator config chosen on 2016-19, then scored once. Calibration "
                          "and the blend are refit each season on earlier seasons only. Lineups are "
                          "the actual starting nine (known ~1-3 h before first pitch)."),
    ]
    mk = f[f.year.isin([2020, 2021]) & f.mkt_p.notna()]
    windows.append(score_window(
        mk, "y", {"market": ("Closing moneyline (benchmark)", "mkt_p"), **core}, "d", ref="market",
        key="2020-21-market", label="2020-2021, games with a closing line", source=src + "; "
        "closing lines: Kaggle vig-free moneylines, SBR archive fallback (data/odds)",
        note="The only seasons in the test window with stored closing lines; the odds archive "
             "stops in 2021. Positive gain = better than the market."))
    m26 = f[(f.year == 2026) & f.v10_p.notna() & f.logit3.notna()]
    windows.append(score_window(
        m26, "y", {"v10": ("V10 (production, as served)", "v10_p"),
                   "logit3": ("3-feature logistic", "logit3"), **core}, "d", ref="v10",
        key="2026", label="2026 regular season to 09-23", source=src,
        note="V10 is its genuine pregame prediction (predicted_at < first pitch). The 3-feature "
             "logistic is refit walk-forward in 100-game steps from game 400. The window "
             "to 09-07 was seen by earlier experiments; 09-08..09-23 was not."))
    unseen = f[(f.year == 2026) & (f.game_date >= "2026-09-08") & f.v10_p.notna() & f.logit3.notna()]
    windows.append(score_window(
        unseen, "y", {"v10": ("V10 (production, as served)", "v10_p"),
                      "logit3": ("3-feature logistic", "logit3"),
                      "sim_blend": core["sim_blend"], "strength": core["strength"]},
        "d", ref="v10", key="2026-unseen", label="2026-09-08..09-23 (unseen)", source=src,
        note="Small: about 200 games. Read the CIs, not the point estimates."))

    rep = json.load(open(os.path.join(ML_DATA, "backtest_2026/rich/test_report.json")))
    tot = rep["totals"]["2020-21"]
    props = rep["props"]
    extras = {
        "totals": {
            "label": "Run totals, 2020-2021 (games with a closing total)",
            "n": tot["n_mkt"],
            "log_score": {"market_nb": round(tot["ls_mkt"], 4),
                          "sim_shape_at_market_mean": round(tot["ls_simmkt"], 4)},
            "gain_sim_shape_at_market_mean": [round(x, 4) for x in tot["sim-shape@mkt-mean vs mkt NB LS"][:3]],
            "p_better": tot["sim-shape@mkt-mean vs mkt NB LS"][3],
            "raw_bias_runs": {k: round(v["bias_raw"], 2) for k, v in rep["totals"].items()
                              if k in ("2020", "2021", "2022-24", "2025", "2026 to 09-07 (seen)")},
            "note": "The simulator's total-runs distribution shape, re-centred on the market total, "
                    "scores better than the market's own negative-binomial distribution. The raw "
                    "sim runs hot, so its mean is never used on its own.",
        },
        "starter_k": {
            "label": "Starter strikeouts, 2020-2026", "n": props["starter_K"]["n"],
            "crps": {"sim": round(props["starter_K"]["crps_sim"], 3),
                     "baseline": round(props["starter_K"]["crps_base"], 3)},
            "mae": {"sim": round(props["starter_K"]["mae_sim"], 3),
                    "baseline": round(props["starter_K"]["mae_base"], 3)},
            "gain_crps": [round(x, 4) for x in props["starter_K"]["crps gain"][:3]],
        },
        "batter_hit": {
            "label": "Batter P(at least one hit), 2020-2026", "n": props["batter_h>=1"]["n"],
            "actual_rate": round(props["batter_h>=1"]["rate"], 4),
            "mean_predicted": round(props["batter_h>=1"]["mean_p_sim"], 4),
            "status": "experimental - over-predicted (no substitutions modelled); not shown until calibrated",
        },
    }
    return {"sport": "mlb", "generated_at": TODAY, "kind": "backtest", "windows": windows,
            "extras": extras}


# --------------------------------------------------------------------------- football

def nfl():
    wf = pd.read_parquet(os.path.join(SCRATCH, "drive_sim/eval_games_wf.parquet"))
    ho = pd.read_parquet(os.path.join(SCRATCH, "drive_sim/eval_games_holdout.parquet"))
    for d in (wf, ho):
        d["wk"] = d.season.astype(str) + "-" + d.week.astype(str)
        d["y"] = d.home_won.astype(float)
    drv_src = ("hanks_tank_ml research/football_2026_09/drive_sim (eval_sim.py), frozen config "
               "c_g chosen on 2010-16 before scoring")
    models = {"market": ("Closing line (benchmark)", "vegas"),
              "ridge": ("Margin ridge", "ridge"),
              "drive_sim": ("Drive simulator (calibrated)", "cal_c_g")}
    margin = {"truth": "margin", "models": {"market": "spread_line", "ridge": "ridge_margin",
                                            "drive_sim": "mcal_c_g"}}
    total = {"truth": "total", "models": {"market": "total_line", "drive_sim": "tcal_c_g"}}
    w1 = score_window(wf[wf.season.between(2017, 2024)], "y", models, "wk", ref="market",
                      margin=margin, total=total, key="2017-24", label="Walk-forward 2017-2024",
                      source=drv_src, note="Ties (home_won missing) dropped. Ridge and sim refit weekly "
                                            "on games before each week.")
    w2 = score_window(ho[ho.season == 2025], "y", models, "wk", ref="market", margin=margin, total=total,
                      key="2025", label="Holdout 2025", source=drv_src,
                      note="Scored once, after the config was frozen.")
    bt = pd.read_parquet(os.path.join(SCRATCH, "cmp/bt_nfl.parquet"))
    bt["wk"] = bt.season.astype(str) + "-" + bt.week.astype(str)
    bt["y"] = bt.home_won.astype(float)
    fpi_src = ("hanks_tank_ml scripts/football/eval_fpi_vs_models.py; ESPN core-API predictor "
               "(pregame gameProjection), backfilled XGBoost with the EPA fix")
    fm = {"market": ("Closing line (benchmark)", "market"), "ridge": ("Margin ridge", "ridge"),
          "xgb": ("XGBoost (EPA fixed, backfilled)", "xgb"), "fpi": ("ESPN FPI (not a target)", "fpi")}
    w3 = score_window(bt[bt.season.isin([2024, 2025])], "y", fm, "wk", ref="market",
                      key="2024-25-fpi", label="2024-2025 with ESPN FPI", source=fpi_src,
                      note="XGBoost here is a backfill with the EPA fix applied, not what production "
                           "served; production ran without EPA all of 2025-26.")
    return {"sport": "nfl", "generated_at": TODAY, "kind": "backtest", "windows": [w1, w2, w3]}


def cfb():
    bt = pd.read_parquet(os.path.join(SCRATCH, "cmp/bt_cfb.parquet"))
    bt["wk"] = bt.season.astype(str) + "-" + bt.week.astype(str)
    bt["y"] = bt.home_won.astype(float)
    src = ("hanks_tank_ml scripts/football/eval_fpi_vs_models.py; ESPN core-API predictor; "
           "market = Phi(spread/15.5) from the consensus spread")
    fm = {"market": ("Market spread (benchmark)", "market"), "ridge": ("Margin ridge", "ridge"),
          "fpi": ("ESPN FPI (not a target)", "fpi"),
          "xgb": ("XGBoost (legacy production, backfilled)", "xgb")}
    s25 = bt[bt.season == 2025]
    return {"sport": "cfb", "generated_at": TODAY, "kind": "backtest", "windows": [
        score_window(s25, "y", fm, "wk", ref="market", key="fbs-2025", label="FBS 2025",
                     source=src, note="FBS games only. XGBoost is a backfill, not the live rows."),
        score_window(s25[s25.week <= 4], "y", fm, "wk", ref="market", key="fbs-2025-w1-4",
                     label="FBS 2025, weeks 1-4", source=src,
                     note="Early season, where FPI's preseason priors help most."),
        score_window(s25[s25.week >= 5], "y", fm, "wk", ref="market", key="fbs-2025-w5",
                     label="FBS 2025, weeks 5+", source=src, note="FPI and the ridge tie here."),
    ]}


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("mlb", mlb), ("nfl", nfl), ("cfb", cfb)):
        data = fn()
        path = os.path.join(OUT, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1)
        print(name, path)
        for w in data["windows"]:
            print(f"  {w['key']:<16} n={w['n']:>6}  " + "  ".join(
                f"{m['key']} {m['log_loss']['value']:.4f}" for m in w["models"]))


if __name__ == "__main__":
    main()
