"""Walk-forward evaluation of rating models: which one predicts FUTURE games best.

    python -m rankings.evaluate --sport nfl --out /tmp/nfl_eval

This is the only yardstick the rankings are tuned against. For every week of a season
the model is fitted on games strictly before that week (plus last season at its decayed
weight, exactly as production does) and scored by log loss on that week's games. No
external ranking — AP, coaches, FPI — enters anywhere.

Selection bias is handled by splitting seasons: every hyperparameter is chosen on the
TUNE seasons, then frozen and scored on the EVAL seasons it never saw. Trying 200
configurations and quoting the best one's score on the same data would overstate it.

Each fit's per-game output is stored as a raw logit (BT) or a raw predicted margin in
points (margin model). That makes the margin model's points-to-log-odds scale and the
BT/margin blend weight cheap post-hoc choices on the tune seasons, with no refitting.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rankings import core, sources  # noqa: E402
from rankings.sources import SPORTS  # noqa: E402

logger = logging.getLogger(__name__)

# Tune seasons precede eval seasons, so the reported numbers are out of sample twice
# over: out of time within a season, and out of the data used to pick the settings.
# 2020 is excluded as a scored season everywhere (60-game MLB season, COVID college
# season) but still serves as the prior for 2021.
SPLITS = {
    "nfl": {"tune": [2016, 2017, 2018, 2019, 2020], "eval": [2021, 2022, 2023, 2024, 2025]},
    "cfb": {"tune": [2017, 2018, 2019, 2021], "eval": [2022, 2023, 2024, 2025]},
    "mlb": {"tune": [2018, 2019, 2021, 2022], "eval": [2023, 2024, 2025, 2026]},
}

GRIDS = {
    "nfl": {
        "C": [0.125, 0.25, 0.5, 1.0, 2.0], "alpha": [1.0, 3.0, 10.0, 30.0],
        "cap": [None, 14.0, 17.0, 21.0, 28.0],
    },
    "cfb": {
        "C": [0.5, 1.0, 2.0, 4.0, 8.0], "alpha": [0.03, 0.1, 0.3, 1.0, 3.0],
        "cap": [None, 28.0, 38.0, 50.0],
    },
    "mlb": {
        "C": [0.015, 0.03, 0.06, 0.12, 0.25], "alpha": [10.0, 30.0, 100.0, 300.0],
        "cap": [None, 2.0, 3.0, 4.0, 6.0],
    },
}
TAUS = {"nfl": [2.0, 4.0, 8.0, 16.0], "cfb": [2.0, 4.0, 8.0, 16.0],
        "mlb": [4.0, 8.0, 20.0, 40.0]}
W0S = [0.25, 0.5, 1.0]
# In-season recency decay (weeks); None is the order-independent fit.
RECENCY = {"nfl": [None, 8.0, 16.0], "cfb": [None, 6.0, 12.0], "mlb": [None, 8.0, 16.0]}

_GAMES: pd.DataFrame | None = None


def load_games(sport: str) -> pd.DataFrame:
    seasons = sorted({s for part in SPLITS[sport].values() for y in part for s in (y, y - 1)})
    games = sources.load(sport, tuple(seasons) if sport == "mlb" else None)
    games = games[games["season"].isin(seasons)]
    # A stable identity for each game, so predictions from different configurations
    # (and different worker processes) line up by game, never by row position.
    key = (games["home_team_name"].astype(str) + "|" + games["away_team_name"].astype(str)
           + "|" + games["game_date"].astype(str))
    # Doubleheaders share teams and date; number them apart.
    key = key + "|" + games.groupby(key).cumcount().astype(str)
    games = games.assign(game_key=key).sort_values(["season", "week", "game_key"])
    return games.reset_index(drop=True)


def _scored(sport: str, test: pd.DataFrame) -> pd.DataFrame:
    """The games a board is judged on. College: both sides Division I that season."""
    if sport == "cfb":
        return test[test["home_division"].notna() & test["away_division"].notna()]
    return test


def _predict(strengths: pd.Series, home_adv: float, test: pd.DataFrame) -> np.ndarray:
    """Raw home-minus-away output in the model's own units, divided by ELO_SCALE.

    For BT that is the log-odds; for the margin model fitted with scale=1 it is the
    predicted margin in points. A team with no games yet rates as average (0).
    """
    h = test["home_team_name"].map(strengths).fillna(0.0).to_numpy()
    a = test["away_team_name"].map(strengths).fillna(0.0).to_numpy()
    home = (1 - test["neutral_site"].to_numpy()) * home_adv
    return (h - a + home) / core.ELO_SCALE


def run_config(args: tuple) -> dict:
    """Walk one configuration forward through every season; per-game raw outputs."""
    sport, cfg, seasons = args
    games = _GAMES
    spec = SPORTS[sport]
    major = spec.major_division
    out_rows, rank_rows = [], []
    kw = dict(C=cfg.get("C", 1.0), w0=cfg["w0"], tau=cfg["tau"], major=major,
              model=cfg["model"], margin_alpha=cfg.get("alpha", 10.0),
              margin_cap=cfg.get("cap"), margin_scale=1.0,
              recency_tau=cfg.get("recency"))

    for season in seasons:
        current = games[games["season"] == season]
        prior = games[games["season"] == season - 1]
        prior = None if prior.empty else prior
        divisions = core.season_divisions(prior, current)
        if prior is not None and not cfg.get("use_prior", True):
            prior = None
        for week in sorted(current["week"].unique()):
            test = _scored(sport, current[current["week"] == week])
            seen = current[current["week"] < week]
            if test.empty:
                continue
            if prior is None and seen.empty:
                # No evidence at all: the honest forecast is a coin flip. Scored, not
                # skipped, so every configuration is judged on the same games.
                out_rows.append(pd.DataFrame({
                    "season": season, "week": int(week),
                    "game_idx": test["game_key"].to_numpy(), "y": test["home_won"].to_numpy(),
                    "raw": np.zeros(len(test)),
                }))
                continue
            try:
                strengths, home_adv, _ = core.fit_with_prior(
                    seen, prior, int(week), divisions=divisions, **kw
                )
            except Exception as exc:  # pragma: no cover - reported, not hidden
                logger.warning("%s %s wk%s failed: %s", cfg, season, week, exc)
                continue
            raw = _predict(strengths, home_adv, test)
            out_rows.append(pd.DataFrame({
                "season": season, "week": int(week),
                "game_idx": test["game_key"].to_numpy(), "y": test["home_won"].to_numpy(),
                "raw": raw,
            }))
            order = strengths.index
            rank_rows.append(pd.DataFrame({
                "season": season, "week": int(week), "team": order,
                "rank": np.arange(1, len(order) + 1), "strength": strengths.to_numpy(),
            }))
    preds = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    ranks = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()
    return {"cfg": cfg, "preds": preds, "ranks": ranks}


def _init(sport: str, games: pd.DataFrame | None = None) -> None:
    """Worker setup. Games are loaded ONCE by the parent and handed over, so every
    worker sees an identical frame (a worker that failed to reach ESPN's conference
    tree would otherwise fall back to a different division map)."""
    global _GAMES
    logging.disable(logging.WARNING)
    _GAMES = games if games is not None else load_games(sport)


def configs(sport: str) -> list[dict]:
    g, rec = GRIDS[sport], RECENCY[sport]
    out = [{"model": "bt", "C": C, "tau": tau, "w0": w0, "recency": r}
           for C, tau, w0, r in itertools.product(g["C"], TAUS[sport], W0S, rec)]
    out += [{"model": "bt", "C": C, "tau": 1.0, "w0": 0.0, "use_prior": False,
             "recency": None} for C in g["C"]]
    out += [{"model": "margin", "alpha": a, "cap": cap, "tau": tau, "w0": w0, "recency": r}
            for a, cap, tau, w0, r in itertools.product(
                g["alpha"], g["cap"], TAUS[sport], W0S, rec)]
    out += [{"model": "margin", "alpha": a, "cap": cap, "tau": 1.0, "w0": 0.0,
             "use_prior": False, "recency": None}
            for a, cap in itertools.product(g["alpha"], g["cap"])]
    return out


def baseline_config(sport: str) -> dict:
    spec = SPORTS[sport]
    return {"model": "bt", "C": spec.ridge_C, "tau": spec.prior_tau, "w0": spec.prior_w0,
            "recency": None}


# ── scoring ─────────────────────────────────────────────────────────────────
def ll_vec(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_scale(raw: np.ndarray, y: np.ndarray) -> float:
    """Points per log-odds unit that minimises log loss (margin model)."""
    res = minimize_scalar(lambda s: ll_vec(y, sigmoid(raw / s)).mean(),
                          bounds=(0.05, 200.0), method="bounded")
    return float(res.x)


def fit_blend(bt: np.ndarray, mg: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(weight on BT, margin scale) minimising log loss of a*bt + (1-a)*mg/scale."""
    def f(v):
        a, s = v
        return ll_vec(y, sigmoid(a * bt + (1 - a) * mg / s)).mean()
    s0 = fit_scale(mg, y)
    best = minimize(f, x0=[0.5, s0], bounds=[(0.0, 1.0), (0.05, 200.0)],
                    method="L-BFGS-B")
    return float(best.x[0]), float(best.x[1])


def block_bootstrap_diff(frame: pd.DataFrame, a: str, b: str, n: int = 2000,
                         seed: int = 0) -> tuple[float, float, float]:
    """Mean per-game log-loss difference a-b with a 95% CI, resampling whole weeks.

    Weeks, not games, are the resampling unit: games in one week share the same fit,
    so treating them as independent would make the interval too narrow.
    """
    d = frame[a] - frame[b]
    blocks = frame.assign(d=d).groupby(["season", "week"])["d"].agg(["sum", "count"])
    sums, counts = blocks["sum"].to_numpy(), blocks["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), (n, len(sums)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def rank_stability(ranks: pd.DataFrame, members: set | None = None) -> float:
    """Mean absolute week-over-week rank change, over teams on consecutive boards."""
    r = ranks if members is None else ranks[ranks["team"].isin(members)].copy()
    if members is not None:
        r["rank"] = r.groupby(["season", "week"])["strength"].rank(ascending=False)
    r = r.sort_values(["season", "team", "week"])
    r["prev"] = r.groupby(["season", "team"])["rank"].shift()
    return float((r["rank"] - r["prev"]).abs().mean())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", required=True, choices=sorted(SPLITS))
    ap.add_argument("--out", required=True, help="directory for raw results")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seasons = SPLITS[args.sport]["tune"] + SPLITS[args.sport]["eval"]
    cfgs = configs(args.sport)
    base = baseline_config(args.sport)
    if base not in cfgs:
        cfgs.append(base)
    logger.info("%s: %d configs x %d seasons", args.sport, len(cfgs), len(seasons))

    results = []
    games = load_games(args.sport)
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.sport, games)) as pool:
        for i, res in enumerate(pool.map(run_config,
                                         [(args.sport, c, seasons) for c in cfgs])):
            results.append(res)
            if (i + 1) % 25 == 0:
                logger.info("%d/%d configs", i + 1, len(cfgs))

    preds = []
    for i, res in enumerate(results):
        p = res["preds"].assign(cfg_id=i)
        preds.append(p)
        res["ranks"].assign(cfg_id=i).to_parquet(out / f"ranks_{i}.parquet")
    pd.concat(preds).to_parquet(out / "preds.parquet")
    (out / "configs.json").write_text(json.dumps([r["cfg"] for r in results]))
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ── report ──────────────────────────────────────────────────────────────────
def _label(cfg: dict) -> str:
    if cfg["model"] == "bt":
        core_ = f"BT C={cfg['C']}"
    else:
        core_ = f"margin a={cfg['alpha']} cap={cfg['cap']}"
    if cfg.get("recency"):
        core_ += f" recency={cfg['recency']}"
    if not cfg.get("use_prior", True):
        return core_ + " no-prior"
    return core_ + f" w0={cfg['w0']} tau={cfg['tau']}"


def report(sport: str, out_dir: str, top_k: int = 6) -> dict:
    """Pick every setting on the tune seasons, then score the frozen picks on eval."""
    out = Path(out_dir)
    cfgs = json.loads((out / "configs.json").read_text())
    preds = pd.read_parquet(out / "preds.parquet")
    tune, evals = SPLITS[sport]["tune"], SPLITS[sport]["eval"]
    preds["part"] = np.where(preds["season"].isin(tune), "tune", "eval")
    key = ["season", "week", "game_idx"]
    wide = preds.pivot_table(index=key + ["y", "part"], columns="cfg_id", values="raw")
    # Every variant must be judged on the same games. A configuration whose fit
    # failed in some week is dropped from the comparison, not allowed to shrink the
    # common test set for everyone else.
    complete = wide.notna().mean() >= 0.999
    dropped = [int(c) for c in wide.columns[~complete]]
    if dropped:
        logger.warning("dropping %d configs with incomplete coverage: %s",
                       len(dropped), dropped)
    wide = wide.loc[:, complete].dropna(axis=0, how="any").reset_index()
    y = wide["y"].to_numpy().astype(float)
    is_tune = (wide["part"] == "tune").to_numpy()

    probs: dict[int, np.ndarray] = {}
    scales: dict[int, float] = {}
    for i, cfg in enumerate(cfgs):
        if i not in wide.columns:
            continue
        raw = wide[i].to_numpy()
        if cfg["model"] == "margin":
            scales[i] = fit_scale(raw[is_tune], y[is_tune])
            probs[i] = sigmoid(raw / scales[i])
        else:
            probs[i] = sigmoid(raw)

    def ll(p, mask):
        return float(ll_vec(y[mask], p[mask]).mean())

    table = pd.DataFrame([{
        "cfg_id": i, "label": _label(c), "model": c["model"],
        "no_prior": not c.get("use_prior", True),
        "tune_ll": ll(probs[i], is_tune), "eval_ll": ll(probs[i], ~is_tune),
    } for i, c in enumerate(cfgs) if i in probs])

    base_cfg = baseline_config(sport)
    base_id = next(i for i, c in enumerate(cfgs) if c == base_cfg)
    pick = {"baseline (production)": base_id}
    for name, mask in [("BT retuned", (table.model == "bt") & ~table.no_prior),
                       ("BT no prior", (table.model == "bt") & table.no_prior),
                       ("margin", (table.model == "margin") & ~table.no_prior),
                       ("margin no prior", (table.model == "margin") & table.no_prior)]:
        pick[name] = int(table[mask].sort_values("tune_ll").iloc[0]["cfg_id"])

    # Blend: the best few of each family, weight and scale fitted on tune only.
    bts = table[(table.model == "bt") & ~table.no_prior].nsmallest(top_k, "tune_ll").cfg_id
    mgs = table[(table.model == "margin") & ~table.no_prior].nsmallest(top_k, "tune_ll").cfg_id
    best = None
    for b in bts:
        for m in mgs:
            braw, mraw = wide[b].to_numpy(), wide[m].to_numpy()
            a, s = fit_blend(braw[is_tune], mraw[is_tune], y[is_tune])
            p = sigmoid(a * braw + (1 - a) * mraw / s)
            score = ll(p, is_tune)
            if best is None or score < best[0]:
                best = (score, int(b), int(m), a, s, p)
    _, bb, bm, ba, bs, bp = best

    columns = {name: probs[i] for name, i in pick.items()}
    columns["blend"] = bp
    frame = wide[key + ["part"]].copy()
    for name, p in columns.items():
        frame[name] = ll_vec(y, p)
    frame["coin flip"] = np.log(2.0)
    ev = frame[frame["part"] == "eval"]

    rows = []
    for name in ["coin flip"] + list(columns):
        d, lo, hi = block_bootstrap_diff(ev, name, "baseline (production)")
        rows.append({"variant": name, "eval_ll": ev[name].mean(),
                     "tune_ll": frame.loc[frame.part == "tune", name].mean(),
                     "diff_vs_base": d, "ci_lo": lo, "ci_hi": hi})
    summary = pd.DataFrame(rows)

    per_season = ev.groupby("season")[["baseline (production)", "BT retuned",
                                       "margin", "blend"]].mean()
    early = ev.assign(early=ev["week"] <= 4).groupby("early")[
        ["baseline (production)", "BT retuned", "margin", "blend"]].mean()

    # Rank stability on eval seasons, per chosen config (blend built from its parts).
    games = load_games(sport)
    members = None
    if sport == "cfb":
        fbs = games[games["season"].isin(evals)]
        members = set(fbs.loc[fbs["home_division"] == "fbs", "home_team_name"]) | \
            set(fbs.loc[fbs["away_division"] == "fbs", "away_team_name"])

    def ranks_of(i):
        r = pd.read_parquet(out / f"ranks_{i}.parquet")
        return r[r["season"].isin(evals)]

    stability = {}
    for name, i in pick.items():
        stability[name] = rank_stability(ranks_of(i), members)
    rb, rm = ranks_of(bb), ranks_of(bm)
    merged = rb.merge(rm, on=["season", "week", "team"], suffixes=("_b", "_m"))
    merged["strength"] = ba * merged["strength_b"] + (1 - ba) * merged["strength_m"] / bs
    merged["rank"] = merged.groupby(["season", "week"])["strength"].rank(ascending=False)
    stability["blend"] = rank_stability(merged[["season", "week", "team", "rank",
                                                "strength"]], members)

    chosen = {name: {**cfgs[i], "label": _label(cfgs[i]),
                     **({"scale": scales[i]} if i in scales else {})}
              for name, i in pick.items()}
    chosen["blend"] = {"bt": _label(cfgs[bb]), "margin": _label(cfgs[bm]),
                       "weight_bt": ba, "scale": bs, "bt_cfg": cfgs[bb],
                       "margin_cfg": cfgs[bm]}
    return {"summary": summary, "per_season": per_season, "early": early,
            "stability": stability, "chosen": chosen, "n_eval": int(len(ev)),
            "n_tune": int((frame.part == "tune").sum()), "grid": table,
            "fbs_only": _fbs_only(sport, games, ev) if sport == "cfb" else None}


def _fbs_only(sport, games, ev):
    g = games.set_index("game_key").loc[ev["game_idx"]]
    mask = ((g["home_division"] == "fbs") & (g["away_division"] == "fbs")).to_numpy()
    sub = ev[mask]
    return sub[["baseline (production)", "BT retuned", "margin", "blend"]].mean(), len(sub)
