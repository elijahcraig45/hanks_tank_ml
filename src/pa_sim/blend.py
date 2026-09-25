"""Production runner for the v2 PA simulator + team-strength blend (SHADOW ONLY).

What gets written, per game on the target slate:

  game_predictions_sim_blend   home_win_probability = the blend
        blend = sigmoid(a + b1*logit(strength_p) + b2*logit(sim_p))
      with frozen coefficients (blend_coefs.json). This is the one simulator output with a
      measured winner gain: +0.0021 nats over strength alone on 15,000 test games
      (2020-26), CI [+0.0010, +0.0032]. The sim on its own only ties strength.
  game_props_sim               the distributions the sim is actually good at:
      total runs pmf (tilted to remove the raw sim's measured over-prediction of runs), and
      each starter's strikeout pmf. Batter props only with experimental=True, because
      P(>=1 hit) is over-predicted (64.5% vs 60.8% actual) — the sim has no substitutions.
      There is no live market total, so market_total_line / p_over_market stay NULL; the
      research's "sim shape at the market mean" (+0.010 log score) applies where a line
      exists and is done at read time, not here.

Nothing here runs by default. The Cloud Function dispatches it only for mode=sim_blend or
{"run_sim_blend": true}, and it refuses (status insufficient_memory) in a container
below SIM_BLEND_MIN_MEMORY_MB (default 3072): peak memory measured 1.3-2.2 GB and the live
function has 1 GB. Both tables are APPEND-ONLY and loaded with CREATE_NEVER; readers take
the latest row per game written before first pitch, and games that have already started
are skipped (a date-wide DELETE + rewrite, as v1 did, would replace rows for games in
progress with post-first-pitch rows that an honest scoreboard has to discard).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT = os.environ.get("GCP_PROJECT", "hankstank")
DATASET = os.environ.get("MLB_2026_DATASET", "mlb_2026_season")
PRED_TABLE = os.environ.get("SIM_BLEND_TABLE", "game_predictions_sim_blend")
PROPS_TABLE = os.environ.get("SIM_PROPS_TABLE", "game_props_sim")
COEFS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blend_coefs.json")
EPS = 1e-4


def load_coefs(path: str = COEFS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def calibrate_sim(sim_p, coefs: dict):
    c = coefs["platt_sim"]
    return sigmoid(c["intercept"] + c["coef"] * logit(sim_p))


def blend(strength_p, sim_p, coefs: dict):
    s = coefs["stack"]
    return sigmoid(s["intercept"] + s["coef_logit_strength"] * logit(strength_p)
                   + s["coef_logit_sim"] * logit(sim_p))


def tilt(pmf: np.ndarray, target: float) -> np.ndarray:
    """Exponentially tilt a pmf (p_k e^{theta k}) so its mean is `target` (44_eval.tilt)."""
    from scipy.optimize import brentq

    k = np.arange(len(pmf)); p = np.asarray(pmf, float) + 1e-12
    if not np.isfinite(target):
        return np.asarray(pmf, float)
    f = lambda th: (p * np.exp(th * k) * k).sum() / (p * np.exp(th * k)).sum() - target
    try:
        th = brentq(f, -1.5, 1.5)
    except ValueError:
        th = 0.0
    q = p * np.exp(th * k)
    return q / q.sum()


def tier(p: float) -> str:
    c = max(p, 1 - p)
    return "high" if c >= 0.64 else ("medium" if c >= 0.57 else "low")


# --------------------------------------------------------------------------- memory guard
def container_memory_mb() -> float | None:
    """Best available memory limit: explicit override, cgroup, then the Functions env."""
    if os.environ.get("SIM_BLEND_MEMORY_MB"):
        return float(os.environ["SIM_BLEND_MEMORY_MB"])
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
            if raw and raw != "max" and int(raw) < 1 << 50:
                return int(raw) / 2 ** 20
        except (OSError, ValueError):
            pass
    if os.environ.get("FUNCTION_MEMORY_MB"):
        return float(os.environ["FUNCTION_MEMORY_MB"])
    return None


def memory_ok() -> tuple[bool, float | None, float]:
    need = float(os.environ.get("SIM_BLEND_MIN_MEMORY_MB", "3072"))
    have = container_memory_mb()
    if have is None:
        # Unknown limit: fine on a workstation, never assumed inside a managed runtime.
        return (not os.environ.get("K_SERVICE")), have, need
    return have >= need, have, need


# --------------------------------------------------------------------------- rows
def game_rows(summary: dict, slate: pd.DataFrame, strength_p: np.ndarray, coefs: dict,
              n_episodes: int, now: datetime, experimental: bool = False) -> tuple[list, list]:
    """Turn v2.summarize() output (games in slate order) into the two tables' rows."""
    mv = coefs["model_version"]
    bias = float(coefs["totals_bias_runs"])
    th = summary["tot_hist"]; hh = summary["h_hist"]; ah = summary["a_hist"]
    k = np.arange(th.shape[1])
    pred, props = [], []
    for i, r in enumerate(slate.itertuples()):
        sim_p = float(summary["p_home"][i])
        p = float(blend(strength_p[i], sim_p, coefs))
        mh = float((hh[i] * np.arange(hh.shape[1])).sum())
        ma = float((ah[i] * np.arange(ah.shape[1])).sum())
        raw_mean = float((th[i] * k).sum())
        q = tilt(th[i], raw_mean - bias)
        base = dict(game_pk=int(r.game_pk), game_date=r.game_date, game_time_utc=r.game_time_utc)
        pred.append(dict(
            **base, home_team_id=int(r.home_team_id), away_team_id=int(r.away_team_id),
            home_team_name=r.home_team_name, away_team_name=r.away_team_name,
            home_starter_id=int(r.home_starter_id), away_starter_id=int(r.away_starter_id),
            home_win_probability=p, away_win_probability=1 - p,
            sim_p_raw=sim_p, sim_p_cal=float(calibrate_sim(sim_p, coefs)),
            strength_p=float(strength_p[i]),
            predicted_winner=r.home_team_name if p >= 0.5 else r.away_team_name,
            confidence_tier=tier(p), model_version=mv, n_episodes=int(n_episodes),
            mean_home_runs=mh, mean_away_runs=ma, predicted_at=now))
        row = dict(
            **base, home_team_name=r.home_team_name, away_team_name=r.away_team_name,
            model_version=mv, n_episodes=int(n_episodes),
            mean_home_runs=mh, mean_away_runs=ma, mean_total_runs=float((q * k).sum()),
            total_runs_pmf=[float(x) for x in q], totals_calibrated=True,
            total_bias_shift=-bias, market_total_line=None, p_over_market=None,
            batter_props_json=None, predicted_at=now)
        for side, s in (("home", 1), ("away", 0)):       # spk[:, 0] = away SP, [:, 1] = home SP
            pmf = summary["spk"][i, s]
            row[f"{side}_starter_id"] = int(getattr(r, f"{side}_starter_id"))
            row[f"{side}_starter_name"] = getattr(r, f"{side}_starter_name", None)
            row[f"{side}_starter_k_mean"] = float((pmf * np.arange(len(pmf))).sum())
            row[f"{side}_starter_k_pmf"] = [float(x) for x in pmf]
        if experimental:
            row["batter_props_json"] = json.dumps(_batter_props(summary, i, r))
        props.append(row)
    return pred, props


def _batter_props(summary, i, r) -> dict:
    """EXPERIMENTAL: P(>=1 hit), P(>=1 HR), P(>=1 K) per lineup slot. Over-predicted."""
    out = {"calibrated": False,
           "warning": "P(>=1 hit) over-predicted in backtest: 64.5% vs 60.8% actual"}
    for side, s in (("home", 1), ("away", 0)):              # batting side: 0 away, 1 home
        lu = getattr(r, f"{side}_lineup", None) or [None] * 9
        out[side] = [dict(slot=j + 1, player_id=None if lu[j] is None else int(lu[j]),
                          p_hit=float(1 - summary["bh"][i, s, j, 0]),
                          p_hr=float(1 - summary["bhr"][i, s, j, 0]),
                          p_k=float(1 - summary["bk"][i, s, j, 0])) for j in range(9)]
    return out


# --------------------------------------------------------------------------- run
def _slate(bq, target: date) -> pd.DataFrame:
    """One row per game with a complete pregame lineup (reuses the v1 slate query)."""
    from google.cloud import bigquery
    from pa_sim.pipeline import SLATE_SQL, _team_abbr_map

    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("d", "DATE", target)])
    long = bq.query(SLATE_SQL.format(proj=PROJECT, ds=DATASET), job_config=cfg).to_dataframe()
    ven = bq.query(f"SELECT game_pk, ANY_VALUE(venue_id) venue_id FROM `{PROJECT}.{DATASET}.games` "
                   "WHERE game_date = @d GROUP BY game_pk", job_config=cfg).to_dataframe()
    return assemble_slate(long, dict(zip(ven.game_pk, ven.venue_id)), _team_abbr_map(bq))


def assemble_slate(long: pd.DataFrame, venue_of: dict, name2ab: dict) -> pd.DataFrame:
    rows = []
    for gid, g in long.groupby("game_pk", sort=False):
        h = g[g.team_type == "home"].sort_values("batting_order")
        a = g[g.team_type == "away"].sort_values("batting_order")
        r0 = g.iloc[0]
        if len(h) != 9 or len(a) != 9 or pd.isna(r0.home_starter_id) or pd.isna(r0.away_starter_id):
            logger.warning("sim_blend: skipping %s (incomplete lineup/starter)", gid)
            continue
        rows.append(dict(
            game_pk=int(gid), game_date=r0.game_date, game_time_utc=r0.game_time_utc,
            home_team_id=int(r0.home_team_id), away_team_id=int(r0.away_team_id),
            home_team_name=r0.home_team_name, away_team_name=r0.away_team_name,
            home_ab=name2ab.get(r0.home_team_name, ""), away_ab=name2ab.get(r0.away_team_name, ""),
            home_starter_id=int(r0.home_starter_id), away_starter_id=int(r0.away_starter_id),
            home_starter_name=r0.get("home_starter_name"), away_starter_name=r0.get("away_starter_name"),
            home_lineup=[int(x) for x in h.player_id], away_lineup=[int(x) for x in a.player_id],
            venue_id=int(venue_of[gid]) if gid in venue_of and pd.notna(venue_of[gid]) else -1))
    return pd.DataFrame(rows)


def strength_for_slate(games: pd.DataFrame, slate: pd.DataFrame, target: date) -> np.ndarray:
    """Strength p for each slate game, with nothing on or after `target` counted."""
    from pa_sim import strength

    g = games.copy()
    g["game_date"] = pd.to_datetime(g.game_date)
    # The games table only gets a row once the collector sees a game, which is after
    # it is played; tonight's slate is not in it yet. Add the slate as unplayed rows
    # so the chronological pass reaches them (they update no ratings: no runs).
    todo = slate[~slate.game_pk.isin(g.game_pk)]
    if len(todo):
        g = pd.concat([g, pd.DataFrame({
            "game_pk": todo.game_pk.astype(int).values,
            "game_date": pd.to_datetime(todo.game_date).values,
            "year": pd.to_datetime(todo.game_date).dt.year.values,
            "home": todo.home_team_id.astype(int).astype(str).values,
            "away": todo.away_team_id.astype(int).astype(str).values,
            "h_runs": np.nan, "a_runs": np.nan,
        })], ignore_index=True)
    late = g.game_date >= pd.Timestamp(target)
    g.loc[late, ["h_runs", "a_runs"]] = np.nan              # same-day results never leak
    G = strength.features(g)
    m = strength.fit(G, target.year)
    if m is None:
        raise RuntimeError("strength: fewer than 1000 games in the 3 prior seasons")
    idx = G.set_index("game_pk")
    missing = [pk for pk in slate.game_pk if pk not in idx.index]
    if missing:
        raise RuntimeError(f"strength: slate games missing from games table: {missing[:5]}")
    return strength.predict(m, idx.loc[slate.game_pk.values].reset_index())


def build_engine(inputs: dict, target: date, coefs: dict):
    from pa_sim import v2, v2_inputs

    cfg = v2.Config(**{k: v for k, v in coefs["sim_config"].items() if k != "variant"},
                    tag=coefs["sim_config"]["variant"])
    pa = inputs["pa"]
    pa15 = pa[pa.game_year == pa.game_year.min()]
    xtab = v2.build_xtable(pa15) if cfg.xw > 0 else None
    D = v2.Data(pa[pa.game_date < pd.Timestamp(target)], inputs["venue_of"], None, xtab)
    hook = inputs["hook"].merge(D.starts[["game_pk", "pitcher", "exp_p"]].rename(columns={"pitcher": "sp"}),
                                on=["game_pk", "sp"], how="left")
    hook["exp_p"] = hook.exp_p.fillna(88.0)
    tto = v2_inputs.est_tto(pa15) if cfg.tto else None
    eng = v2.Engine(D, cfg, inputs["trans"], hook, inputs["vht"], tto, None)
    hand = pa.groupby("pitcher").p_throws.agg(lambda s: int(s.mean() >= 0.5)).to_dict()
    st = pd.DataFrame(dict(b=D.bi, t=D.throws, s=D.stand)).groupby(["b", "t"]).s.mean()
    eng.prepare(hand, {k: int(v >= 0.5) for k, v in st.items()})
    eng.fit(pd.Timestamp(target))
    return eng


def run_slate(target: date, dry_run: bool = False, experimental: bool = False,
              n_episodes: int | None = None, bq=None, game_pks=None,
              now: datetime | None = None) -> dict:
    coefs = load_coefs()
    n_episodes = int(n_episodes or os.environ.get("SIM_BLEND_EPISODES", coefs["n_episodes"]))
    out = {"step": "sim_blend", "date": str(target), "tables": [PRED_TABLE, PROPS_TABLE],
           "model_version": coefs["model_version"], "n_episodes": n_episodes}
    ok, have, need = memory_ok()
    if not ok:
        return {**out, "status": "insufficient_memory", "memory_mb": have, "required_mb": need,
                "reason": "v2 sim peaks at 1.3-2.2 GB; run it in a >=4 GiB function or job"}

    from google.cloud import bigquery
    from pa_sim import v2, v2_inputs, strength

    bq = bq or bigquery.Client(project=PROJECT)
    slate = _slate(bq, target)
    if game_pks and not slate.empty:
        slate = slate[slate.game_pk.isin([int(g) for g in game_pks])].reset_index(drop=True)
    now = now or datetime.now(timezone.utc)
    if not dry_run and not slate.empty:
        started = slate.game_time_utc.map(_utc) <= _utc(now)
        out["skipped_started"] = int(started.sum())
        slate = slate[~started].reset_index(drop=True)
    if slate.empty:
        return {**out, "status": "no_games"}
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("cutoff", "DATE", target),
        bigquery.ScalarQueryParameter("min_year", "INT64", 2015)])
    games = bq.query(strength.GAMES_SQL.format(proj=PROJECT, hist=v2_inputs.HIST_DATASET, ds=DATASET),
                     job_config=cfg).to_dataframe()
    sp = strength_for_slate(games, slate, target)

    inputs = v2_inputs.load_inputs(bq, target)
    eng = build_engine(inputs, target, coefs)
    specs = [v2.GameSpec(r.home_lineup, r.away_lineup, r.home_starter_id, r.away_starter_id,
                         r.home_ab, r.away_ab, r.venue_id, 9, True, np.nan)
             for r in slate.itertuples()]
    res = v2.simulate(eng, specs, n=n_episodes, seed=int(pd.Timestamp(target).value // 10 ** 9) % 100000)
    summary = v2.summarize(res, len(specs))
    pred, props = game_rows(summary, slate, sp, coefs, n_episodes, now, experimental)
    out["games"] = len(pred)
    if dry_run:
        return {**out, "status": "dry_run", "sample": [
            {k: v for k, v in r.items() if not isinstance(v, (datetime, date))} for r in pred[:3]]}
    try:
        write(bq, pred, props)
    except Exception as e:
        if "Not found: Table" in str(e) or "notFound" in str(e):
            return {**out, "status": "table_missing", "error": str(e)[:300]}
        raise
    return {**out, "status": "ok"}


def _utc(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def write(bq, pred: list, props: list) -> None:
    """Append both tables (load jobs, CREATE_NEVER). Never deletes."""
    from google.cloud import bigquery

    for table, rows in ((PRED_TABLE, pred), (PROPS_TABLE, props)):
        tbl = f"{PROJECT}.{DATASET}.{table}"
        bq.load_table_from_dataframe(
            pd.DataFrame(rows), tbl,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND",
                                              create_disposition="CREATE_NEVER")).result()
