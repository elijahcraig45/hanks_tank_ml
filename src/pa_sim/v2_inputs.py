"""Inputs for the v2 simulator, ported from research/backtest_2026/39_, 41_ and 42_.

The research scripts build these tables from local parquet files. This module is the
same logic as importable functions plus BigQuery loaders, so the production runner
(pa_sim/blend.py) and the research harness agree on what a plate appearance is.

    classify()            9-class outcome (v2.CLASSES) from Savant event names
    prepare_hist/cur()    starter flag, top/bottom, per-season schema differences
    build_pa_table()      one row per PA, 2015 -> cutoff (v2.Data's input)
    build_transitions()   empirical base-out transitions (historical seasons only)
    build_hook()          starter-removal opportunities for the hook hazard
    est_tto()             times-through-order multipliers, from 2015 only
    venue_home_team()     venue -> the statcast abbreviation that plays home there

Known differences from research, all documented in docs/SHADOW_MODELS.md:
  * venue ids come from mlb_historical_data.games_historical / mlb_2026_season.games,
    not statsapi schedule metadata (same ids, fewer HTTP calls);
  * scheduled innings are always 9 (7-inning doubleheaders ended after 2021);
  * weather is not loaded: the frozen config has weather=False.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from pa_sim import v2
from pa_sim.v2 import NC

logger = logging.getLogger(__name__)

PROJECT = os.environ.get("GCP_PROJECT", "hankstank")
DATASET = os.environ.get("MLB_2026_DATASET", "mlb_2026_season")
HIST_DATASET = os.environ.get("MLB_HIST_DATASET", "mlb_historical_data")

DROP = {"truncated_pa", "ejection", "game_advisory"}

HIST_SQL = """
WITH p AS (
  SELECT game_pk, game_date, game_year, at_bat_number, pitch_number, batter, pitcher,
         stand, p_throws, inning, inning_topbot, outs_when_up,
         on_1b IS NOT NULL AS r1, on_2b IS NOT NULL AS r2, on_3b IS NOT NULL AS r3,
         events, bb_type, launch_speed, launch_angle,
         bat_score, post_bat_score, home_team, away_team, n_thruorder_pitcher,
         post_home_score, post_away_score
  FROM `{proj}.{ds}.statcast_pitches`
  WHERE game_type = 'R' AND game_year >= @min_year AND game_date < @cutoff
),
np AS (SELECT game_pk, at_bat_number, COUNT(*) AS n_pitches FROM p GROUP BY 1, 2)
SELECT p.* EXCEPT(pitch_number), np.n_pitches
FROM p JOIN np USING (game_pk, at_bat_number)
WHERE p.events IS NOT NULL AND p.events != ''
"""

# The 2026 table lacks at_bat_number / on_Xb / post scores, so it supplies outcomes,
# batted-ball quality and pitch counts only — exactly as in research.
CUR_SQL = """
WITH p AS (
  SELECT game_pk, game_date, game_year, batter, pitcher, stand, p_throws, inning,
         inning_topbot, outs_when_up, events, launch_speed, launch_angle,
         home_team, away_team
  FROM `{proj}.{ds}.statcast_pitches`
  WHERE game_type = 'R' AND game_date < @cutoff
),
np AS (SELECT game_pk, inning, inning_topbot, batter, pitcher, COUNT(*) AS n_pitches
       FROM p GROUP BY 1, 2, 3, 4, 5)
SELECT p.*, np.n_pitches
FROM p JOIN np USING (game_pk, inning, inning_topbot, batter, pitcher)
WHERE p.events IS NOT NULL AND p.events != ''
"""

VENUE_SQL = """
SELECT game_pk, ANY_VALUE(venue_id) AS venue_id FROM `{proj}.{hist}.games_historical`
WHERE venue_id IS NOT NULL GROUP BY game_pk
UNION ALL
SELECT game_pk, ANY_VALUE(venue_id) AS venue_id FROM `{proj}.{ds}.games`
WHERE venue_id IS NOT NULL GROUP BY game_pk
"""


# --------------------------------------------------------------------------- pure
def classify(df: pd.DataFrame, air: np.ndarray) -> np.ndarray:
    """K BB 1B 2B 3B HR E GO AO -> 0..8; -1 for events the simulator does not model."""
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


def prepare_hist(h: pd.DataFrame) -> pd.DataFrame:
    """Historical PAs: class, top flag, starter = first pitcher of each defence."""
    h = h[~h.events.isin(DROP)].copy()
    air = h.bb_type.isin(["fly_ball", "line_drive", "popup"]).values if "bb_type" in h else \
        np.zeros(len(h), bool)
    if "bb_type" in h:
        air = np.where(h.bb_type.isna(), (h.launch_angle.fillna(20) >= 10).values, air)
    h["cls"] = classify(h, air)
    h = h[h.cls >= 0]
    h = h.sort_values(["game_pk", "at_bat_number"]).reset_index(drop=True)
    h["top"] = h.inning_topbot.eq("Top")
    h["_first"] = h.groupby(["game_pk", "top"]).pitcher.transform("first")
    h["is_starter"] = h.pitcher.eq(h["_first"])
    return h


def prepare_cur(c: pd.DataFrame) -> pd.DataFrame:
    """Current-season PAs: no at-bat order, so the starter is the pitcher seen most in
    inning 1 for that defence."""
    c = c[~c.events.isin(DROP)].copy()
    c["cls"] = classify(c, (c.launch_angle.fillna(20) >= 10).values)
    c = c[c.cls >= 0]
    c["top"] = c.inning_topbot.eq("Top")
    fi = c[c.inning == 1].groupby(["game_pk", "top"]).pitcher.agg(lambda s: s.value_counts().index[0])
    c = c.join(fi.rename("sp"), on=["game_pk", "top"])
    c["is_starter"] = c.pitcher.eq(c.sp)
    return c


PA_COLS = ["game_pk", "game_date", "game_year", "batter", "pitcher", "stand", "p_throws", "inning",
           "top", "home_team", "away_team", "cls", "n_pitches", "launch_speed", "launch_angle",
           "is_starter"]


def build_pa_table(h: pd.DataFrame, c: pd.DataFrame | None) -> pd.DataFrame:
    parts = [h[PA_COLS + ["n_thruorder_pitcher"]]]
    if c is not None and len(c):
        parts.append(c[PA_COLS])
    pa = pd.concat(parts, ignore_index=True)
    pa["game_date"] = pd.to_datetime(pa.game_date)
    pa["stand"] = (pa.stand == "R").astype(np.int8)
    pa["p_throws"] = (pa.p_throws == "R").astype(np.int8)
    for k in ("batter", "pitcher", "game_pk"):
        pa[k] = pa[k].astype(np.int64)
    pa["inning"] = pa.inning.astype(np.int16)
    pa["game_year"] = pa.game_year.astype(np.int16)
    return pa.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def build_transitions(h: pd.DataFrame) -> pd.DataFrame:
    """(bases, outs, cls) -> (next bases, next outs, runs) from Savant state."""
    h = h.copy()
    h["bases"] = (h.r1.astype(int) + 2 * h.r2.astype(int) + 4 * h.r3.astype(int)).astype(np.int8)
    g = h.groupby(["game_pk", "inning", "top"], sort=False)
    nxt_b = g.bases.shift(-1); nxt_o = g.outs_when_up.shift(-1); nxt_s = g.bat_score.shift(-1)
    last = nxt_b.isna()
    runs = np.where(last, h.post_bat_score - h.bat_score, nxt_s - h.bat_score)
    no = np.where(last, 3, nxt_o)
    nb = np.where(last, 0, nxt_b)
    maxinn = h.groupby("game_pk").inning.transform("max")
    walk = last & ~h.top & (h.inning >= 9) & (h.inning == maxinn) & (h.post_home_score > h.post_away_score)
    ok = ~walk & (runs >= 0) & (runs <= 4) & (no >= h.outs_when_up)
    T = pd.DataFrame(dict(year=h.game_year.values, bases=h.bases.values, outs=h.outs_when_up.values,
                          cls=h.cls.values, nb=nb, no=no, runs=runs))[ok.values]
    return T.astype(np.int16)


def build_hook(h: pd.DataFrame) -> pd.DataFrame:
    """Starter PAs in order; y=1 on the first PA after the starter is gone."""
    h = h.copy()
    h["bases"] = (h.r1.astype(int) + 2 * h.r2.astype(int) + 4 * h.r3.astype(int)).astype(np.int8)
    grp = h.groupby(["game_pk", "top"])
    h["cum_p"] = grp.n_pitches.cumsum() - h.n_pitches
    h["bf"] = grp.cumcount()
    h["ra"] = h.bat_score - grp.bat_score.transform("first")
    h["inn_start"] = h.groupby(["game_pk", "inning", "top"]).cumcount().eq(0)
    st = h.is_starter.values
    prev_st = grp.is_starter.shift(1).fillna(True).astype(bool).values
    gone_before = (~h.is_starter).groupby([h.game_pk, h.top]).cumsum().values - (~st).astype(int)
    opp = prev_st & (gone_before == 0)
    hk = h.loc[opp, ["game_pk", "game_date", "game_year", "top", "pitcher", "inning", "outs_when_up",
                     "cum_p", "bf", "ra", "inn_start", "bases"]].copy()
    hk["sp"] = h["_first"][opp].values
    hk["y"] = (~h.is_starter[opp]).astype(np.int8).values
    return hk


def est_tto(pa15: pd.DataFrame) -> np.ndarray:
    """Class multipliers for a starter's 1st/2nd/3rd+ time through, vs log5 (2015 only)."""
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
        if (t == k).any():
            m[k] = oh[t == k].sum(0) / np.maximum(e[t == k].sum(0), 1e-9)
    avg = (m * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    return m / np.where(avg > 0, avg, 1.0)


def venue_home_team(pa: pd.DataFrame, venue_of: dict) -> dict:
    """venue_id -> the abbreviation that most often bats last there."""
    v = pa[["game_pk", "home_team"]].drop_duplicates("game_pk")
    v["venue"] = v.game_pk.map(venue_of)
    v = v.dropna(subset=["venue"])
    return v.groupby("venue").home_team.agg(lambda s: s.value_counts().index[0]).astype(str) \
        .rename(index=int).to_dict()


# --------------------------------------------------------------------------- I/O
def load_inputs(bq, cutoff, min_year: int | None = None) -> dict:
    """Everything v2.Engine needs, from BigQuery, using only data before `cutoff`.

    Peak memory is the cost: ~2M PAs plus derived tables measured 1.3-2.2 GB, which is
    why the caller refuses to run in the 1 GB function (see blend.memory_ok).
    """
    from google.cloud import bigquery

    min_year = int(min_year or os.environ.get("PA_SIM_V2_MIN_YEAR", "2015"))
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("cutoff", "DATE", cutoff),
        bigquery.ScalarQueryParameter("min_year", "INT64", min_year)])
    h = bq.query(HIST_SQL.format(proj=PROJECT, ds=HIST_DATASET), job_config=cfg).to_dataframe()
    cfg_c = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("cutoff", "DATE", cutoff)])
    c = bq.query(CUR_SQL.format(proj=PROJECT, ds=DATASET), job_config=cfg_c).to_dataframe()
    venues = bq.query(VENUE_SQL.format(proj=PROJECT, hist=HIST_DATASET, ds=DATASET)).to_dataframe()
    return assemble(h, c, dict(zip(venues.game_pk.astype(int), venues.venue_id.astype(int))))


def assemble(h_raw: pd.DataFrame, c_raw: pd.DataFrame | None, venue_of: dict) -> dict:
    h = prepare_hist(h_raw)
    c = prepare_cur(c_raw) if c_raw is not None and len(c_raw) else None
    pa = build_pa_table(h, c)
    return dict(pa=pa, trans=build_transitions(h), hook=build_hook(h), venue_of=venue_of,
                vht=venue_home_team(pa, venue_of))
