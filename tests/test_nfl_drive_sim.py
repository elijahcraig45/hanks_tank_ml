"""NFL drive-simulator shadow: determinism, summaries, flags, write safety, table shape.

Everything here runs on synthetic drives (no network, no BigQuery). The one test that
checks the port against the research outputs needs NFL_DRIVE_SIM_RESEARCH_DIR pointing at
the research run directory (nfl_drives.parquet, sim_c_g_N4000_s0_2010_2025.parquet, and
../nfl.csv) and is skipped without it.
"""
import importlib.util
import json
import os
import re
import sys
import unittest.mock

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)                       # rankings/ and stats/ ride along in deploys
sys.path.insert(0, os.path.join(SRC, "nfl"))

import drive_sim as ds  # noqa: E402
import predict_nfl  # noqa: E402

TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
DDL = os.path.join(HERE, "..", "scripts", "gcp", "football", "create_drive_sim_tables.sql")


def synthetic_drives(seasons=(2025,), weeks=range(1, 18), seed=7, extra=()):
    """Plausible-looking drive rows for 6 teams; `extra` adds (season, week) blocks."""
    rng = np.random.default_rng(seed)
    probs = np.array([.22, .15, .03, .38, .11, .01, .06, .01, .03])
    pts = {"TD": 7, "FG": 3}
    rows = []
    blocks = [(s, w) for s in seasons for w in weeks] + list(extra)
    for s, w in blocks:
        order = rng.permutation(TEAMS)
        for gi in range(3):
            home, away = order[2 * gi], order[2 * gi + 1]
            gid = f"{s}_{w:02d}_{away}_{home}"
            hs = as_ = 0
            fd = 0
            for half in ("Half1", "Half2"):
                clock = 1800.0
                pos = int(rng.integers(0, 2))
                while clock > 0:
                    fd += 1
                    off, de = (home, away) if pos == 0 else (away, home)
                    res = ds.OUT[rng.choice(len(ds.OUT), p=probs)]
                    dur = float(min(clock, rng.uniform(40, 250)))
                    hs0, as0 = hs, as_
                    if res in pts:
                        if pos == 0:
                            hs += pts[res]
                        else:
                            as_ += pts[res]
                    rows.append(dict(game_id=gid, fixed_drive=float(fd), season=s, week=w,
                                     season_type="REG", home_team=home, away_team=away,
                                     posteam=off, defteam=de, qtr=1.0 if half == "Half1" else 3.0,
                                     game_half=half, res=res, hsr0=clock,
                                     yl=float(rng.integers(20, 90)), hs0=float(hs0), as0=float(as0),
                                     hs1=float(hs), as1=float(as_), off_home=(pos == 0)))
                    clock -= dur
                    pos = 1 - pos
    d = pd.DataFrame(rows)
    d["dh"] = d.hs1 - d.hs0
    d["da"] = d.as1 - d.as0
    d["off_pts"] = np.where(d.off_home, d.dh, d.da)
    d["def_pts"] = np.where(d.off_home, d.da, d.dh)
    d["sd0"] = np.where(d.off_home, d.hs0 - d.as0, d.as0 - d.hs0)
    g = d.groupby(["game_id", "game_half"])
    d["hsr_next"] = g.hsr0.shift(-1)
    d["yl_next"] = g.yl.shift(-1)
    d["dur"] = (d.hsr0 - d.hsr_next.fillna(0)).clip(0, 1800)
    d["yl_end"] = d.yl
    return d


def schedule(season=2026, week=3):
    return pd.DataFrame({
        "game_id": [f"{season}_{week:02d}_BBB_AAA", f"{season}_{week:02d}_DDD_CCC",
                    f"{season}_{week:02d}_FFF_EEE"],
        "season": season, "week": week, "game_type": "REG", "location": "Home",
        "home_team": ["AAA", "CCC", "EEE"], "away_team": ["BBB", "DDD", "FFF"],
        "gameday": [f"{season}-09-24", f"{season}-09-27", f"{season}-09-28"],
        "gametime": ["20:15", "13:00", "20:15"],
        "result": [np.nan, np.nan, np.nan],
        "spread_line": [3.0, np.nan, -2.5], "total_line": [44.5, np.nan, 41.0],
    })


@pytest.fixture(scope="module")
def drives():
    return synthetic_drives(seasons=(2024, 2025), extra=[(2026, 1), (2026, 2)])


@pytest.fixture(scope="module")
def model(drives):
    return ds.fit_week(ds.prep_drives(drives, schedule()), schedule(), 2026, 3)


# ---------------------------------------------------------------- determinism
def test_same_seed_same_game_is_bit_identical(model):
    a = ds.simulate(model, "AAA", "BBB", False, 2026, "REG", N=500, rng=ds.game_rng("g1"))
    b = ds.simulate(model, "AAA", "BBB", False, 2026, "REG", N=500, rng=ds.game_rng("g1"))
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_different_game_ids_get_different_streams(model):
    a = ds.simulate(model, "AAA", "BBB", False, 2026, "REG", N=500, rng=ds.game_rng("g1"))
    b = ds.simulate(model, "AAA", "BBB", False, 2026, "REG", N=500, rng=ds.game_rng("g2"))
    assert not np.array_equal(a[0], b[0])


def test_scores_are_football_shaped(model):
    hs, aw, ot = ds.simulate(model, "AAA", "BBB", False, 2026, "REG", N=2000,
                             rng=ds.game_rng("g1"))
    assert (hs >= 0).all() and (aw >= 0).all()
    assert 5 < hs.mean() < 45 and 5 < aw.mean() < 45
    assert 0 <= ot.mean() < 0.3


def test_fit_refuses_without_enough_history(drives):
    few = drives[drives.week <= 2]
    with pytest.raises(RuntimeError):
        ds.fit_week(ds.prep_drives(few, schedule()), schedule(), 2026, 3)


# ---------------------------------------------------------------- summaries
def test_dist_integer_percentiles():
    x = np.arange(0, 101)                 # 0..100, uniform
    d = ds.dist(x)
    assert d["p50"] == 50 and d["p05"] == 5 and d["p95"] == 95
    assert d["min"] == 0 and d["max"] == 100 and d["n"] == 101
    assert all(isinstance(d[k], int) for k in ("p05", "p25", "p50", "p75", "p95", "min", "max"))
    assert abs(d["mean"] - 50) < 1e-9 and abs(d["sd"] - np.std(x)) < 1e-9


def test_summarize_game_lines_and_margin_exact():
    rng = np.random.default_rng(0)
    hs = rng.poisson(24, 4000).astype(float)
    aw = rng.poisson(21, 4000).astype(float)
    ot = np.zeros(4000, bool)
    s = ds.summarize_game(hs, aw, ot, spread_line=3.0, total_line=44.5)
    lines = json.loads(s["p_over_by_line"])
    assert list(lines) == [f"{x:g}" for x in np.arange(37.5, 52.5, 1.0)]
    vals = list(lines.values())
    assert all(a >= b for a, b in zip(vals, vals[1:]))           # P(over) falls with the line
    me = json.loads(s["margin_exact"])
    assert list(me) == [str(k) for k in range(-21, 22)]
    assert 0.9 < sum(me.values()) <= 1.0
    assert s["margin_exact_basis"] == "sim_shape_at_spread"
    # tilted to the spread: the full-grid mean is 3.0 (checked via the pmf directly)
    assert s["p_home_cover"] == pytest.approx(((hs - aw) > 3.0).mean())
    raw = ds.summarize_game(hs, aw, ot)
    assert raw["p_home_cover"] is None and raw["margin_exact_basis"] == "raw_sim"
    assert len(json.loads(raw["p_over_by_line"])) == 15


def test_tilt_hits_target_mean():
    P = np.full(len(ds.MARGIN_GRID), 1.0 / len(ds.MARGIN_GRID))
    q = ds.tilt(P, ds.MARGIN_GRID, 3.0)
    assert (q * ds.MARGIN_GRID).sum() == pytest.approx(3.0, abs=1e-3)


def test_platt_is_monotone_and_near_identity_at_half():
    p = ds.calibrate([0.3, 0.5, 0.7])
    assert p[0] < p[1] < p[2]
    assert abs(p[1] - 0.5) < 0.01


# ---------------------------------------------------------------- predict path
NOW = pd.Timestamp("2026-09-27 18:00", tz="UTC")   # AAA game (Thu) and CCC game (13:00) started


def test_predict_skips_started_games(drives):
    dist, pred, info = predict_nfl.predict_week_drive_sim(
        2026, 3, schedule=schedule(), drives=drives, now=NOW, n_sims=300)
    assert dist["game_id"].tolist() == ["2026_03_FFF_EEE"]
    assert pred["game_id"].tolist() == ["2026_03_FFF_EEE"]
    assert (dist["predicted_at"] == NOW.to_pydatetime()).all()


def test_predict_rows_match_ddl(drives):
    dist, pred, _ = predict_nfl.predict_week_drive_sim(
        2026, 3, schedule=schedule(), drives=drives,
        now=pd.Timestamp("2026-09-20", tz="UTC"), n_sims=300)
    assert len(dist) == 3
    ddl = open(DDL).read()
    for table, df in (("game_sim_distributions", dist), ("game_predictions_drive_sim", pred)):
        body = re.search(rf"`hankstank\.nfl_season\.{table}` \((.*?)\n\)", ddl, re.S).group(1)
        cols = set(re.findall(r"^\s*([a-z0-9_]+) [A-Z]", body, re.M))
        cols |= set(re.findall(r",\s*([a-z0-9_]+) [A-Z]", body))
        assert set(df.columns) == cols, (table, set(df.columns) ^ cols)
    assert dist["model_version"].eq("drive_sim_v1").all()
    assert dist.loc[dist.game_id == "2026_03_DDD_CCC", "p_home_cover"].isna().all()


def test_predict_refuses_without_prior_season_drives(drives):
    with pytest.raises(RuntimeError):
        predict_nfl.predict_week_drive_sim(2027, 3, schedule=schedule(2027), drives=drives,
                                           now=pd.Timestamp("2027-09-01", tz="UTC"))


# ---------------------------------------------------------------- flags + write safety
def _load_main():
    spec = importlib.util.spec_from_file_location("nfl_main_drive_sim_test",
                                                  os.path.join(SRC, "nfl", "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Req:
    def __init__(self, body):
        self._body = body

    def get_json(self, silent=True):
        return self._body


def test_flag_from_body_or_env(monkeypatch):
    main = _load_main()
    monkeypatch.delenv("NFL_DRIVE_SIM_SHADOW", raising=False)
    assert not main._drive_sim_enabled({})
    assert main._drive_sim_enabled({"shadow_drive_sim": True})
    monkeypatch.setenv("NFL_DRIVE_SIM_SHADOW", "1")
    assert main._drive_sim_enabled({})
    monkeypatch.setenv("NFL_DRIVE_SIM_SHADOW", "0")
    assert not main._drive_sim_enabled({})


def _frames():
    ids = ["2026_03_FFF_EEE"]
    return (pd.DataFrame({"game_id": ids, "season": 2026, "week": 3}),
            pd.DataFrame({"game_id": ids, "season": 2026, "week": 3}),
            {"fit_s": 0.1, "games": 1})


def _run(body, monkeypatch):
    import bq_io

    main = _load_main()
    monkeypatch.delenv("NFL_DRIVE_SIM_SHADOW", raising=False)
    monkeypatch.delenv("NFL_RIDGE_SHADOW", raising=False)
    monkeypatch.delenv("FPI_SNAPSHOT", raising=False)
    rows = pd.DataFrame({"game_id": ["2026_03_FFF_EEE"]})
    with unittest.mock.patch.object(predict_nfl, "predict_week", return_value=rows), \
         unittest.mock.patch.object(predict_nfl, "predict_week_drive_sim",
                                    return_value=_frames()) as sim, \
         unittest.mock.patch.object(bq_io, "upsert_week", return_value=1) as up, \
         unittest.mock.patch.object(bq_io, "ensure_dataset"):
        out, status = main.nfl_pipeline(_Req(body))
    return out, status, sim, up


def test_dry_run_computes_but_writes_nothing(monkeypatch):
    out, status, sim, up = _run({"mode": "predict_week", "season": 2026, "week": 3,
                                 "dry_run": True, "shadow_drive_sim": True}, monkeypatch)
    assert status == 200 and out["dry_run"] is True
    sim.assert_called_once()
    up.assert_not_called()
    assert out["steps"]["shadow_drive_sim"]["written"] == 0


def test_shadow_off_by_default(monkeypatch):
    out, status, sim, up = _run({"mode": "predict_week", "season": 2026, "week": 3}, monkeypatch)
    assert status == 200
    sim.assert_not_called()
    assert [c.args[2] for c in up.call_args_list] == ["game_predictions"]


def test_shadow_writes_both_tables_game_scoped_and_create_never(monkeypatch):
    out, status, sim, up = _run({"mode": "predict_week", "season": 2026, "week": 3,
                                 "shadow_drive_sim": True}, monkeypatch)
    assert status == 200
    tables = {c.args[2]: c for c in up.call_args_list}
    assert set(tables) == {"game_predictions", "game_sim_distributions",
                           "game_predictions_drive_sim"}
    for t in ("game_sim_distributions", "game_predictions_drive_sim"):
        c = tables[t]
        assert c.args[1] == "nfl_season" and c.args[3:5] == (2026, 3)
        assert c.kwargs["create_disposition"] == "CREATE_NEVER"
        assert c.args[0]["game_id"].tolist() == ["2026_03_FFF_EEE"]


def test_shadow_failure_is_not_fatal(monkeypatch):
    import bq_io

    main = _load_main()
    rows = pd.DataFrame({"game_id": ["x"]})
    with unittest.mock.patch.object(predict_nfl, "predict_week", return_value=rows), \
         unittest.mock.patch.object(predict_nfl, "predict_week_drive_sim",
                                    side_effect=SystemExit("every game has kicked off")), \
         unittest.mock.patch.object(bq_io, "upsert_week", return_value=1), \
         unittest.mock.patch.object(bq_io, "ensure_dataset"):
        out, status = main.nfl_pipeline(_Req({"mode": "predict_week", "season": 2026, "week": 3,
                                              "shadow_drive_sim": True}))
    assert status == 200 and "error" in out["steps"]["shadow_drive_sim"]


def test_drives_extract_matches_contracted_columns():
    pl = pytest.importorskip("polars")
    import drives as dv

    n = 6
    pbp = pl.DataFrame({
        "game_id": ["g"] * n, "season": [2026] * n, "week": [1] * n,
        "season_type": ["REG"] * n, "home_team": ["AAA"] * n, "away_team": ["BBB"] * n,
        "posteam": ["AAA", "AAA", "BBB", "BBB", "AAA", "AAA"],
        "defteam": ["BBB", "BBB", "AAA", "AAA", "BBB", "BBB"],
        "qtr": [1, 1, 1, 1, 3, 3], "game_half": ["Half1"] * 4 + ["Half2"] * 2,
        "half_seconds_remaining": [1800, 1750, 1700, 1650, 1800, 1700],
        "play_type": ["run", "pass", "run", "punt", "pass", "field_goal"],
        "yardline_100": [75, 60, 70, 55, 75, 20], "fixed_drive": [1, 1, 2, 2, 3, 3],
        "fixed_drive_result": ["Touchdown", "Touchdown", "Punt", "Punt", "Field goal", "Field goal"],
        "total_home_score": [0, 7, 7, 7, 7, 10], "total_away_score": [0] * n,
        "play_id": list(range(1, n + 1)),
    })
    d = dv.extract_drives(pbp).to_pandas()
    assert list(d.columns) == ds.DRIVE_COLUMNS
    assert d.res.tolist() == ["TD", "PUNT", "FG"]
    assert d.off_pts.tolist() == [7, 0, 3]
    assert dv.extract_drives(pl.DataFrame()).height == 0


# ---------------------------------------------------------------- research reproduction
RESEARCH = os.environ.get("NFL_DRIVE_SIM_RESEARCH_DIR")


@pytest.mark.skipif(not RESEARCH, reason="set NFL_DRIVE_SIM_RESEARCH_DIR to the research run dir")
def test_port_reproduces_research_week():
    g = pd.read_csv(os.path.join(RESEARCH, "..", "nfl.csv"))
    d = ds.prep_drives(pd.read_parquet(os.path.join(RESEARCH, "nfl_drives.parquet")), g)
    ref = pd.read_parquet(os.path.join(RESEARCH, "sim_c_g_N4000_s0_2010_2025.parquet")).set_index("game_id")
    m = ds.fit_week(d, g, 2025, 10)
    G = g[(g.season == 2025) & (g.week == 10) & g.result.notna() & (g.result != 0)]
    for r in G.itertuples():
        hs, aw, _ = ds.simulate(m, r.home_team, r.away_team, r.location == "Neutral", r.season,
                                r.game_type, N=4000, rng=ds.game_rng(r.game_id))
        mg = hs - aw
        assert np.array_equal(np.bincount(np.clip(mg, -80, 80).astype(int) + 80, minlength=161),
                              np.asarray(ref.loc[r.game_id].mh))
