"""College drive-simulator shadow: config, CFBD transform, overtime, write safety, DDL.

Synthetic data only (no network, no BigQuery). The research reproduction check (254/254
games bit-identical to research/football_2026_09/cfb_drive_sim runs) needs the research
run directory and is skipped without CFB_DRIVE_SIM_RESEARCH_DIR.
"""
import importlib.util
import os
import re
import sys
import unittest.mock

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "nfl"))
sys.path.insert(0, os.path.join(SRC, "cfb"))

import drive_sim as ds  # noqa: E402
import cfb_drives  # noqa: E402
import cfb_drive_sim  # noqa: E402

DDL = os.path.join(HERE, "..", "scripts", "gcp", "football", "create_cfb_drive_sim_tables.sql")
FBS = ["AAA", "BBB", "CCC", "DDD"]
FCS = ["EEE", "FFF"]
TEAMS = FBS + FCS
NOW = pd.Timestamp("2026-09-26T15:00", tz="UTC")


def synthetic_drives(seasons=(2024, 2025), weeks=range(1, 16), extra=(), seed=3):
    rng = np.random.default_rng(seed)
    probs = np.array([.26, .09, .03, .36, .10, .01, .07, .003, .077])
    probs = probs / probs.sum()
    pts = {"TD": 7, "FG": 3}
    rows = []
    for s, w in [(s, w) for s in seasons for w in weeks] + list(extra):
        order = rng.permutation(TEAMS)
        for gi in range(3):
            home, away = order[2 * gi], order[2 * gi + 1]
            gid = f"{s}{w:02d}{gi}"
            score = {home: 0, away: 0}
            dn = 0
            for half in ("Half1", "Half2"):
                clock = 1800.0
                pos = int(rng.integers(0, 2))
                while clock > 0:
                    dn += 1
                    off, de = (home, away) if pos == 0 else (away, home)
                    res = ds.OUT[rng.choice(len(ds.OUT), p=probs)]
                    sd0 = score[off] - score[de]
                    score[off] += pts.get(res, 0)
                    rows.append(dict(game_id=gid, season=s, week=w, season_type="REG",
                                     home_team=home, away_team=away, posteam=off, defteam=de,
                                     qtr=1.0 if half == "Half1" else 3.0, game_half=half, res=res,
                                     hsr0=clock, yl=float(rng.integers(15, 90)), sd0=float(sd0),
                                     off_pts=float(pts.get(res, 0)), def_pts=0.0,
                                     off_home=(pos == 0), drive_number=float(dn),
                                     cfbd_offense=off, cfbd_defense=de, division="fbs"))
                    clock -= float(min(clock, rng.uniform(40, 230)))
                    pos = 1 - pos
    d = pd.DataFrame(rows)
    g = d.groupby(["game_id", "game_half"])
    d["yl_next"] = g.yl.shift(-1)
    d["dur"] = (d.hsr0 - g.hsr0.shift(-1).fillna(0)).clip(0, 1800)
    return d[cfb_drives.DRIVE_COLUMNS]


def played_games(drives):
    g = drives.drop_duplicates("game_id")[["game_id", "season", "week", "home_team", "away_team"]].copy()
    g["division"] = "fbs"
    g["home_won"] = 1
    g["neutral_site"] = 0
    g["is_postseason"] = 0
    g["cross_division"] = (g.home_team.isin(FCS) != g.away_team.isin(FCS)).astype(int)
    g.loc[g.home_team.isin(FCS) & g.away_team.isin(FCS), "division"] = "fcs"
    g["game_date"] = pd.Timestamp("2025-09-01")
    return g


def upcoming(season=2026, week=3):
    return pd.DataFrame({
        "game_id": ["u1", "u2", "u3"], "season": season, "week": week,
        "division": ["fbs", "fbs", "fcs"],
        "game_date": pd.to_datetime(["2026-09-26 12:00", "2026-09-26 19:30", "2026-09-27 00:00"]),
        "home_team": ["AAA", "CCC", "EEE"], "away_team": ["BBB", "DDD", "FFF"],
        "home_team_name": ["A U", "C U", "E U"], "away_team_name": ["B U", "D U", "F U"],
        "home_won": [np.nan] * 3, "neutral_site": [0, 1, 0],
    })


LINES = pd.DataFrame({"game_id": ["u2", "u3"], "spread_line": [3.5, -7.0],
                      "total_line": [51.5, 44.0]})


@pytest.fixture(scope="module")
def drives():
    return synthetic_drives(extra=[(2026, 1), (2026, 2)])


@pytest.fixture(scope="module")
def model(drives):
    g = played_games(drives)
    fbs = {t: int(t in FBS) for t in TEAMS}
    return ds.fit_week(ds.prep_drives(drives, g, ds.CFB), g, 2026, 3, cfg=ds.CFB, fbs=fbs)


# ---------------------------------------------------------------- config
def test_nfl_default_is_untouched():
    assert ds.DriveModel().cfg is ds.NFL
    assert ds.NFL.nsd == 9 and ds.NFL.tied == 4 and not ds.NFL.division and not ds.NFL.sparse
    assert ds.NFL.model_version == "drive_sim_v1" and ds.NFL.C == 0.03


def test_cfb_config_is_the_frozen_one():
    c = ds.CFB
    assert (c.variant, c.C, c.tau, c.ktau, c.prior_w) == ("c", 0.1, 16.0, 6.0, 1.0)
    assert c.nsd == 11 and c.tied == 5 and c.division and c.ot == "cfb"
    assert c.model_version == "cfb_drive_sim_v1"
    assert list(c.margin_grid[[0, -1]]) == [-100, 100]


# ---------------------------------------------------------------- CFBD transform
def test_classify_uses_score_deltas_for_ambiguous_labels():
    r = pd.Series(["TD", "PUNT", "FUMBLE TD", "FUMBLE TD", "Uncategorized", "Uncategorized",
                   "SF", "END OF GAME", "DOWNS", "INT TD"])
    dop = pd.Series([7, 0, 7, 0, 0, 3, 0, 0, 0, 0])
    dde = pd.Series([0, 0, 0, 7, 0, 0, 2, 0, 0, 0])
    got = cfb_drives.classify(r, dop, dde)
    assert got.isna().tolist() == [False] * 4 + [True] + [False] * 5
    assert got.dropna().tolist() == ["TD", "PUNT", "TD", "OTD", "FG", "SAF", "EOH", "TOD", "OTD"]


def _rec(gid, n, off, de, home_off, res, per=1, mins=15, s0=(0, 0), s1=(0, 0), ytg=75):
    return {"gameId": gid, "driveNumber": n, "offense": off, "defense": de,
            "isHomeOffense": home_off, "driveResult": res, "startPeriod": per,
            "startTime": {"minutes": mins, "seconds": 0}, "startYardsToGoal": ytg,
            "startOffenseScore": s0[0], "startDefenseScore": s0[1],
            "endOffenseScore": s1[0], "endDefenseScore": s1[1]}


def test_transform_orients_by_crosswalk_not_cfbd_home_flag():
    games = pd.DataFrame({"game_id": ["9"], "season": [2026], "week": [4], "is_postseason": [0],
                          "home_team": ["ARMY"], "away_team": ["NAVY"], "division": ["fbs"]})
    xw = {"Army": "ARMY", "Navy": "NAVY"}
    # CFBD calls Navy the home team here (neutral site); ESPN says Army.
    recs = [_rec(9, 1, "Navy", "Army", True, "TD", s1=(7, 0)),
            _rec(9, 2, "Army", "Navy", False, "PUNT", mins=11, ytg=70),
            _rec(9, 3, "Navy", "Army", True, "Uncategorized", mins=9)]
    d = cfb_drives.transform(recs, games, xw)
    assert list(d.columns) == cfb_drives.DRIVE_COLUMNS
    assert d.posteam.tolist() == ["NAVY", "ARMY"]            # Uncategorized, 0 pts: dropped
    assert d.off_home.tolist() == [False, True]
    assert d.res.tolist() == ["TD", "PUNT"] and d.off_pts.tolist() == [7.0, 0.0]
    assert d.week.tolist() == [4, 4] and d.hsr0.tolist() == [1800.0, 1560.0]
    assert d.yl_next.iloc[0] == 70 and d.dur.iloc[0] == 240
    assert cfb_drives.transform([], games, xw).empty


def test_committed_crosswalk_loads():
    xw = cfb_drives.load_crosswalk()
    assert len(xw) > 300 and xw.get("Alabama") == "ALA"


def test_missing_weeks_only_completed_unstored():
    g = pd.DataFrame({"game_id": ["a", "b", "c", "d"], "season": 2026, "week": [1, 2, 3, 4],
                      "home_won": [1, 0, 1, np.nan], "is_postseason": 0})
    have = pd.DataFrame({"game_id": ["a"]})
    assert cfb_drives.missing_weeks(g, have, 2026) == [2, 3]


# ---------------------------------------------------------------- simulation
def test_college_overtime_always_produces_a_winner(model):
    hs, aw, ot = ds.simulate(model, "AAA", "BBB", False, 2026, "REG", N=3000,
                             rng=ds.game_rng("x"))
    assert (hs != aw).all()
    assert 0 < ot.mean() < 0.2
    assert 5 < hs.mean() < 60


def test_same_seed_is_bit_identical(model):
    a = ds.simulate(model, "AAA", "EEE", True, 2026, "REG", N=500, rng=ds.game_rng("g"))
    b = ds.simulate(model, "AAA", "EEE", True, 2026, "REG", N=500, rng=ds.game_rng("g"))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_sparse_and_dense_designs_agree(drives):
    import dataclasses

    g = played_games(drives)
    fbs = {t: int(t in FBS) for t in TEAMS}
    out = []
    for sparse in (True, False):
        cfg = dataclasses.replace(ds.CFB, sparse=sparse)
        m = ds.fit_week(ds.prep_drives(drives, g, cfg), g, 2026, 3, cfg=cfg, fbs=fbs)
        out.append(ds.simulate(m, "CCC", "FFF", False, 2026, "REG", N=800, rng=ds.game_rng("s")))
    assert abs(out[0][0].mean() - out[1][0].mean()) < 0.5


# ---------------------------------------------------------------- predict rows
def _predict(drives, now=NOW, up=None):
    return cfb_drive_sim.predict_week_drive_sim(
        2026, 3, played=played_games(drives), upcoming=upcoming() if up is None else up,
        drives=drives, lines=LINES, now=now, n_sims=300)


def test_predict_skips_started_games(drives):
    dist, pred, info = _predict(drives)          # u1 kicked off at 12:00Z < now
    assert dist.game_id.tolist() == ["u2", "u3"] == pred.game_id.tolist()
    assert info["skipped_started_or_final"] == 1
    assert (dist["predicted_at"] == NOW.to_pydatetime()).all()
    assert (pred["game_date"] > NOW).all()


def test_predict_everything_started_returns_empty(drives):
    dist, pred, info = _predict(drives, now=pd.Timestamp("2026-09-28", tz="UTC"))
    assert dist.empty and pred.empty and info["games"] == 0


def test_predict_rows_match_ddl(drives):
    dist, pred, _ = _predict(drives, now=pd.Timestamp("2026-09-20", tz="UTC"))
    assert len(dist) == 3
    ddl = open(DDL).read()
    for table, df in (("game_sim_distributions", dist), ("game_predictions_drive_sim", pred)):
        body = re.search(rf"`hankstank\.cfb_season\.{table}` \((.*?)\n\)", ddl, re.S).group(1)
        cols = set(re.findall(r"^\s*([a-z0-9_]+) [A-Z]", body, re.M))
        cols |= set(re.findall(r",\s*([a-z0-9_]+) [A-Z]", body))
        assert set(df.columns) == cols, (table, set(df.columns) ^ cols)
    assert dist.model_version.eq("cfb_drive_sim_v1").all()
    assert dist.set_index("game_id").loc["u1", "margin_exact_basis"] == "raw_sim"
    assert dist.set_index("game_id").loc["u2", "margin_exact_basis"] == "sim_shape_at_spread"
    assert dist.division.tolist() == ["fbs", "fbs", "fcs"]
    assert pred.home_team_name.tolist() == ["A U", "C U", "E U"]
    assert (dist.p_tie == 0).all()


def test_ddl_mirrors_nfl_tables_plus_division():
    nfl = open(os.path.join(HERE, "..", "scripts", "gcp", "football",
                            "create_drive_sim_tables.sql")).read()
    cfb = open(DDL).read()

    def cols(sql, ds_, table):
        body = re.search(rf"`hankstank\.{ds_}\.{table}` \((.*?)\n\)", sql, re.S).group(1)
        return re.findall(r"(?:^|,)\s*([a-z0-9_]+) ([A-Z0-9]+)", body, re.M)

    for t in ("game_sim_distributions", "game_predictions_drive_sim"):
        assert cols(cfb, "cfb_season", t) == cols(nfl, "nfl_season", t) + [("division", "STRING")]


def test_predict_refuses_without_prior_season_drives(drives):
    with pytest.raises(RuntimeError):
        cfb_drive_sim.predict_week_drive_sim(
            2026, 3, played=played_games(drives), upcoming=upcoming(),
            drives=drives[drives.season == 2026], lines=LINES,
            now=pd.Timestamp("2026-09-20", tz="UTC"))


def test_write_is_game_scoped_and_create_never():
    import backfill_cfb

    dist = pd.DataFrame({"game_id": ["u2"]})
    with unittest.mock.patch.object(backfill_cfb, "replace_game_ids", return_value=1) as rep:
        out = cfb_drive_sim.write(dist, dist)
    assert out == {"game_sim_distributions": 1, "game_predictions_drive_sim": 1}
    for c in rep.call_args_list:
        assert c.args[1] == "cfb_season" and c.kwargs["create_disposition"] == "CREATE_NEVER"
        assert c.kwargs["partition_field"] == "game_date"
        assert c.kwargs["cluster_fields"] == ["season", "week"]


# ---------------------------------------------------------------- handler: flags + safety
def _load_main():
    spec = importlib.util.spec_from_file_location("cfb_main_drive_sim_test",
                                                  os.path.join(SRC, "cfb", "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Req:
    def __init__(self, body):
        self._body = body

    def get_json(self, silent=True):
        return self._body


def _clear_env(monkeypatch):
    for k in ("CFB_DRIVE_SIM_SHADOW", "CFB_RIDGE_SHADOW", "FPI_SNAPSHOT"):
        monkeypatch.delenv(k, raising=False)


def test_flag_from_body_or_env(monkeypatch):
    main = _load_main()
    _clear_env(monkeypatch)
    assert not main._drive_sim_enabled({})
    assert main._drive_sim_enabled({"shadow_drive_sim": True})
    monkeypatch.setenv("CFB_DRIVE_SIM_SHADOW", "1")
    assert main._drive_sim_enabled({})
    monkeypatch.setenv("CFB_DRIVE_SIM_SHADOW", "0")
    assert not main._drive_sim_enabled({})


def _frames():
    ids = ["u2"]
    return (pd.DataFrame({"game_id": ids, "season": 2026, "week": 3}),
            pd.DataFrame({"game_id": ids, "season": 2026, "week": 3}),
            {"fit_s": 0.1, "games": 1})


def _run(body, monkeypatch, sim_effect=None):
    import backfill_cfb
    import pipeline

    main = _load_main()
    _clear_env(monkeypatch)
    rows = pd.DataFrame({"game_id": ["u2"]})
    sim_kw = {"side_effect": sim_effect} if sim_effect else {"return_value": _frames()}
    with unittest.mock.patch.object(pipeline, "predict_week", return_value=rows), \
         unittest.mock.patch.object(cfb_drive_sim, "predict_week_drive_sim", **sim_kw) as sim, \
         unittest.mock.patch.object(backfill_cfb, "replace_game_ids", return_value=1) as rep, \
         unittest.mock.patch.object(backfill_cfb, "ensure_datasets"), \
         unittest.mock.patch.object(backfill_cfb, "load", return_value=1) as ld:
        out, status = main.cfb_pipeline(_Req(body))
    return out, status, sim, rep, ld


def test_dry_run_computes_but_writes_nothing(monkeypatch):
    out, status, sim, rep, ld = _run({"mode": "predict_week", "season": 2026, "week": 3,
                                      "dry_run": True, "shadow_drive_sim": True}, monkeypatch)
    assert status == 200 and out["dry_run"] is True
    sim.assert_called_once()
    rep.assert_not_called()
    ld.assert_not_called()
    assert out["steps"]["shadow_drive_sim"]["written"] == 0


def test_shadow_off_by_default(monkeypatch):
    out, status, sim, rep, _ = _run({"mode": "predict_week", "season": 2026, "week": 3},
                                    monkeypatch)
    assert status == 200
    sim.assert_not_called()
    assert [c.args[2] for c in rep.call_args_list] == ["game_predictions"]


def test_shadow_writes_both_tables_create_never(monkeypatch):
    out, status, sim, rep, _ = _run({"mode": "predict_week", "season": 2026, "week": 3,
                                     "shadow_drive_sim": True}, monkeypatch)
    assert status == 200
    calls = {c.args[2]: c for c in rep.call_args_list}
    assert set(calls) == {"game_predictions", "game_sim_distributions",
                          "game_predictions_drive_sim"}
    for t in ("game_sim_distributions", "game_predictions_drive_sim"):
        assert calls[t].kwargs["create_disposition"] == "CREATE_NEVER"
        assert calls[t].args[0]["game_id"].tolist() == ["u2"]


def test_shadow_failure_is_not_fatal(monkeypatch):
    out, status, *_ = _run({"mode": "predict_week", "season": 2026, "week": 3,
                            "shadow_drive_sim": True}, monkeypatch,
                           sim_effect=RuntimeError("no 2025 drives"))
    assert status == 200 and "error" in out["steps"]["shadow_drive_sim"]
    assert out["steps"]["predicted"] == 1


def test_dry_run_still_refused_for_other_modes(monkeypatch):
    main = _load_main()
    for mode in ("ingest", "score", "rankings", "cfbd", "backfill", "fpi_snapshot"):
        out, status = main.cfb_pipeline(_Req({"mode": mode, "dry_run": True}))
        assert status == 400, mode


def test_drives_mode_dry_run_fetches_but_writes_nothing(monkeypatch):
    import backfill_cfb
    import pipeline
    from stats import cfbd

    main = _load_main()
    monkeypatch.setenv("CFBD_API_KEY", "test-not-a-key")
    games = pd.DataFrame({"game_id": ["9"], "season": [2026], "week": [4], "is_postseason": [0],
                          "home_team": ["ARMY"], "away_team": ["NAVY"], "division": ["fbs"],
                          "home_won": [1]})
    recs = [_rec(9, 1, "Army", "Navy", True, "TD", s1=(7, 0))]
    with unittest.mock.patch.object(pipeline, "load_played_games", return_value=games), \
         unittest.mock.patch.object(cfb_drives, "load_drives",
                                    return_value=pd.DataFrame(columns=cfb_drives.DRIVE_COLUMNS)), \
         unittest.mock.patch.object(cfb_drives, "fetch_week", return_value=recs) as fw, \
         unittest.mock.patch.object(backfill_cfb, "replace_game_ids") as rep, \
         unittest.mock.patch.object(cfbd, "has_api_key", return_value=True):
        out, status = main.cfb_pipeline(_Req({"mode": "drives", "season": 2026, "dry_run": True}))
    assert status == 200 and out["dry_run"] is True
    fw.assert_called_once()
    rep.assert_not_called()
    assert out["steps"]["drives"]["drives"] == 1 and out["steps"]["drives"]["written"] == 0


def test_drives_ingest_writes_game_scoped_create_never(monkeypatch):
    import backfill_cfb
    import pipeline

    games = pd.DataFrame({"game_id": ["9"], "season": [2026], "week": [4], "is_postseason": [0],
                          "home_team": ["ARMY"], "away_team": ["NAVY"], "division": ["fbs"],
                          "home_won": [1]})
    recs = [_rec(9, 1, "Army", "Navy", True, "TD", s1=(7, 0))]
    with unittest.mock.patch.object(pipeline, "load_played_games", return_value=games), \
         unittest.mock.patch.object(cfb_drives, "load_drives",
                                    return_value=pd.DataFrame(columns=cfb_drives.DRIVE_COLUMNS)), \
         unittest.mock.patch.object(cfb_drives, "fetch_week", return_value=recs), \
         unittest.mock.patch.object(backfill_cfb, "ensure_datasets"), \
         unittest.mock.patch.object(backfill_cfb, "replace_game_ids", return_value=1) as rep:
        info = cfb_drives.ingest(2026)
    assert info["weeks"] == [4] and info["written"] == 1
    c = rep.call_args
    assert c.args[1:3] == ("cfb_historical", "drives")
    assert c.kwargs["create_disposition"] == "CREATE_NEVER"


# ---------------------------------------------------------------- research reproduction
RESEARCH = os.environ.get("CFB_DRIVE_SIM_RESEARCH_DIR")


@pytest.mark.skipif(not RESEARCH, reason="set CFB_DRIVE_SIM_RESEARCH_DIR to the research run dir")
def test_port_reproduces_research_week():
    from espn_data import resolve_team_divisions

    g = pd.read_parquet(os.path.join(RESEARCH, "cfb_games_bq.parquet"))
    g = g[g.season <= 2025]
    fbs = {k: int(v == "fbs") for k, v in resolve_team_divisions(g).items()}
    teams = pd.Index(sorted(set(g.home_team) | set(g.away_team)))
    ref = pd.read_parquet(os.path.join(
        RESEARCH, "runs", "sim_c_C0.1_t16_pw1_N4000_s0_2022_2025.parquet")).set_index("game_id")
    d = ds.prep_drives(pd.read_parquet(os.path.join(RESEARCH, "cfb_drives.parquet")), g, ds.CFB)
    m = ds.fit_week(d, g, 2025, 10, teams=teams, cfg=ds.CFB, fbs=fbs)
    G = g[(g.season == 2025) & (g.week == 10) & g.home_won.notna()]
    for r in G.itertuples():
        hs, aw, _ = ds.simulate(m, r.home_team, r.away_team, bool(r.neutral_site), 2025, "REG",
                                N=4000, rng=ds.game_rng(r.game_id))
        h = np.bincount(np.clip(hs - aw + 100, 0, 200).astype(int), minlength=201)
        assert np.array_equal(h, np.asarray(ref.loc[r.game_id].mh))
