"""Distribution summaries, player projections and write scoping for the sim_blend writer."""
import json
import os
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pa_sim import blend, dists, v2


# ------------------------------------------------------------------ Dist
def test_dist_from_pmf_integer_percentiles():
    pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])            # support 0..4
    d = dists.dist_from_pmf(pmf, n=3000)
    assert d["mean"] == pytest.approx(2.0)
    assert d["sd"] == pytest.approx(np.sqrt(1.2))
    assert (d["p05"], d["p25"], d["p50"], d["p75"], d["p95"]) == (0, 1, 2, 3, 4)
    assert (d["min"], d["max"], d["n"]) == (0, 4, 3000)
    assert all(isinstance(d[k], int) for k in dists.QNAMES + ("min", "max"))


def test_dist_matches_numpy_inverted_cdf_on_samples():
    x = np.random.default_rng(3).poisson(4.6, 3000)
    d = dists.dist_from_samples(x)
    for name, q in zip(dists.QNAMES, dists.QS):
        assert d[name] == int(np.percentile(x, 100 * q, method="inverted_cdf"))
    assert d["mean"] == pytest.approx(x.mean()) and d["sd"] == pytest.approx(x.std())
    assert d["n"] == 3000 and d["min"] == x.min() and d["max"] == x.max()


def test_weighted_samples_equal_replicated_samples():
    x = np.array([-3, 0, 1, 1, 5])
    w = np.array([1, 2, 1, 1, 3], float)
    a = dists.dist_from_samples(x, w, n=5)
    b = dists.dist_from_samples(np.repeat(x, w.astype(int)), n=5)
    assert {k: a[k] for k in dists.DIST_FIELDS} == pytest.approx({k: b[k] for k in dists.DIST_FIELDS})


def test_thin_scales_the_mean_and_stays_a_pmf():
    pmf = np.array([0.3, 0.4, 0.2, 0.1])
    q = dists.thin(pmf, 0.9)
    k = np.arange(4)
    assert q.sum() == pytest.approx(1.0) and (q >= 0).all()
    assert (q * k).sum() == pytest.approx(0.9 * (pmf * k).sum())
    assert q[0] > pmf[0]
    assert np.allclose(dists.thin(pmf, 1.0), pmf)


def test_pmfs_counts_per_game():
    arr = np.array([[0, 1], [2, 1], [1, 1], [1, 0]])       # lanes x 2 sides
    gi = np.array([0, 0, 1, 1])
    P = dists.pmfs(arr, gi, 2, 3)
    assert P.shape == (2, 2, 4)
    assert np.allclose(P[0, 0], [0.5, 0, 0.5, 0]) and np.allclose(P[1, 1], [0.5, 0.5, 0, 0])


# ------------------------------------------------------------------ game distribution row
def _episodes(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.poisson(4.6, n); a = rng.poisson(4.4, n)
    tie = h == a
    h[tie] += rng.integers(0, 2, tie.sum()); a[tie & (h == a)] += 1
    inn = np.where(rng.random(n) < 0.09, 10, 9)
    return h, a, inn, np.full(n, 9)


def test_game_row_applies_the_totals_tilt_consistently():
    h, a, inn, sched = _episodes()
    row = dists.game_distribution_row(h, a, inn, sched, bias_runs=0.47)
    raw = (h + a).mean()
    assert row["total_mean"] == pytest.approx(raw - 0.47, abs=1e-6)
    assert row["home_runs_mean"] + row["away_runs_mean"] == pytest.approx(row["total_mean"])
    assert row["home_runs_mean"] - row["away_runs_mean"] == pytest.approx(row["margin_mean"])
    assert row["totals_calibrated"] is True and row["total_bias_shift"] == pytest.approx(-0.47)
    over = json.loads(row["p_over_by_line"])
    assert list(over) == ["6.5", "7.5", "8.5", "9.5", "10.5", "11.5"]
    assert all(np.diff(list(over.values())) <= 0)
    assert 0 < row["p_home_cover_rl"] < row["p_home_win"] < 1
    assert row["p_extra_innings"] == pytest.approx(0.09, abs=0.03)


def test_game_row_without_tilt_is_the_raw_episodes():
    h, a, inn, sched = _episodes()
    row = dists.game_distribution_row(h, a, inn, sched, bias_runs=0.47, apply_bias=False)
    assert row["total_mean"] == pytest.approx((h + a).mean())
    assert row["p_home_win"] == pytest.approx((h > a).mean())
    assert row["totals_calibrated"] is False and row["total_bias_shift"] == 0.0


# ------------------------------------------------------------------ simulator accumulators
class Stub:
    cfg = v2.Config(hook="curve")
    curve = np.array([1, 1, 1, .97, .9, .75, .5, .25, .1])
    trans = v2.fixed_transitions()

    def tables(self, g):
        L = np.array([.22, .09, .15, .045, .004, .03, .007, .23, .224]); L = L / L.sum()
        return np.tile(L, (2, 9, 4, 1))


def _sim(exposure=None, n=1500, G=2):
    g = v2.GameSpec([0] * 9, [0] * 9, 1, 2, "H", "A", 0, 9, True)
    return v2.simulate(Stub(), [g] * G, n=n, seed=4, exposure=exposure)


def test_exposure_never_changes_the_game_itself():
    S = np.full((9, 8), 0.8); S[:, 0] = 1.0
    a, b = _sim(), _sim(S)
    for k in ("home", "away", "innings", "bpa", "bh", "sp_k", "sp_r", "sp_bf"):
        assert np.array_equal(a[k], b[k]), k
    assert "bpa_x" not in a
    assert (b["bpa_x"] <= b["bpa"]).all() and (b["bh_x"] <= b["bh"]).all()
    assert (b["btb_x"] <= b["btb"]).all() and b["bpa_x"].mean() < b["bpa"].mean()


def test_exposure_of_one_credits_every_turn():
    b = _sim(np.ones((9, 8)))
    for k in ("bpa", "bh", "bhr", "bk", "btb", "bbb"):
        assert np.array_equal(b[k], b[k + "_x"])


def test_new_accumulators_are_consistent():
    r = _sim()
    assert (r["btb"] >= r["bh"]).all() and (r["btb"] <= 4 * r["bh"]).all()
    assert (r["bhr"] <= r["bh"]).all()
    # a starter faces at least 3 batters (curve hook), allows no more than the batters faced
    assert (r["sp_bf"] >= 3).all() and ((r["sp_h"] + r["sp_bb"]) <= r["sp_bf"]).all()
    # runs charged to the away starter (defensive side 0) cannot exceed the home team's runs
    assert (r["sp_r"][:, 0] <= r["home"]).all() and (r["sp_r"][:, 1] <= r["away"]).all()
    # the away side bats 27+ times (3 per slot); home may skip the 9th (24+, 2 per slot)
    assert r["bpa"][:, 0].min() >= 3 and r["bpa"][:, 1].min() >= 2
    assert r["innings"].min() >= 9


# ------------------------------------------------------------------ rows
def _slate(G=2, start="2026-09-20T23:05Z"):
    return pd.DataFrame([dict(game_pk=100 + i, game_date=date(2026, 9, 20),
                              game_time_utc=pd.Timestamp(start), home_team_id=1, away_team_id=2,
                              home_team_name="H", away_team_name="A", home_starter_id=11,
                              away_starter_id=22, home_starter_name="Hs", away_starter_name="As",
                              home_lineup=list(range(1, 10)), away_lineup=list(range(21, 30)),
                              home_lineup_names=[f"h{j}" for j in range(9)],
                              away_lineup_names=[f"a{j}" for j in range(9)],
                              home_ab="H", away_ab="A", venue_id=0) for i in range(G)])


CAL = {"exposure": {"S": np.full((9, 8), 0.9).tolist()},
       "batter": {"H": {"calibrated": True, "apply_thin": True, "thin": 0.95, "note": "ok"}},
       "starter": {"K": {"calibrated": True, "apply_thin": False, "thin": 1.0, "note": "sp ok"}}}

PLAYER_COLS = {"game_pk", "game_date", "predicted_at", "model_version", "player_id", "player_name",
               "team_id", "role", "batting_order", "stat", "mean", "sd", "p05", "p25", "p50", "p75",
               "p95", "min", "max", "n_sims", "p_at_least_1", "calibrated", "calibration_note"}


def test_player_rows_shape_and_calibration_flags():
    c = blend.load_coefs()
    res = _sim(np.array(CAL["exposure"]["S"]))
    now = datetime(2026, 9, 20, 18, tzinfo=timezone.utc)
    rows = blend.player_rows(res, _slate(), c, 1500, now, CAL)
    assert len(rows) == 2 * (18 * 6 + 2 * 6)
    assert set(rows[0]) == PLAYER_COLS
    df = pd.DataFrame(rows)
    assert set(df[df.role == "batter"].stat) == {"PA", "H", "HR", "TB", "BB", "K"}
    assert set(df[df.role == "starter"].stat) == {"K", "BF", "IP_outs", "ER", "H_allowed", "BB_allowed"}
    assert not {"R", "RBI"} & set(df.stat)
    assert df[df.role == "starter"].batting_order.isna().all()
    assert df[df.role == "starter"].p_at_least_1.isna().all()
    assert df[(df.role == "batter")].calibrated.sum() == 36          # H only, 18 batters x 2 games
    h = df[(df.role == "batter") & (df.stat == "H") & (df.game_pk == 100) & (df.batting_order == 1) & (df.team_id == 1)].iloc[0]
    raw = res["bh_x"][:1500, 1, 0]
    pm = np.bincount(raw, minlength=9) / 1500.0
    assert h["mean"] == pytest.approx(0.95 * (pm * np.arange(len(pm))).sum())
    assert h.player_id == 1 and h.player_name == "h0"
    sp = df[(df.role == "starter") & (df.stat == "K") & (df.player_id == 22) & (df.game_pk == 100)].iloc[0]
    assert sp.team_id == 2 and sp["mean"] == pytest.approx(res["sp_k"][:1500, 0].mean())
    assert bool(sp.calibrated) and sp.calibration_note == "sp ok"


def test_player_rows_without_calibration_are_flagged_uncalibrated():
    res = _sim()
    rows = blend.player_rows(res, _slate(1), blend.load_coefs(), 1500,
                             datetime(2026, 9, 20, tzinfo=timezone.utc), None)
    assert not any(r["calibrated"] for r in rows)


def test_distribution_rows_match_the_contract():
    res = _sim()
    c = blend.load_coefs()
    rows = blend.distribution_rows(res, _slate(), c, 1500, datetime(2026, 9, 20, tzinfo=timezone.utc))
    want = {"game_pk", "game_date", "game_time_utc", "predicted_at", "model_version", "n_sims",
            "p_home_win", "p_extra_innings", "p_home_cover_rl", "p_over_by_line",
            "totals_calibrated", "total_bias_shift"}
    for pre in ("home_runs", "away_runs", "total", "margin"):
        want |= {f"{pre}_{k}" for k in dists.DIST_FIELDS}
    assert set(rows[0]) == want and len(rows) == 2
    raw = (res["home"][:1500] + res["away"][:1500]).mean()
    assert rows[0]["total_mean"] == pytest.approx(raw - c["totals_bias_runs"], abs=1e-6)
    assert rows[0]["model_version"] == c["model_version"]


def test_frame_keeps_nullable_ints():
    df = blend.frame([{"batting_order": 1, "p": 0.5}, {"batting_order": None, "p": None}])
    assert str(df.batting_order.dtype) == "Int64"


def test_shipped_player_calibration_is_well_formed():
    cal = blend.load_player_calibration()
    assert cal is not None
    S = np.array(cal["exposure"]["S"])
    assert S.shape[0] == 9 and (np.diff(S, axis=1) <= 1e-9).all() and (S <= 1).all()
    for role, stats in (("batter", dists.BATTER_STATS), ("starter", dists.STARTER_STATS)):
        for s in stats:
            e = cal[role][s]
            assert isinstance(e["calibrated"], bool) and e["note"]
            assert 0 < e.get("thin", 1.0) <= 1.0


# ------------------------------------------------------------------ run_slate: flags and write scoping
class FakeBQ:
    def query(self, *a, **k):
        class J:
            def to_dataframe(self_inner):
                return pd.DataFrame()
        return J()


def _patch_run(monkeypatch, slate):
    import pa_sim.v2_inputs as v2i
    import pa_sim.strength as strength
    monkeypatch.setenv("SIM_BLEND_MEMORY_MB", "4096")
    monkeypatch.setattr(blend, "_slate", lambda bq, t: slate.copy())
    monkeypatch.setattr(blend, "strength_for_slate", lambda g, s, t: np.full(len(s), 0.55))
    monkeypatch.setattr(v2i, "load_inputs", lambda bq, t: {})
    monkeypatch.setattr(blend, "build_engine", lambda inputs, t, c: Stub())
    monkeypatch.setattr(strength, "GAMES_SQL", "{proj}{hist}{ds}")
    calls = []
    monkeypatch.setattr(blend, "append", lambda bq, table, rows: calls.append((table, rows)))
    return calls


def test_dry_run_writes_nothing(monkeypatch):
    calls = _patch_run(monkeypatch, _slate(2))
    out = blend.run_slate(date(2026, 9, 20), dry_run=True, bq=FakeBQ(), n_episodes=300,
                          now=datetime(2026, 9, 20, 18, tzinfo=timezone.utc))
    assert out["status"] == "dry_run" and calls == []
    assert out["distribution_rows"] == 2 and out["player_rows"] == 240


def test_writes_only_pregame_games_of_the_request(monkeypatch):
    s = _slate(3)
    s.loc[2, "game_time_utc"] = pd.Timestamp("2026-09-20T17:00Z")      # already started
    calls = _patch_run(monkeypatch, s)
    out = blend.run_slate(date(2026, 9, 20), bq=FakeBQ(), n_episodes=300, game_pks=[100, 102],
                          now=datetime(2026, 9, 20, 18, tzinfo=timezone.utc))
    assert out["status"] == "ok" and out["skipped_started"] == 1
    tables = [t for t, _ in calls]
    assert tables == [blend.PRED_TABLE, blend.PROPS_TABLE, blend.DIST_TABLE, blend.PLAYER_TABLE]
    for _, rows in calls:
        assert {r["game_pk"] for r in rows} == {100}
    assert out["writes"] == {blend.DIST_TABLE: "ok", blend.PLAYER_TABLE: "ok"}


def test_missing_new_table_does_not_cost_the_existing_rows(monkeypatch):
    calls = _patch_run(monkeypatch, _slate(1))

    def append(bq, table, rows):
        if table == blend.PLAYER_TABLE:
            raise RuntimeError("404 Not found: Table hankstank:mlb_2026_season.player_sim_projections")
        calls.append(table)
    monkeypatch.setattr(blend, "append", append)
    out = blend.run_slate(date(2026, 9, 20), bq=FakeBQ(), n_episodes=300,
                          now=datetime(2026, 9, 20, 18, tzinfo=timezone.utc))
    assert out["status"] == "ok" and calls == [blend.PRED_TABLE, blend.PROPS_TABLE, blend.DIST_TABLE]
    assert out["writes"][blend.PLAYER_TABLE] == "table_missing"
