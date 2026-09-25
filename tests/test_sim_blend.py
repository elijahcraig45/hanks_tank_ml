import json
import os
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pa_sim import blend, strength, v2_inputs


# ------------------------------------------------------------------ strength
def _season(year, n_days=40, seed=0, strong="A"):
    rng = np.random.default_rng(seed + year)
    teams = ["A", "B", "C", "D"]
    rows, pk = [], year * 10000
    for d in range(n_days):
        for h, a in (("A", "B"), ("C", "D")) if d % 2 else (("B", "C"), ("D", "A")):
            ph = 0.54 + (0.15 if h == strong else 0) - (0.15 if a == strong else 0)
            hw = rng.random() < ph
            rows.append(dict(game_pk=pk, game_date=pd.Timestamp(f"{year}-04-01") + pd.Timedelta(days=d),
                             year=year, home=h, away=a, h_runs=5 if hw else 3, a_runs=3 if hw else 5))
            pk += 1
    return pd.DataFrame(rows)


def test_elo_and_pythag_use_only_prior_games():
    g = _season(2024)
    G = strength.features(g)
    first = G.iloc[0]
    assert first.elo_d == pytest.approx(24.0)       # everyone at 1500, plus home points
    assert first.pyth_d == pytest.approx(0.0)
    # changing a game's own result must not change its own features
    g2 = g.copy(); g2.loc[5, ["h_runs", "a_runs"]] = [0, 10]
    G2 = strength.features(g2)
    assert G2.loc[G2.game_pk == g.game_pk[5], "elo_d"].item() == pytest.approx(
        G.loc[G.game_pk == g.game_pk[5], "elo_d"].item())


def test_strong_team_rises_and_season_regresses_one_third():
    g = pd.concat([_season(2023), _season(2024)])
    G = strength.features(g)
    a_home = G[(G.home == "A") & (G.year == 2023)]
    assert a_home.elo_d.iloc[-1] > a_home.elo_d.iloc[0]
    # an unplayed game has no label
    g3 = g.copy(); g3.loc[g3.index[-1], ["h_runs", "a_runs"]] = np.nan
    assert np.isnan(strength.features(g3).y.iloc[-1])


def test_strength_fit_needs_three_prior_seasons_of_games():
    g = pd.concat([_season(y, n_days=300) for y in (2022, 2023, 2024)] + [_season(2025, 5)])
    G = strength.features(g)
    m = strength.fit(G, 2025)
    assert m is not None
    p = strength.predict(m, G[G.year == 2025])
    assert ((p > 0) & (p < 1)).all()


def test_strength_for_slate_ignores_same_day_results():
    g = pd.concat([_season(y, n_days=300) for y in (2022, 2023, 2024)] + [_season(2025, 10)])
    target = date(2025, 4, 10)
    slate = g[g.game_date == pd.Timestamp(target)][["game_pk"]]
    a = blend.strength_for_slate(g, slate, target)
    g2 = g.copy(); g2.loc[g2.game_date == pd.Timestamp(target), ["h_runs", "a_runs"]] = [0, 20]
    b = blend.strength_for_slate(g2, slate, target)
    assert np.allclose(a, b)


def test_strength_for_slate_scores_games_not_yet_in_the_games_table():
    """Tonight's games have no games-table row until they are played."""
    ids = {"A": "101", "B": "102", "C": "103", "D": "104"}
    g = pd.concat([_season(y, n_days=300) for y in (2022, 2023, 2024)] + [_season(2025, 10)])
    g["home"] = g.home.map(ids); g["away"] = g.away.map(ids)
    target = date(2025, 4, 11)
    slate = pd.DataFrame({"game_pk": [999001], "game_date": [pd.Timestamp(target)],
                          "home_team_id": [101], "away_team_id": [102]})
    p = blend.strength_for_slate(g, slate, target)
    assert len(p) == 1 and 0 < p[0] < 1


# ------------------------------------------------------------------ frozen blend math
def test_frozen_coefficients_load_with_provenance():
    c = blend.load_coefs()
    assert c["model_version"] == "sim_blend_v2"
    assert c["provenance"]["n_games"] > 20000
    assert c["sim_config"]["xw"] == 0.5 and c["sim_config"]["weather"] is False


def test_blend_is_the_stacked_logistic():
    c = blend.load_coefs()
    s = c["stack"]
    p = blend.blend(0.6, 0.55, c)
    z = s["intercept"] + s["coef_logit_strength"] * np.log(0.6 / 0.4) + s["coef_logit_sim"] * np.log(0.55 / 0.45)
    assert p == pytest.approx(1 / (1 + np.exp(-z)))
    # monotone in both inputs
    assert blend.blend(0.65, 0.55, c) > p and blend.blend(0.6, 0.6, c) > p
    assert 0.45 < blend.calibrate_sim(0.5, c) < 0.5    # slight away tilt from the intercept


def test_tilt_hits_target_mean_and_keeps_a_pmf():
    from scipy.stats import poisson
    k = np.arange(31); pmf = poisson.pmf(k, 9.5); pmf /= pmf.sum()
    q = blend.tilt(pmf, 9.0)
    assert q.sum() == pytest.approx(1.0) and (q * k).sum() == pytest.approx(9.0, abs=1e-6)


# ------------------------------------------------------------------ rows
def _summary(G=2, seed=0):
    rng = np.random.default_rng(seed)
    def pmf(*shape):
        x = rng.random(shape); return x / x.sum(-1, keepdims=True)
    return {"p_home": np.array([0.55, 0.40][:G]), "tot_hist": pmf(G, 31), "h_hist": pmf(G, 31),
            "a_hist": pmf(G, 31), "spk": pmf(G, 2, 21), "bh": pmf(G, 2, 9, 6),
            "bhr": pmf(G, 2, 9, 4), "bk": pmf(G, 2, 9, 6)}


def _slate(G=2):
    return pd.DataFrame([dict(game_pk=100 + i, game_date=date(2026, 9, 20),
                              game_time_utc=pd.Timestamp("2026-09-20T23:05Z"),
                              home_team_id=1, away_team_id=2, home_team_name="H", away_team_name="A",
                              home_starter_id=11, away_starter_id=22, home_starter_name="Hs",
                              away_starter_name="As", home_lineup=list(range(1, 10)),
                              away_lineup=list(range(21, 30))) for i in range(G)])


def test_rows_match_the_table_contracts():
    c = blend.load_coefs()
    now = datetime(2026, 9, 20, 18, tzinfo=timezone.utc)
    sm = _summary()
    pred, props = blend.game_rows(sm, _slate(), np.array([0.6, 0.45]), c, 3000, now)
    want_pred = {"game_pk", "game_date", "game_time_utc", "home_team_id", "away_team_id",
                 "home_team_name", "away_team_name", "home_starter_id", "away_starter_id",
                 "home_win_probability", "away_win_probability", "sim_p_raw", "sim_p_cal",
                 "strength_p", "predicted_winner", "confidence_tier", "model_version",
                 "n_episodes", "mean_home_runs", "mean_away_runs", "predicted_at"}
    assert set(pred[0]) == want_pred
    want_props = {"game_pk", "game_date", "game_time_utc", "home_team_name", "away_team_name",
                  "model_version", "n_episodes", "mean_home_runs", "mean_away_runs",
                  "mean_total_runs", "total_runs_pmf", "totals_calibrated", "total_bias_shift",
                  "market_total_line", "p_over_market", "home_starter_id", "home_starter_name",
                  "home_starter_k_mean", "home_starter_k_pmf", "away_starter_id",
                  "away_starter_name", "away_starter_k_mean", "away_starter_k_pmf",
                  "batter_props_json", "predicted_at"}
    assert set(props[0]) == want_props
    r = pred[0]
    assert r["home_win_probability"] == pytest.approx(float(blend.blend(0.6, 0.55, c)))
    assert r["sim_p_raw"] == 0.55 and r["strength_p"] == 0.6
    p0 = props[0]
    raw = (sm["tot_hist"][0] * np.arange(31)).sum()
    assert p0["mean_total_runs"] == pytest.approx(raw - c["totals_bias_runs"], abs=1e-6)
    assert p0["total_bias_shift"] == pytest.approx(-c["totals_bias_runs"])
    # spk[:, 1] is the HOME starter (indexed by defensive side)
    assert p0["home_starter_k_pmf"] == pytest.approx(list(sm["spk"][0, 1]))
    assert p0["market_total_line"] is None and p0["batter_props_json"] is None


def test_batter_props_only_when_experimental():
    c = blend.load_coefs()
    now = datetime(2026, 9, 20, 18, tzinfo=timezone.utc)
    _, props = blend.game_rows(_summary(), _slate(), np.array([0.6, 0.45]), c, 3000, now,
                               experimental=True)
    bp = json.loads(props[0]["batter_props_json"])
    assert bp["calibrated"] is False and len(bp["home"]) == 9
    assert bp["home"][0]["player_id"] == 1


# ------------------------------------------------------------------ memory guard
def test_memory_guard(monkeypatch):
    monkeypatch.setenv("SIM_BLEND_MEMORY_MB", "1024")
    ok, have, need = blend.memory_ok()
    assert not ok and have == 1024 and need == 3072
    monkeypatch.setenv("SIM_BLEND_MEMORY_MB", "4096")
    assert blend.memory_ok()[0]
    monkeypatch.setenv("SIM_BLEND_MEMORY_MB", "1024")
    assert blend.run_slate(date(2026, 9, 20))["status"] == "insufficient_memory"


def test_memory_unknown_inside_managed_runtime_is_refused(monkeypatch):
    monkeypatch.delenv("SIM_BLEND_MEMORY_MB", raising=False)
    monkeypatch.setattr(blend, "container_memory_mb", lambda: None)
    monkeypatch.setenv("K_SERVICE", "mlb-2026-daily-pipeline")
    assert not blend.memory_ok()[0]
    monkeypatch.delenv("K_SERVICE")
    assert blend.memory_ok()[0]


# ------------------------------------------------------------------ inputs port
def test_classify_maps_events_to_nine_classes():
    df = pd.DataFrame({"events": ["strikeout", "walk", "single", "double", "triple", "home_run",
                                  "field_error", "grounded_into_double_play", "sac_fly",
                                  "field_out", "field_out", "caught_stealing_2b"]})
    air = np.array([0] * 9 + [1, 0, 0], bool)
    assert list(v2_inputs.classify(df, air)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, -1]


def test_assemble_slate_needs_nine_and_starters():
    rows = []
    for side, n in (("home", 9), ("away", 8)):
        for j in range(n):
            rows.append(dict(game_pk=5, game_date=date(2026, 9, 20), team_type=side,
                             batting_order=j + 1, player_id=100 + j, home_starter_id=1,
                             away_starter_id=2, home_starter_name="h", away_starter_name="a",
                             game_time_utc=pd.Timestamp("2026-09-20T23:00Z"), home_team_id=1,
                             away_team_id=2, home_team_name="H", away_team_name="A",
                             home_park_factor=1.0))
    assert blend.assemble_slate(pd.DataFrame(rows), {5: 15}, {}).empty
