"""Tests for the margin ridge and the football paths wired to it.

What is pinned: the closed-form solve is the textbook weighted ridge; the fit recovers
known ratings, home field and covariate effects; walk-forward never sees the week it
predicts or anything after it; and the two production bugs found alongside it stay
fixed — NFL predictions carrying NULL EPA, and CFB predictions built without the
current season. Nothing here touches BigQuery, ESPN or nflverse.
"""

import math
import os
import sys
import unittest.mock

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src", "cfb"))
sys.path.insert(0, os.path.join(HERE, "..", "src", "nfl"))

import margin_ridge as mr  # noqa: E402


# ----------------------------------------------------------------- synthetic data
def league(n_teams=12, seasons=(2023, 2024), weeks=14, hfa=2.5, noise=0.0,
           seed=0, strength=None, cov=None):
    """Round-robin-ish schedule with margins drawn from the ridge's own model."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    strength = strength or {t: (i - n_teams / 2) * 1.5 for i, t in enumerate(teams)}
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            order = rng.permutation(teams)
            for k in range(0, n_teams, 2):
                h, a = order[k], order[k + 1]
                neutral = int(rng.random() < 0.1)
                x = float(rng.normal()) if cov else 0.0
                m = strength[h] - strength[a] + hfa * (1 - neutral) + (cov or 0) * x
                m += noise * rng.normal()
                rows.append({"game_id": f"{s}-{w}-{h}", "season": s, "week": w,
                             "home_team": h, "away_team": a, "neutral": neutral,
                             "margin": m, "x": x})
    return pd.DataFrame(rows), strength


# ----------------------------------------------------------------- core maths
def test_solve_matches_sklearn_weighted_ridge():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(1)
    X, y, w = rng.normal(size=(80, 6)), rng.normal(size=80), rng.random(80)
    ours = mr._solve(X, y, w, alpha=2.5)
    ref = Ridge(alpha=2.5, fit_intercept=False).fit(X, y, sample_weight=w).coef_
    np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-10)


def test_norm_cdf_and_win_prob():
    assert mr.norm_cdf(0.0) == pytest.approx(0.5)
    assert mr.norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    p = mr.win_prob(np.array([-7.0, 7.0]), 13.0)
    assert p[0] + p[1] == pytest.approx(1.0)
    assert p[1] == pytest.approx(0.5 * (1 + math.erf(7 / 13 / math.sqrt(2))))


def test_fit_recovers_ratings_and_home_field():
    g, strength = league(noise=0.0)
    cfg = mr.RidgeConfig(alpha=0.01, tau=1e6, sigma=13.0)
    m = mr.fit(g, mr.time_index(2025, 1), cfg)
    assert m.hfa == pytest.approx(2.5, abs=0.05)
    # Ratings are identified only up to a constant: compare centred.
    got = pd.Series(m.ratings)
    want = pd.Series(strength)[got.index]
    np.testing.assert_allclose(got - got.mean(), want - want.mean(), atol=0.05)


def test_covariate_coefficient_is_points_per_unit():
    g, _ = league(noise=0.5, cov=3.0, seed=3)
    cfg = mr.RidgeConfig(alpha=0.1, tau=1e6, sigma=13.0, covariates=(("x", 2.0),))
    m = mr.fit(g, mr.time_index(2025, 1), cfg)
    # beta is on the scaled column, so points per unit of x = beta * scale.
    assert m.betas["x"] * 2.0 == pytest.approx(3.0, abs=0.15)
    one = g.iloc[[0]]
    base = m.predict_margin(one.assign(x=0.0))[0]
    assert m.predict_margin(one.assign(x=1.0))[0] - base == pytest.approx(
        m.betas["x"] * 2.0)


def test_unknown_team_rates_as_average():
    g, _ = league()
    m = mr.fit(g, mr.time_index(2025, 1), mr.RidgeConfig(alpha=1, tau=16, sigma=13))
    new = pd.DataFrame({"home_team": ["EXPANSION"], "away_team": ["EXPANSION2"],
                        "neutral": [1]})
    assert m.predict_margin(new)[0] == 0.0


def test_fit_returns_none_when_window_too_thin():
    g, _ = league(seasons=(2024,), weeks=2)
    assert mr.fit(g, mr.time_index(2024, 3),
                  mr.RidgeConfig(alpha=1, tau=16, sigma=13, min_train=50)) is None


def test_window_drops_games_older_than_window_seasons():
    g, _ = league(seasons=(2020, 2021, 2022, 2023))
    cfg = mr.RidgeConfig(alpha=1, tau=16, sigma=13, window_seasons=2)
    m = mr.fit(g, mr.time_index(2023, 1), cfg)
    assert m.n_train == int(g.season.isin([2021, 2022]).sum())


def test_fit_sigma_recovers_simulated_sigma():
    rng = np.random.default_rng(7)
    m = rng.normal(0, 10, 20000)
    won = (m + rng.normal(0, 14, len(m)) > 0).astype(int)
    assert mr.fit_sigma(m, won) == pytest.approx(14, rel=0.06)


# ----------------------------------------------------------------- causality
def test_walk_forward_ignores_current_and_future_weeks():
    g, _ = league(noise=5.0, seed=4)
    cfg = mr.RidgeConfig(alpha=1, tau=16, sigma=13)
    target = ((g.season == 2024) & (g.week == 6)).to_numpy()
    before = mr.walk_forward(g, target, cfg)

    poisoned = g.copy()
    later = (poisoned.season == 2024) & (poisoned.week >= 6)
    poisoned.loc[later, "margin"] = 500.0   # the week itself and everything after
    after = mr.walk_forward(poisoned, target, cfg)
    np.testing.assert_allclose(before[target], after[target])
    assert np.isnan(before[~target]).all()


def test_walk_forward_matches_fit_for_each_block():
    g, _ = league(noise=5.0, seed=5)
    cfg = mr.RidgeConfig(alpha=3, tau=10, sigma=13)
    target = (g.season == 2024).to_numpy()
    wf = mr.walk_forward(g, target, cfg)
    wk = g[(g.season == 2024) & (g.week == 9)]
    m = mr.fit(g, mr.time_index(2024, 9), cfg)
    np.testing.assert_allclose(wf[wk.index], m.predict_margin(wk), atol=1e-8)


def test_margin_cap_limits_blowouts():
    g, _ = league(noise=0.0)
    g.loc[0, "margin"] = 300.0
    cap = mr.RidgeConfig(alpha=1, tau=1e6, sigma=13, margin_cap=45.0)
    nocap = cap.with_(margin_cap=None)
    t = mr.time_index(2025, 1)
    h = g.loc[0, "home_team"]
    assert mr.fit(g, t, cap).ratings[h] < mr.fit(g, t, nocap).ratings[h]


# ----------------------------------------------------------------- CFB wiring
def cfb_games(seasons=(2024, 2025), weeks=10, upcoming_week=None, seed=0):
    """ESPN-shaped CFB games: 10 FBS teams and 6 FCS teams, some cross-division."""
    rng = np.random.default_rng(seed)
    fbs = [f"F{i}" for i in range(10)]
    fcs = [f"C{i}" for i in range(6)]
    power = {t: 10 - i for i, t in enumerate(fbs)} | {t: -15 - i for i, t in enumerate(fcs)}
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            pairs = [tuple(rng.permutation(fbs)[:2]) for _ in range(4)]
            pairs += [tuple(rng.permutation(fcs)[:2]) for _ in range(2)]
            pairs += [(fbs[w % 10], fcs[w % 6])]
            for k, (h, a) in enumerate(pairs):
                play = not (upcoming_week and s == seasons[-1] and w == upcoming_week)
                if upcoming_week and s == seasons[-1] and w > upcoming_week:
                    continue
                m = power[h] - power[a] + 3 + rng.normal(0, 8)
                m = m if abs(m) >= 1 else 1.0
                hs = 28 + m / 2
                rows.append({
                    "game_id": f"{s}{w:02d}{k}", "season": s, "week": w,
                    "division": "fbs" if h in fbs or a in fbs else "fcs",
                    "cross_division": int((h in fbs) != (a in fbs)),
                    "is_postseason": 0,
                    "game_date": pd.Timestamp(f"{s}-09-01") + pd.Timedelta(days=7 * w),
                    "home_team": h, "away_team": a,
                    "home_team_name": h + " U", "away_team_name": a + " U",
                    "home_score": round(hs) if play else None,
                    "away_score": round(hs - m) if play else None,
                    "home_won": int(m > 0) if play else None,
                    "result": m if play else None,
                    "neutral_site": 0, "conference_game": 0,
                })
    return pd.DataFrame(rows)


def test_cfb_ridge_frame_division_covariate():
    import pipeline as cp

    rg = cp.ridge_frame(cfb_games())
    cross = rg[rg.home_team.str.startswith("F") & rg.away_team.str.startswith("C")]
    same = rg[rg.home_team.str[0] == rg.away_team.str[0]]
    assert (cross.fbs_diff == 1).all() and (same.fbs_diff == 0).all()


def test_cfb_ridge_predict_week_is_shadow_shaped():
    import pipeline as cp

    g = cfb_games(upcoming_week=6)
    played, upcoming = g[g.home_won.notna()], g[g.home_won.isna()]
    rows = cp.predict_week(2025, 6, model="ridge", played=played, upcoming=upcoming,
                           now=pd.Timestamp("2000-01-01", tz="UTC"))

    assert len(rows) == len(upcoming)
    assert set(rows.model_version) == {cp.RIDGE_MODEL_VERSION}
    # additive columns, and NOT spread_line (the backend joins its own onto p.*)
    assert {"predicted_home_margin", "home_power_rating", "away_power_rating"} <= set(rows)
    assert "spread_line" not in rows
    np.testing.assert_allclose(
        rows.home_win_probability,
        mr.win_prob(rows.predicted_home_margin, mr.CFB_RIDGE.sigma))
    # the FBS side of a cross-division game should be the clear favourite
    cross = rows[rows.cross_division == 1]
    assert (cross.predicted_home_margin > 10).all()


def test_cfb_xgb_rows_keep_their_schema():
    import pipeline as cp

    g = cfb_games(upcoming_week=6)
    feats = cp.build(g)
    test = feats[feats.home_won.notna()].head(3)
    rows = cp._prediction_rows(test, np.full(3, 0.6), "fbs", cp.MODEL_VERSION)
    assert "predicted_home_margin" not in rows and "home_power_rating" not in rows
    assert list(rows.columns)[:10] == [
        "game_id", "season", "week", "division", "game_date", "home_team_id",
        "away_team_id", "home_team_name", "away_team_name", "home_win_probability"]


def test_cfb_ridge_backfill_is_out_of_sample():
    import pipeline as cp

    g = cfb_games()
    feats = cp.build(g)
    rows = cp.backfill_division(feats, "fbs", 2025, model="ridge", games=g)
    assert len(rows) and (rows.season == 2025).all() and (rows.division == "fbs").all()
    assert rows.prediction_correct.isin([0, 1]).all()


def test_cfb_played_games_prefer_bigquery_over_cold_cache():
    """On a cold container the parquet cache is empty and fetch_history stops at 2025:
    predict_week must read the current season from BigQuery instead."""
    import pipeline as cp

    bq = cfb_games(seasons=(2025, 2026))
    client = unittest.mock.MagicMock()
    client.query.return_value.to_dataframe.return_value = bq
    with unittest.mock.patch("google.cloud.bigquery.Client", return_value=client):
        got = cp.load_played_games()
    assert 2026 in set(got.season)


# ----------------------------------------------------------------- NFL wiring
def nfl_schedule(seasons=(2023, 2024, 2025), weeks=10, upcoming=(2025, 10), seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"N{i:02d}" for i in range(8)]
    power = {t: 6 - 1.5 * i for i, t in enumerate(teams)}
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            order = rng.permutation(teams)
            for k in range(0, 8, 2):
                h, a = order[k], order[k + 1]
                m = round(power[h] - power[a] + 2 + rng.normal(0, 10)) or 1
                played = (s, w) < upcoming
                rows.append({
                    "game_id": f"{s}_{w:02d}_{a}_{h}", "season": s, "week": w,
                    "game_type": "REG", "gameday": f"{s}-09-{w + 5:02d}",
                    "home_team": h, "away_team": a,
                    "home_score": 20 + m if played else np.nan,
                    "away_score": 20 if played else np.nan,
                    "result": m if played else np.nan,
                    "home_rest": 7, "away_rest": 7, "div_game": 0,
                    "location": "Home", "spread_line": 1.0, "total_line": 44.0,
                })
    return pd.DataFrame(rows)


def nfl_epa(schedule, through=(2025, 9)):
    rows = []
    for r in schedule.itertuples():
        if (r.season, r.week) > through or pd.isna(r.result):
            continue
        for team, sign in ((r.home_team, 1), (r.away_team, -1)):
            rows.append({"season": r.season, "week": r.week, "team": team,
                         "off_epa_play": 0.01 * sign * r.result,
                         "def_epa_play": -0.01 * sign * r.result})
    df = pd.DataFrame(rows)
    for c in ("off_pass_epa", "off_rush_epa", "def_pass_epa", "def_rush_epa",
              "off_success_rate", "def_success_rate", "off_explosive_rate",
              "def_explosive_rate"):
        df[c] = 0.0
    return df


def test_nfl_predict_week_carries_epa():
    """The regression: production predicted with net_epa_8g NULL on every row."""
    import predict_nfl as pn
    from data import completed_games

    sched = nfl_schedule()
    rows = pn.predict_week(2025, 10, played=completed_games(sched), schedule=sched,
                           epa=nfl_epa(sched), now=pd.Timestamp("2000-01-01", tz="UTC"))
    assert len(rows) == 4
    assert rows.net_epa_8g.notna().all() and (rows.net_epa_8g != 0).any()
    assert set(rows.model_version) == {pn.MODEL_VERSION}


def test_nfl_predict_refuses_without_epa():
    import predict_nfl as pn

    with pytest.raises(RuntimeError, match="team_week_epa"):
        pn._require_epa(None, 2026)
    with pytest.raises(RuntimeError, match="2025"):
        pn._require_epa(pd.DataFrame({"season": [2023, 2024]}), 2026)


def test_nfl_epa_era_keeps_seasons_without_epa_yet():
    import predict_nfl as pn

    games = pd.DataFrame({"season": [2005, 2006, 2025, 2026]})
    kept = pn._epa_era(games, pd.DataFrame({"season": [2006, 2025]}))
    assert kept.season.tolist() == [2006, 2025, 2026]


def test_nfl_load_epa_reads_bigquery_then_cache():
    import predict_nfl as pn

    bq = pd.DataFrame({"season": [2026], "week": [1], "team": ["KC"]})
    with unittest.mock.patch("bq_io.query", return_value=bq):
        assert pn.load_epa().equals(bq)
    with unittest.mock.patch("bq_io.query", side_effect=RuntimeError("no ADC")), \
            unittest.mock.patch("epa.EPA_CACHE") as cache:
        cache.exists.return_value = False
        assert pn.load_epa() is None
        with pytest.raises(RuntimeError):
            pn.load_epa(source="bq")


def test_nfl_epa_lag_is_detected():
    import predict_nfl as pn
    from data import completed_games

    sched = nfl_schedule()
    played = completed_games(sched)
    assert pn._epa_lag_weeks(played, nfl_epa(sched, through=(2025, 9)), 2025) == 0
    assert pn._epa_lag_weeks(played, nfl_epa(sched, through=(2025, 7)), 2025) == 2


def test_nfl_ridge_predict_week_needs_no_epa():
    import predict_nfl as pn
    from data import completed_games

    sched = nfl_schedule()
    rows = pn.predict_week(2025, 10, model="ridge", played=completed_games(sched),
                           schedule=sched, now=pd.Timestamp("2000-01-01", tz="UTC"))
    assert len(rows) == 4 and set(rows.model_version) == {pn.RIDGE_MODEL_VERSION}
    np.testing.assert_allclose(rows.home_win_probability,
                               mr.win_prob(rows.predicted_home_margin,
                                           mr.NFL_RIDGE.sigma))
    assert "spread_line" in rows   # NFL keeps nflverse's line, as before


def test_nfl_epa_rating_covariate_is_causal():
    import predict_nfl as pn
    from data import completed_games

    sched = nfl_schedule(upcoming=(2099, 1))
    games = completed_games(sched)
    epa = nfl_epa(sched, through=(2099, 1))
    base = pn.ridge_frame(games, epa=epa)
    epa2 = epa.copy()
    epa2.loc[(epa2.season == 2025) & (epa2.week >= 5), "off_epa_play"] = 9.0
    poisoned = pn.ridge_frame(games, epa=epa2)
    early = ((base.season < 2025) | (base.week <= 5)).to_numpy()
    np.testing.assert_allclose(base.epa_rating_diff[early],
                               poisoned.epa_rating_diff[early], equal_nan=True)


def test_win_pct_is_named_for_its_window():
    import features

    st = features._TeamState()
    for _ in range(10):
        st.update(21, 14, 1)
    snap = st.snapshot("home")
    assert "home_win_pct_8g" in snap and "home_win_pct_season" not in snap
    assert len(st.results) == features.LONG_WINDOW
