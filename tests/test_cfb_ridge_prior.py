"""The CFB ridge-with-prior shadow (cfb_v3_ridge_prior).

What is pinned: an empty or all-zero prior IS the plain ridge; a prior pulls ratings toward itself and stands in for a team with no game in the
window; the prior for a season is built only from earlier seasons; it recovers persistent program strength; the predict path emits its own
model version and equals the headline ridge when there is no prior; and the handler writes it only to its own table, never at the headline's
expense. Synthetic data and mocks only: no BigQuery, no ESPN.
"""
import os
import sys
import unittest.mock

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "src", "cfb"))
sys.path.insert(0, os.path.join(HERE, "..", "src", "nfl"))

import margin_ridge as mr  # noqa: E402
from test_margin_ridge import cfb_games, league  # noqa: E402

CFG = mr.CFB_RIDGE.with_(covariates=())      # synthetic leagues have no division covariate
SEASONS = tuple(range(2018, 2026))


def _persistent_league(seed=0):
    """Programs whose strength carries across seasons with a little drift, so last seasons predict the next."""
    rng = np.random.default_rng(seed)
    base = {f"T{i:02d}": (i - 6) * 2.0 for i in range(12)}
    frames = []
    for k, s in enumerate(SEASONS):
        strength = {t: v + rng.normal(0, 1.0) for t, v in base.items()}
        g, _ = league(seasons=(s,), weeks=10, noise=8.0, seed=seed + k, strength=strength)
        frames.append(g)
    return pd.concat(frames, ignore_index=True), base


# ------------------------------------------------------------------------------------------------------------------------------- fit
def test_an_empty_or_zero_prior_is_exactly_the_plain_ridge():
    g, _ = league(noise=3.0)
    t = mr.time_index(2024, 10)
    plain = mr.fit(g, t, CFG).ratings
    for prior in (None, {}, {team: 0.0 for team in plain}):
        np.testing.assert_allclose([mr.fit(g, t, CFG, prior=prior).ratings[k] for k in plain], list(plain.values()), atol=1e-12)


def test_a_prior_pulls_ratings_toward_itself_and_stands_in_for_unseen_teams():
    g, _ = league(noise=3.0)
    t = mr.time_index(2023, 3)                                   # thin window: two weeks of games
    plain = mr.fit(g, t, CFG.with_(min_train=1)).ratings
    prior = {team: plain[team] + 10.0 for team in plain} | {"NEW": 7.5}
    pulled = mr.fit(g, t, CFG.with_(min_train=1), prior=prior).ratings
    assert all(pulled[k] > plain[k] for k in plain)              # every rating moved toward its prior
    assert pulled["NEW"] == 7.5                                  # no game in the window: the prior is the rating


# ---------------------------------------------------------------------------------------------------------------------- program prior
def test_the_prior_for_a_season_uses_only_earlier_seasons():
    g, _ = _persistent_league()
    div = {t: "fbs" for t in set(g.home_team) | set(g.away_team)}
    base = mr.program_prior(g, 2025, CFG, div)
    assert base
    later = g.copy()
    later.loc[later.season >= 2025, "margin"] *= -1                 # rewrite the season itself
    assert mr.program_prior(later, 2025, CFG, div) == base
    earlier = g.copy()
    earlier.loc[earlier.season == 2024, "margin"] *= -1             # rewrite last season
    assert mr.program_prior(earlier, 2025, CFG, div) != base


def test_the_prior_recovers_persistent_program_strength():
    g, base = _persistent_league(seed=3)
    div = {t: "fbs" for t in base}
    prior = mr.program_prior(g, 2025, CFG, div)
    teams = sorted(base)
    assert np.corrcoef([prior[t] for t in teams], [base[t] for t in teams])[0, 1] > 0.9


def test_a_division_too_thin_for_its_regression_gets_no_prior():
    g, base = _persistent_league()
    div = {t: "fbs" for t in base} | {"T00": "fcs", "T01": "fcs"}   # two FCS programs: far fewer than 30 team-seasons
    prior = mr.program_prior(g, 2025, CFG, div)
    assert "T00" not in prior and "T01" not in prior and "T05" in prior


# ------------------------------------------------------------------------------------------------------------------------ predict path
def _predict(model):
    import pipeline as cp

    g = cfb_games(seasons=(2021, 2022, 2023, 2024, 2025), upcoming_week=6)
    played, upcoming = g[g.home_won.notna()], g[g.home_won.isna()]
    return cp.predict_week(2025, 6, model=model, played=played, upcoming=upcoming, now=pd.Timestamp("2000-01-01", tz="UTC"))


def test_predict_week_emits_the_prior_version_and_ridge_shaped_rows():
    import pipeline as cp

    rows, plain = _predict("ridge_prior"), _predict("ridge")
    assert set(rows.model_version) == {cp.PRIOR_MODEL_VERSION} and len(rows) == len(plain)
    assert {"predicted_home_margin", "home_power_rating", "away_power_rating"} <= set(rows)
    np.testing.assert_allclose(rows.home_win_probability, mr.win_prob(rows.predicted_home_margin, mr.CFB_RIDGE.sigma))
    assert not np.allclose(rows.predicted_home_margin, plain.predicted_home_margin)   # the prior changed something
    assert np.corrcoef(rows.predicted_home_margin, plain.predicted_home_margin)[0, 1] > 0.9


def test_with_no_prior_the_shadow_is_the_headline_ridge(monkeypatch):
    monkeypatch.setattr(mr, "program_prior", lambda *a, **k: {})
    rows = _predict("ridge_prior")
    plain = _predict("ridge")
    np.testing.assert_allclose(rows.predicted_home_margin.to_numpy(), plain.predicted_home_margin.to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------------------------------------------------------- handler
def _handler(monkeypatch, predict, state=None):
    import backfill_cfb
    import model_control
    import pipeline
    from test_cfb_drive_sim import _clear_env, _load_main, _Req

    main = _load_main()
    _clear_env(monkeypatch)
    monkeypatch.delenv("FPI_SNAPSHOT", raising=False)
    with unittest.mock.patch.object(model_control, "get_state", return_value=state), \
         unittest.mock.patch.object(pipeline, "predict_week", side_effect=predict) as pw, \
         unittest.mock.patch.object(backfill_cfb, "replace_game_ids", return_value=1) as rep, \
         unittest.mock.patch.object(backfill_cfb, "ensure_datasets"):
        out, status = main.cfb_pipeline(_Req({"mode": "predict_week", "season": 2026, "week": 11}))
    return out, status, pw, rep


def _rows(model):
    return pd.DataFrame({"game_id": ["g1"], "home_win_probability": [0.6], "confidence_tier": ["low"], "model_version": [model]})


def test_the_shadow_writes_only_its_own_table(monkeypatch):
    out, status, pw, rep = _handler(monkeypatch, lambda s, w, model="xgb", **k: _rows(model))
    assert status == 200 and out["steps"]["shadow_ridge_prior"] == 1
    written = {c.args[2]: c.args[0]["model_version"].iloc[0] for c in rep.call_args_list}
    assert written["game_predictions_ridge_prior_shadow"] == "ridge_prior" and written["game_predictions"] == "ridge"


def test_a_failing_shadow_never_costs_the_headline(monkeypatch):
    def predict(s, w, model="xgb", **k):
        if model == "ridge_prior":
            raise RuntimeError("singular")
        return _rows(model)

    out, status, pw, rep = _handler(monkeypatch, predict)
    assert status == 200 and out["status"] == "ok" and out["steps"]["predicted"] == 1
    assert out["steps"]["shadow_ridge_prior"] == {"error": "singular"}
    assert "game_predictions_ridge_prior_shadow" not in [c.args[2] for c in rep.call_args_list]


def test_a_paused_shadow_is_skipped(monkeypatch):
    from test_cfb_drive_sim import _ctl_state

    out, status, pw, rep = _handler(monkeypatch, lambda s, w, model="xgb", **k: _rows(model),
                                    _ctl_state({"target": "ridge_prior", "run_state": "paused"}))
    assert status == 200 and "shadow_ridge_prior" not in out["steps"]
    assert "ridge_prior" not in [c.kwargs.get("model") for c in pw.call_args_list]


# ------------------------------------------------------------------------------------------------------------------- division map
def _div_games():
    """A program that moved up (M), an FBS program with one FCS-feed game (B), and an FCS program that only ever crossed (X)."""
    rows = []
    def game(s, h, a, div, cross=0):
        rows.append({"season": s, "home_team": h, "away_team": a, "division": div, "cross_division": cross})
    for s in (2024, 2025):
        game(s, "M", "F1", "fcs"); game(s, "M", "F2", "fcs")         # M in the FCS feed while it was FCS
        game(s, "B", "F1", "fbs"); game(s, "B", "F2", "fbs"); game(s, "B", "C1", "fcs")
    game(2026, "M", "F1", "fbs"); game(2026, "M", "F2", "fbs")       # M moved to FBS in 2026
    game(2026, "F1", "F2", "fbs"); game(2026, "C1", "C2", "fcs")
    game(2026, "F1", "X", "fbs", cross=1)                            # X: one cross-division game only
    return pd.DataFrame(rows)


def test_majority_divisions_follows_the_latest_season_and_ignores_stray_feed_tags():
    import pipeline as cp

    div = cp.majority_divisions(_div_games())
    assert div["M"] == "fbs"                 # the latest season, not the most games overall
    assert div["B"] == "fbs"                 # two FBS tags beat one stray FCS-feed tag
    assert div["X"] == "fcs"                 # crossed only: inferred from its FBS opponent
    assert div["C1"] == "fcs"


def test_majority_divisions_does_not_depend_on_row_order():
    import pipeline as cp

    g = _div_games()
    first = cp.majority_divisions(g)
    for seed in range(5):
        assert cp.majority_divisions(g.sample(frac=1, random_state=seed)) == first


def test_the_ridge_division_term_does_not_depend_on_row_order():
    import pipeline as cp

    g = cfb_games(seasons=(2024, 2025))
    first = cp.ridge_frame(g).set_index("game_id")["fbs_diff"].sort_index()
    for seed in range(3):
        again = cp.ridge_frame(g.sample(frac=1, random_state=seed)).set_index("game_id")["fbs_diff"].sort_index()
        pd.testing.assert_series_equal(again, first)


def test_a_stray_fcs_feed_tag_no_longer_moves_an_fbs_program():
    import pipeline as cp

    g = cfb_games(seasons=(2024, 2025))
    stray = g[(g.home_team == "F0") & (g.season == 2025)].head(1).copy()
    stray["division"], stray["cross_division"], stray["game_id"] = "fcs", 0, "stray"
    rg = cp.ridge_frame(pd.concat([g, stray], ignore_index=True))
    cross = rg[(rg.home_team == "F0") & rg.away_team.str.startswith("C")]
    assert (cross.fbs_diff == 1).all()           # F0 still counts as FBS in its games against FCS teams
