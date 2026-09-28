"""Tests for the rest-of-season Monte Carlo (src/season_sim), an experiment in shadow.

Covers the three things that would make it silently wrong - the linear refit, the NFL
tiebreakers, the CFP formats - and the write safety of the `season_sim` mode.
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
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "nfl"))

from season_sim import cfb as cfbm  # noqa: E402
from season_sim import engine as eng  # noqa: E402
from season_sim import nfl as nflm  # noqa: E402
from season_sim import run, store  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "nflverse_games_2022_2025.csv")
DDL = os.path.join(HERE, "..", "scripts", "gcp", "football", "create_season_sim_tables.sql")
ALTER = os.path.join(HERE, "..", "scripts", "gcp", "football", "alter_season_sim_per_game.sql")


@pytest.fixture(scope="module")
def sched():
    return pd.read_csv(FIXTURE)


# ------------------------------------------------------------------ engine
def _toy_league(seed=0, n_teams=8, seasons=(2024, 2025), weeks=10):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strength = rng.normal(0, 7, n_teams)
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            order = rng.permutation(n_teams)
            for k in range(0, n_teams, 2):
                h, a = order[k], order[k + 1]
                m = strength[h] - strength[a] + 2 + rng.normal(0, 12)
                rows.append({"season": s, "week": w, "home_team": teams[h],
                             "away_team": teams[a], "neutral": 0.0, "margin": round(m)})
    return pd.DataFrame(rows)


def test_linear_refit_equals_a_direct_fit():
    """b = c + G y_sim must be exactly the ridge fit on the frame with y_sim filled in."""
    g = _toy_league()
    g["sim"] = (g["season"] == 2025) & (g["week"] > 4)
    y_sim = np.random.default_rng(1).normal(0, 14, g["sim"].sum()).round()
    frame = eng.ridge_frame_from(g.assign(margin=np.where(g["sim"], np.nan, g["margin"])))
    cfg = eng.mr.NFL_RIDGE.with_(min_train=5)
    ops = eng.RidgeOps(frame, cfg)
    t_now = eng.mr.time_index(2025, 8)
    c, G, cols, _, _ = ops.solve(t_now)
    b_lin = c + G @ y_sim[cols]
    filled = frame.copy()
    filled.loc[filled["sim"], "margin"] = y_sim
    direct = eng.mr.fit(filled.drop(columns="sim"), t_now, cfg)
    for t, r in direct.ratings.items():
        assert b_lin[ops.teams.get_loc(t)] == pytest.approx(r, abs=1e-8)
    assert b_lin[ops.T] * eng.mr.HFA_SCALE == pytest.approx(direct.hfa, abs=1e-8)


def test_nullable_bigquery_dtypes_become_plain_numpy():
    g = _toy_league().head(20)
    g["season"] = g["season"].astype("Int64")
    g["margin"] = g["margin"].astype("Float64")
    g.loc[3, "margin"] = pd.NA
    g["sim"] = g["margin"].isna()
    f = eng.ridge_frame_from(g)
    assert f["margin"].dtype == np.float64 and np.isnan(f.loc[3, "margin"])
    assert f["season"].dtype.kind == "i"


def test_rating_draws_keep_single_game_variance(sched):
    """Deflation: posterior variance of a matchup + per-game noise = the fitted sigma^2."""
    frame = nflm.build_frame(sched, 2025, as_of_week=3)
    ops = eng.RidgeOps(frame, eng.mr.NFL_RIDGE)
    res = eng.simulate(ops, 2025, eng.VARIANTS["draw"], 4000, np.random.default_rng(0))
    Xs = ops.X[ops.sim_idx]
    means = res.B_play @ Xs.T               # per-path game means, (S, n)
    total = means.var(0) + res.noise_sigma ** 2
    assert np.sqrt(total.mean()) == pytest.approx(eng.mr.NFL_RIDGE.sigma, rel=0.05)
    point = eng.simulate(ops, 2025, eng.VARIANTS["point"], 10, np.random.default_rng(0))
    assert point.noise_sigma == eng.mr.NFL_RIDGE.sigma
    assert np.allclose(point.B_play, point.B_play[0])


def test_norm_ppf_inverts_the_cdf():
    p = np.array([1e-6, 0.01, 0.2, 0.5, 0.8, 0.99])
    assert np.allclose(eng.mr.norm_cdf(eng.norm_ppf(p)), p, atol=1e-8)


# ------------------------------------------------------------------ NFL tiebreakers
@pytest.mark.parametrize("season", [2023, 2024, 2025])
def test_standings_reproduce_the_real_playoff_field(sched, season):
    """Seeds from the real results must match the real bracket: field, byes, pairings.
    (Measured over 2002-2025 in the backtest: 48 of 48 conference-seasons exact.)"""
    frame = nflm.build_frame(sched, season)
    ns = nflm.NflSeason(frame, season)
    st = ns.standings(frame["margin"].to_numpy()[ns.rows][None, :], np.random.default_rng(0))
    g = nflm.normalise_schedule(sched)
    wc = g[(g["season"] == season) & (g["game_type"] == "WC")]
    dv = g[(g["season"] == season) & (g["game_type"] == "DIV")]
    for ci, conf in enumerate(nflm.CONFS):
        mine = [nflm.TEAMS[i] for i in st["seeds"][0, ci]]
        assert {t for t in set(wc.home_team) | set(wc.away_team) | set(dv.home_team)
                if nflm.CONF_OF[t] == conf} == set(mine)
        bye = set(dv.home_team) - set(wc.home_team) - set(wc.away_team)
        assert mine[0] in bye
        pairs = {frozenset(p) for p in zip(wc.home_team, wc.away_team) if nflm.CONF_OF[p[0]] == conf}
        assert pairs == {frozenset((mine[1], mine[6])), frozenset((mine[2], mine[5])),
                         frozenset((mine[3], mine[4]))}


class _FakeSeason:
    def __init__(self, n, meetings, div):
        self.Ml = [[0] * n for _ in range(n)]
        for i, j, k in meetings:
            self.Ml[i][j] = self.Ml[j][i] = k
        self.opps = [set(j for j in range(n) if self.Ml[i][j]) for i in range(n)]
        self.div = div


def _ctx(season, h2h, pct=None, divp=None, confp=None, sov=None):
    n = len(h2h)
    z = [0.0] * n
    return nflm._Ctx(season, pct or [0.5] * n, divp or z, confp or z, sov or z, z, z, z,
                     h2h, np.random.default_rng(0))


def test_division_three_way_head_to_head_then_revert():
    # A, B, C in one division, twice each. A 3-1 in the group; B and C 2-2 and 1-3.
    s = _FakeSeason(3, [(0, 1, 2), (0, 2, 2), (1, 2, 2)], ["D"] * 3)
    h2h = [[0, 1, 2], [1, 0, 1], [0, 1, 0]]
    c = _ctx(s, h2h)
    assert c.div_top([0, 1, 2]) == 0
    # Remove A: B and C restart at two-club head-to-head, where B split 1-1 with C; the
    # next step is division record, which C has better.
    c2 = _ctx(s, h2h, divp=[0.5, 0.4, 0.6])
    assert nflm._top_k(c2, [1, 2], c2.div_top, 2) == [2, 1]


def test_wild_card_three_way_uses_sweep_only():
    # Three clubs from different divisions. 0 beat 1 and 2 -> sweep, 0 advances even
    # though 1 has the better conference record.
    s = _FakeSeason(3, [(0, 1, 1), (0, 2, 1)], ["A", "B", "C"])
    h2h = [[0, 1, 1], [0, 0, 0], [0, 0, 0]]
    assert _ctx(s, h2h, confp=[0.5, 0.9, 0.1]).wc_top([0, 1, 2]) == 0
    # No sweep (1 and 2 never met, 0 split): falls to conference record.
    s2 = _FakeSeason(3, [(0, 1, 2), (0, 2, 2)], ["A", "B", "C"])
    h2h2 = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    assert _ctx(s2, h2h2, confp=[0.5, 0.9, 0.1]).wc_top([0, 1, 2]) == 1


def test_wild_card_same_division_pair_uses_the_division_tiebreaker():
    s = _FakeSeason(2, [(0, 1, 2)], ["D", "D"])
    h2h = [[0, 2], [0, 0]]  # 0 swept 1
    assert _ctx(s, h2h, confp=[0.2, 0.9]).wc_top([0, 1]) == 0


# ------------------------------------------------------------------ CFP formats
def _field_case(season, champ_ranks, g6_team=None):
    T = 30
    score = -np.arange(T, dtype=float)[None, :]            # team i is ranked i+1
    confs = np.array(["8", "5", "1", "4", "151", "17", "37", "12", "15", "9"] * 3)
    if g6_team is not None:
        confs[:] = np.where(np.isin(confs, list(cfbm.GROUP6)), "5", confs)
        confs[g6_team] = "151"
    champ_of = {c: np.array([r]) for c, r in champ_ranks.items()}
    return cfbm.select_field(score, champ_of, confs, season)[0].tolist()


def test_cfp_2024_top_four_seeds_go_to_conference_champions():
    # Champions ranked 3, 6, 9, 14, 20 (0-based). Top-4 seeds = the four best champions.
    seeds = _field_case(2024, {"8": 3, "5": 6, "1": 9, "4": 14, "151": 20})
    assert seeds[:4] == [3, 6, 9, 14]
    assert 20 in seeds and len(set(seeds)) == 12


def test_cfp_2025_seeds_straight_by_ranking():
    seeds = _field_case(2025, {"8": 3, "5": 6, "1": 9, "4": 14, "151": 20})
    assert seeds[:4] == [0, 1, 2, 3]
    assert seeds[-1] == 20 and 14 in seeds


def test_cfp_2026_power_champions_plus_best_group_of_six():
    # A Big 12 champion ranked 25th gets in; the best G6 team (ranked 18th, not a
    # champion) gets the G6 bid; seeding is straight by ranking.
    seeds = _field_case(2026, {"8": 0, "5": 1, "1": 2, "4": 25}, g6_team=18)
    assert 25 in seeds and 18 in seeds
    assert seeds == sorted(seeds)


def test_four_team_format_before_2024():
    assert _field_case(2023, {"8": 5}) == [0, 1, 2, 3]


# ------------------------------------------------------------------ tables + writes
def _ddl_columns(table, path=DDL, dataset="nfl_season"):
    sql = open(path).read()
    block = re.search(rf"{dataset}\.{table}` \((.*?)\n\)", sql, re.S).group(1)
    return [ln.split()[0] for ln in block.strip().splitlines()
            if ln.strip() and not ln.strip().startswith("--")]


@pytest.fixture(scope="module")
def nfl_tables(sched):
    o = run.sim_nfl(sched, 2025, as_of_week=6, n_sims=300, seed=1)
    return o, run.tables(o)


def test_tables_match_the_ddl(nfl_tables):
    o, (team, bracket) = nfl_tables
    for ds in ("nfl_season", "cfb_season"):
        assert list(team.columns) == _ddl_columns("season_sim_team", dataset=ds)
        assert list(bracket.columns) == _ddl_columns("season_sim_bracket", dataset=ds)
        assert list(run.games_table(o).columns) == _ddl_columns("season_sim_games", dataset=ds)
        assert _ddl_columns("season_sim_games", ALTER, ds) == _ddl_columns("season_sim_games", dataset=ds)


def test_alter_adds_exactly_the_new_team_columns():
    old = set(_ddl_columns("season_sim_team")) - {
        "rem_wins_mean", "rem_wins_dist", "projected_wins_games", "modal_sequence",
        "modal_sequence_freq", "modal_sequence_record_p"}
    new = [c for c in _ddl_columns("season_sim_team") if c not in old]
    for ds in ("nfl_season", "cfb_season"):
        block = re.search(rf"ALTER TABLE `hankstank\.{ds}\.season_sim_team`(.*?);",
                          open(ALTER).read(), re.S).group(1)
        assert re.findall(r"ADD COLUMN IF NOT EXISTS (\w+)", block) == new


# ------------------------------------------------------------------ per-game projections
@pytest.fixture(scope="module")
def nfl_games(sched):
    o = run.sim_nfl(sched, 2025, as_of_week=6, n_sims=4000, seed=7)
    team, _ = run.tables(o)
    return o, team, run.games_table(o)


def _team_p_wins(games, team):
    h = games[games["home"] == team]
    a = games[games["away"] == team]
    return pd.concat([h["p_home_win"], 1 - a["p_home_win"]])


def test_per_game_p_win_sums_to_expected_remaining_wins(sched, nfl_games):
    """Per-game P(win) summed over a team's remaining games is its expected remaining wins:
    exactly against the team table (the same simulated seasons), and to Monte Carlo
    tolerance against an independent run with another seed."""
    o, team, games = nfl_games
    other, _ = run.tables(run.sim_nfl(sched, 2025, as_of_week=6, n_sims=4000, seed=8))
    other = other.set_index("team")
    for r in team.itertuples():
        pw = _team_p_wins(games, r.team)
        assert len(pw) == r.remaining_games
        assert pw.sum() == pytest.approx(r.rem_wins_mean, abs=1e-9)
        assert r.mean_wins == pytest.approx(r.wins + 0.5 * r.ties + r.rem_wins_mean, abs=1e-9)
        # Two independent runs differ by MC error: sd = sqrt(2 Var(rem wins) / S). The
        # variance comes from the simulated distribution itself, since the rating draws
        # correlate a team's games and a Bernoulli bound would be too tight. Allow 5 sd.
        dist = np.array(json.loads(r.rem_wins_dist))
        k = np.arange(len(dist))
        var = float((k * k * dist).sum() - (k * dist).sum() ** 2)
        tol = 5 * np.sqrt(2 * var / 4000) + 1e-3
        assert pw.sum() == pytest.approx(other.loc[r.team, "rem_wins_mean"], abs=tol)


def test_games_table_is_one_row_per_remaining_game(nfl_games):
    o, team, games = nfl_games
    assert games["game_id"].is_unique
    assert len(games) == team["remaining_games"].sum() / 2   # NFL: both sides are teams
    assert set(games["week"]) == set(range(7, 19))
    assert ((games["p_home_win"] >= 0) & (games["p_home_win"] <= 1)).all()
    assert (games["margin_p10"] <= games["margin_mean"]).all()
    assert (games["margin_mean"] <= games["margin_p90"]).all()
    # A favourite by the mean margin is a favourite by P(win).
    fav = games["margin_mean"].abs() > 3
    assert ((games.loc[fav, "margin_mean"] > 0) == (games.loc[fav, "p_home_win"] > 0.5)).all()
    assert games["game_date"].map(lambda d: d.isoformat()).str.match(r"2025-(09|1[0-2])-|2026-01-").all()
    assert set(games["sport"]) == {"nfl"} and set(games["as_of_week"]) == {6}


def test_projected_wins_games_and_modal_sequence(nfl_games):
    o, team, games = nfl_games
    for r in team.itertuples():
        pw = _team_p_wins(games, r.team)
        ids = json.loads(r.projected_wins_games)
        assert len(ids) == int(np.floor(r.rem_wins_mean + 0.5))
        gid = pd.concat([games.loc[games["home"] == r.team, "game_id"],
                         games.loc[games["away"] == r.team, "game_id"]])
        p_of = dict(zip(gid, pw))
        chosen = [p_of[i] for i in ids]
        rest = [p for g, p in p_of.items() if g not in ids]
        assert chosen == sorted(chosen, reverse=True)
        assert not rest or min(chosen, default=1) >= max(rest) - 1e-12
        dist = json.loads(r.rem_wins_dist)
        assert len(dist) == r.remaining_games + 1 and sum(dist) == pytest.approx(1, abs=1e-4)
        assert sum(k * p for k, p in enumerate(dist)) == pytest.approx(r.rem_wins_mean, abs=1e-3)
        assert len(r.modal_sequence) == r.remaining_games and set(r.modal_sequence) <= {"W", "L"}
        assert 0 < r.modal_sequence_freq <= r.modal_sequence_record_p + 1e-12
        assert r.modal_sequence_record_p == pytest.approx(dist[r.modal_sequence.count("W")], abs=1e-5)


def test_modal_sequence_is_the_most_common_path():
    """Hand-built outcome: 3 games, the sequence W-L-W appears in 5 of 8 paths."""
    M = np.array([[1, -1, 1]] * 5 + [[1, 1, 1], [-1, -1, -1], [-1, 1, 1]], float)
    g = pd.DataFrame({"game_id": ["a", "b", "c"], "week": [5, 6, 7], "game_day": [None] * 3,
                      "home": ["X", "X", "Y"], "away": ["Y", "Z", "X"],
                      "home_name": ["X", "X", "Y"], "away_name": ["Y", "Z", "X"],
                      "neutral": [False] * 3})
    o = unittest.mock.Mock(teams=["X"], team_abbr={"X": "X"}, games=g, game_margins=M,
                           tie_half=False)
    f = run.team_game_fields(o)["X"]
    # X: home wins a (M>0), home in b, away in c (wins when M<0)
    assert f["modal_sequence"] == "WLL"
    assert f["modal_sequence_freq"] == pytest.approx(5 / 8)
    # paths: WLL x5, WWL, LLW, LWL -> 5 + 2 + 1 + 1 = 9 wins over 8 paths
    assert f["rem_wins_mean"] == pytest.approx(9 / 8)
    assert json.loads(f["rem_wins_dist"]) == [0.0, 0.875, 0.125, 0.0]
    assert f["modal_sequence_record_p"] == pytest.approx(7 / 8)
    assert json.loads(f["projected_wins_games"]) == ["a"]   # round(1.0) = 1


def test_nfl_outcome_invariants(nfl_tables):
    o, (team, bracket) = nfl_tables
    assert team["p_playoffs"].sum() == pytest.approx(14)
    assert team["p_division"].sum() == pytest.approx(8)
    assert team["p_bye"].sum() == pytest.approx(2)
    assert team["p_champion"].sum() == pytest.approx(1)
    assert (team["p_champion"] <= team["p_final"] + 1e-12).all()
    assert (team["p_final"] <= team["p_semis"] + 1e-12).all()
    for d in team["wins_dist"]:
        assert sum(json.loads(d)) == pytest.approx(1, abs=1e-4)
    modal = bracket[bracket["is_modal"] & (bracket["round"] == "seed")]
    assert len(modal) == 14 and modal["team"].nunique() == 14
    sb = bracket[bracket["is_modal"] & (bracket["round"] == "super_bowl")]
    assert len(sb) == 2


def test_dry_run_creates_no_bigquery_client(sched):
    with unittest.mock.patch("google.cloud.bigquery.Client",
                             side_effect=AssertionError("no BigQuery in dry_run")):
        out = store.run_mode("nfl", sched[sched["season"] <= 2025].assign(
            result=lambda d: np.where((d["season"] == 2025) & (d["week"] > 5), np.nan, d["result"])),
            2025, "p", "nfl_season", {"n_sims": 200}, dry_run=True)
    assert out["dry_run"] is True and out["written"] == 0 and out["as_of_week"] == 5
    reg = sched[(sched["season"] == 2025) & (sched["game_type"] == "REG") & (sched["week"] > 5)]
    assert out["game_rows"] == len(reg)


def test_write_is_scoped_to_one_season_and_week(nfl_tables):
    _, (team, bracket) = nfl_tables
    from google.cloud.bigquery import SchemaField

    fake = unittest.mock.MagicMock()
    fake.get_table.side_effect = lambda tid: unittest.mock.Mock(schema=[
        SchemaField(c, "STRING") for c in (team.columns if tid.endswith("team") else bracket.columns)])
    store.write(team, bracket, "p", "nfl_season", 2025, 6, client=fake)
    deletes = [c for c in fake.query.call_args_list]
    assert len(deletes) == 2
    for call in deletes:
        assert "WHERE season = @season AND as_of_week = @week" in call[0][0]
        params = {p.name: p.value for p in call[1]["job_config"].query_parameters}
        assert params == {"season": 2025, "week": 6}
    for call in fake.load_table_from_dataframe.call_args_list:
        cfg = call[1]["job_config"]
        assert cfg.create_disposition == "CREATE_NEVER" and cfg.write_disposition == "WRITE_APPEND"
    with pytest.raises(ValueError):
        store.write(team, bracket, "p", "nfl_season", 2025, 7, client=fake)


def test_games_write_is_scoped_and_checks_every_schema_first(nfl_tables):
    o, (team, bracket) = nfl_tables
    games = run.games_table(o)
    from google.cloud.bigquery import SchemaField

    cols = {"season_sim_team": team.columns, "season_sim_bracket": bracket.columns,
            "season_sim_games": games.columns}
    fake = unittest.mock.MagicMock()
    fake.get_table.side_effect = lambda tid: unittest.mock.Mock(schema=[
        SchemaField(c, "STRING") for c in cols[tid.split(".")[-1]]])
    out = store.write(team, bracket, "p", "nfl_season", 2025, 6, client=fake, games_df=games)
    assert out["season_sim_games"] == len(games) > 0
    tables = [c[0][0].split("`")[1] for c in fake.query.call_args_list]
    assert tables == ["p.nfl_season.season_sim_team", "p.nfl_season.season_sim_bracket",
                      "p.nfl_season.season_sim_games"]
    for call in fake.query.call_args_list:
        params = {p.name: p.value for p in call[1]["job_config"].query_parameters}
        assert params == {"season": 2025, "week": 6}
    with pytest.raises(ValueError):
        store.write(team, bracket, "p", "nfl_season", 2025, 6, client=fake,
                    games_df=games.assign(as_of_week=7))

    # The team table without the migrated columns: refuse before any DELETE.
    old = [c for c in team.columns if c not in ("rem_wins_mean", "modal_sequence")]
    fake2 = unittest.mock.MagicMock()
    fake2.get_table.side_effect = lambda tid: unittest.mock.Mock(schema=[
        SchemaField(c, "STRING") for c in (old if tid.endswith("team") else cols[tid.split(".")[-1]])])
    with pytest.raises(ValueError, match="alter_season_sim_per_game"):
        store.write(team, bracket, "p", "nfl_season", 2025, 6, client=fake2, games_df=games)
    fake2.query.assert_not_called()
    fake2.load_table_from_dataframe.assert_not_called()


def test_postseason_guard(sched):
    out = store.run_mode("nfl", sched, 2025, "p", "nfl_season", {"n_sims": 10}, dry_run=True)
    assert "skipped" in out and out["written"] == 0


# ------------------------------------------------------------------ Cloud Function modes
def _load_main(sport):
    path = os.path.join(SRC, sport, "main.py")
    spec = importlib.util.spec_from_file_location(f"{sport}_main_ss_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Req:
    def __init__(self, body):
        self._body = body

    def get_json(self, silent=True):
        return self._body


@pytest.mark.parametrize("sport,fn", [("nfl", "nfl_pipeline"), ("cfb", "cfb_pipeline")])
def test_season_sim_mode_accepts_dry_run_and_never_writes(sport, fn):
    main = _load_main(sport)
    calls = {}

    def fake_run_mode(sp, games, season, project, dataset, req, dry_run, client=None):
        calls["dry_run"] = dry_run
        return {"season": season, "dry_run": dry_run, "written": 0}

    with unittest.mock.patch.object(store, "run_mode", side_effect=fake_run_mode), \
         unittest.mock.patch.object(store, "write", side_effect=AssertionError("wrote")):
        if sport == "nfl":
            with unittest.mock.patch.dict(sys.modules, {"data": unittest.mock.Mock(
                    load_schedules=lambda refresh=False: pd.DataFrame())}):
                body, status = getattr(main, fn)(_Req({"mode": "season_sim", "dry_run": True}))
        else:
            with unittest.mock.patch.object(main, "season_sim_games", return_value=pd.DataFrame()):
                body, status = getattr(main, fn)(_Req({"mode": "season_sim", "dry_run": True}))
    assert status == 200, body
    assert body["dry_run"] is True and calls["dry_run"] is True


@pytest.mark.parametrize("sport,fn,mode", [("nfl", "nfl_pipeline", "ingest"),
                                           ("nfl", "nfl_pipeline", "score"),
                                           ("cfb", "cfb_pipeline", "ingest"),
                                           ("cfb", "cfb_pipeline", "backfill")])
def test_other_modes_still_refuse_dry_run(sport, fn, mode):
    body, status = getattr(_load_main(sport), fn)(_Req({"mode": mode, "dry_run": True}))
    assert status == 400 and "dry_run" in body["error"]


def test_cfb_season_sim_games_appends_only_unplayed_schedule():
    played = pd.DataFrame({"game_id": ["1", "2"], "season": [2026, 2026], "week": [1, 2],
                           "home_won": [1, 0]})
    slates = {2: pd.DataFrame({"game_id": ["2", "3"], "season": [2026, 2026], "week": [2, 2],
                               "home_won": [0, None]}),
              3: pd.DataFrame({"game_id": ["4"], "season": [2026], "week": [3], "home_won": [None]})}
    main = _load_main("cfb")
    out = main.season_sim_games(2026, played=played,
                                fetch=lambda s, w: slates.get(w, pd.DataFrame()))
    assert out["game_id"].tolist() == ["1", "2", "3", "4"]


def test_cfb_game_day_is_the_eastern_calendar_date():
    ts = pd.Series(pd.to_datetime(["2026-10-10 04:00", "2026-11-07 05:00", "2026-09-27 02:30",
                                   "2026-09-19 16:00"]))
    assert cfbm._eastern_day(ts).tolist() == ["2026-10-10", "2026-11-07", "2026-09-26",
                                              "2026-09-19"]
