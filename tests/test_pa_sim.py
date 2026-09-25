"""Tests for the PA simulator. The key one is invariance: the Monte Carlo engine and
the exact analytic chain are independent implementations of the same process, so they
must agree on the run distribution."""
import sys, os
import numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pa_sim import CLASSES
from pa_sim.markov import NB, NO, RA, END, half_inning, game_runs
from pa_sim.sim import simulate_game, to_cum
from pa_sim.game import apply_scale, win_prob, expected_runs
from pa_sim.rates import log5, estimate_k

LEAGUE = np.array([0.2219, 0.0944, 0.1501, 0.0443, 0.0042, 0.0312, 0.4325, 0.0214])
LEAGUE = LEAGUE / LEAGUE.sum()
ORDER = np.tile(LEAGUE, (9, 1))


def test_transition_table_conserves_runners():
    """Runs scored + runners left + outs recorded must equal runners entering + batter."""
    for bases in range(8):
        on = bin(bases).count("1")
        for outs in range(3):
            for ei, ev in enumerate(CLASSES):
                nb, no, ra = NB[bases, outs, ei], NO[bases, outs, ei], RA[bases, outs, ei]
                left = bin(int(nb)).count("1") if not END[bases, outs, ei] else 0
                outs_made = (3 if END[bases, outs, ei] else int(no)) - outs
                if END[bases, outs, ei]:
                    assert ra + outs_made <= on + 1 + 1
                else:
                    assert ra + left + outs_made == on + 1, (ev, bases, outs)


def test_home_run_scores_everyone():
    hr = CLASSES.index("HR")
    assert RA[7, 0, hr] == 4 and NB[7, 0, hr] == 0


def test_walk_does_not_score_unless_loaded():
    bb = CLASSES.index("BB")
    for bases in range(7):
        assert RA[bases, 0, bb] == 0
    assert RA[7, 0, bb] == 1


def test_strikeout_ends_inning_on_third_out():
    k = CLASSES.index("K")
    assert END[0, 2, k] and not END[0, 1, k]


def test_half_inning_mass_conserved():
    hi = half_inning(ORDER)
    assert np.isclose(hi.sum(), 9.0, atol=1e-9)


def test_game_runs_is_a_distribution():
    d = game_runs([ORDER] * 9)
    assert np.isclose(d.sum(), 1.0, atol=1e-9) and (d >= 0).all()


def test_monte_carlo_matches_analytic_chain():
    """The load-bearing test: two independent implementations must agree."""
    exact = expected_runs(game_runs([ORDER] * 9))
    cum = to_cum([ORDER] * 9)
    r = simulate_game(cum, cum, n_episodes=40000, seed=7)
    se = np.std(r["away_runs_reg"]) / np.sqrt(40000)
    assert abs(r["mean_away_reg"] - exact) < 4 * se, (r["mean_away_reg"], exact, se)


def test_home_scores_less_than_away_in_regulation():
    """Home skips the bottom of the 9th when already ahead, so it scores less."""
    cum = to_cum([ORDER] * 9)
    r = simulate_game(cum, cum, n_episodes=20000, seed=3)
    assert r["mean_home_reg"] < r["mean_away_reg"]


def test_identical_lineups_give_near_even_odds():
    cum = to_cum([ORDER] * 9)
    r = simulate_game(cum, cum, n_episodes=20000, seed=5)
    assert 0.47 < r["home_win_prob"] < 0.56


def test_log5_is_neutral_at_league_average():
    """A league-average batter against a league-average pitcher must stay league-average."""
    out = log5(LEAGUE[None, :], LEAGUE[None, :], LEAGUE[None, :])[0]
    assert np.allclose(out, LEAGUE, atol=1e-9)


def test_log5_moves_the_right_way():
    hi_k = LEAGUE.copy(); hi_k[CLASSES.index("K")] *= 1.5; hi_k /= hi_k.sum()
    out = log5(LEAGUE[None, :], hi_k[None, :], LEAGUE[None, :])[0]
    assert out[CLASSES.index("K")] > LEAGUE[CLASSES.index("K")]


def test_apply_scale_increases_offence():
    lo = expected_runs(game_runs([ORDER] * 9))
    hi = expected_runs(game_runs([apply_scale(ORDER, 1.3)] * 9))
    assert hi > lo


def test_win_prob_symmetry():
    d = game_runs([ORDER] * 9)
    assert abs(win_prob(d, d, p_extra_home=0.5) - 0.5) < 0.01


def test_estimate_k_is_bounded():
    rng = np.random.default_rng(0)
    n = rng.integers(50, 600, 400).astype(float)
    c = rng.binomial(n.astype(int), 0.2).astype(float)
    assert 20.0 <= estimate_k(c, n, 0.2) <= 5000.0


# ---------------------------------------------------------------- v2 (experimental)
from pa_sim import v2


def test_v2_fixed_transitions_are_distributions():
    cum, nb, no, ra = v2.fixed_transitions()
    assert np.allclose(cum[..., -1], 1.0)
    assert (np.diff(cum, axis=-1) >= -1e-12).all()
    hr = v2.CLASSES.index("HR")
    assert ra[7, 0, hr, 0] == 4 and nb[7, 0, hr, 0] == 0


def test_v2_tilt_hits_target_mean():
    from scipy.stats import poisson
    import importlib.util, pathlib
    k = np.arange(31)
    pmf = poisson.pmf(k, 8.5)[None, :]; pmf = pmf / pmf.sum()
    spec = importlib.util.spec_from_file_location(
        "ev44", pathlib.Path(__file__).parent.parent / "research" / "backtest_2026" / "44_eval.py")
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    q = ev.tilt(pmf, np.array([9.2]))
    assert abs((q[0] * k).sum() - 9.2) < 1e-3 and np.isclose(q.sum(), 1.0)


def test_v2_simulator_symmetric_game_is_near_even():
    """League-average everything, no home edge in the tables -> home wins ~ half
    (slightly above: last at-bat); runs per team in a sane range."""
    class Stub:
        cfg = v2.Config(hook="curve")
        curve = np.array([1, 1, 1, .97, .9, .75, .5, .25, .1])
        trans = v2.fixed_transitions()
        def tables(self, g):
            L = np.array([.22, .09, .15, .045, .004, .03, .007, .23, .224]); L = L / L.sum()
            return np.tile(L, (2, 9, 4, 1))
    g = v2.GameSpec([0] * 9, [0] * 9, 1, 2, "H", "A", 0, 9, True)
    res = v2.simulate(Stub(), [g], n=20000, seed=1)
    s = v2.summarize(res, 1)
    assert 0.47 < s["p_home"][0] < 0.56
    assert 3.0 < res["away"].mean() < 5.5
    assert np.isclose(s["tot_hist"].sum(), 1.0) and np.isclose(s["f5"].sum(), 1.0)
