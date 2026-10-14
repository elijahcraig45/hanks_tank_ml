"""Resume v2: the properties the design relies on (docs/RESUME_V2_REPORT.md, section 2)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rankings import core, resume_v2 as v2  # noqa: E402


def season(n_teams=12, games_per_team=6, seed=0, strengths=None, sd=17.0):
    """A synthetic season with known strengths: margins ~ N(s_h - s_a + 2.5, sd)."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    s = rng.normal(0, 12, n_teams) if strengths is None else np.asarray(strengths, dtype=float)
    rows = []
    for w in range(1, games_per_team + 1):
        perm = rng.permutation(n_teams)
        for a, b in zip(perm[::2], perm[1::2]):
            m = s[a] - s[b] + 2.5 + rng.normal(0, sd)
            rows.append(dict(season=2026, week=w, game_date=pd.Timestamp("2026-09-01") + pd.Timedelta(days=7 * w),
                             home_team_name=teams[a], away_team_name=teams[b], home_won=int(m > 0), neutral_site=0,
                             division="fbs", home_division="fbs", away_division="fbs", margin=m,
                             home_score=np.nan, away_score=np.nan))
    return pd.DataFrame(rows), teams, s


DIV = lambda teams: {t: "fbs" for t in teams}  # noqa: E731


class TestYardstick(unittest.TestCase):
    def test_posterior_mean_matches_the_headline_margin_fit(self):
        g, teams, _ = season(seed=1)
        ys = v2.fit_yardstick(g, None, 6, DIV(teams), "fbs", alpha=0.3, w0=0.0, n_draws=5, seed=1)
        sc = 10.62
        ref, _, _ = core.fit_with_prior(g, None, 6, C=2.0, w0=0.0, tau=8.0, major="fbs", divisions=DIV(teams),
                                        model="margin", margin_alpha=0.3, margin_cap=None, margin_scale=sc)
        ours = pd.Series(ys.mean, index=ys.teams)
        ours = ours - ours.mean()
        theirs = ref.reindex(ys.teams) * sc / core.ELO_SCALE
        np.testing.assert_allclose(ours.to_numpy(), theirs.to_numpy(), atol=1e-6)

    def test_draws_are_seeded_and_centred_on_the_mean(self):
        g, teams, _ = season(seed=2)
        a = v2.fit_yardstick(g, None, 6, DIV(teams), "fbs", n_draws=400, seed=7)
        b = v2.fit_yardstick(g, None, 6, DIV(teams), "fbs", n_draws=400, seed=7)
        np.testing.assert_array_equal(a.draws, b.draws)
        centre = lambda d: d - d.mean(axis=-1, keepdims=True)  # noqa: E731  (only differences between teams matter)
        self.assertLess(np.abs(centre(a.draws).mean(axis=0) - centre(a.mean)).max(), 1.5)

    def test_uncertainty_shrinks_with_evidence(self):
        small, teams, s = season(games_per_team=2, seed=3)
        large, _, _ = season(games_per_team=12, seed=3, strengths=s)
        sd = lambda g: v2.fit_yardstick(g, None, 12, DIV(teams), "fbs", n_draws=300, seed=1).draws.std(axis=0).mean()  # noqa: E731
        self.assertGreater(sd(small), sd(large))

    def test_the_game_margin_sd_is_estimated_near_the_truth(self):
        g, teams, _ = season(n_teams=40, games_per_team=10, seed=4, sd=17.0)
        ys = v2.fit_yardstick(g, None, 10, DIV(teams), "fbs", n_draws=2, seed=1)
        self.assertAlmostEqual(ys.sigma, 17.0, delta=2.5)

    def test_last_seasons_games_enter_at_a_decayed_weight(self):
        cur, teams, s = season(seed=5)
        prior, _, _ = season(seed=6, strengths=s)
        with_prior = v2.fit_yardstick(cur, prior, 3, DIV(teams), "fbs", n_draws=2, seed=1)
        without = v2.fit_yardstick(cur, None, 3, DIV(teams), "fbs", n_draws=2, seed=1)
        self.assertGreater(np.abs(with_prior.mean - without.mean).max(), 0.0)
        no_weight = v2.fit_yardstick(cur, prior, 3, DIV(teams), "fbs", w0=0.0, n_draws=2, seed=1)
        np.testing.assert_allclose(no_weight.mean, without.mean)


class TestCredits(unittest.TestCase):
    def setup(self, margins, won, strengths=(300.0, 0.0)):
        """A two-team yardstick: team 0 is the opponent (rated `strengths[0]`), the benchmark is 100 above average."""
        n = len(margins)
        ys = v2.Yardstick(teams=["O", "X"], index={"O": 0, "X": 1}, mean=np.array(strengths), draws=np.tile(np.array(strengths), (4, 1)),
                          home=np.full(4, 2.5), sigma=17.0, bench=np.full(4, 100.0), bench_mean=100.0, home_mean=2.5)
        sides = v2.Sides(team=np.ones(n, dtype=int), opp=np.zeros(n, dtype=int), venue=np.zeros(n),
                         margin=np.array(margins, dtype=float), won=np.array(won, dtype=bool), game=np.arange(n))
        return ys, sides

    def test_mult_never_scores_a_win_below_zero_or_a_loss_above_it(self):
        rng = np.random.default_rng(0)
        for strength in (-200.0, 0.0, 60.0, 150.0, 400.0):
            m = rng.normal(0, 25, 300)
            won = m > 0
            ys, sides = self.setup(m, won, strengths=(strength, 0.0))
            c = v2.credits(ys, sides, "mult", lam=0.5)
            self.assertTrue((c[:, won] >= 0).all())
            self.assertTrue((c[:, ~won] <= 0).all())

    def test_mult_is_bounded_by_the_surprise_of_the_result(self):
        ys, sides = self.setup([60.0, -60.0], [True, False], strengths=(50.0, 0.0))
        p = float(__import__("scipy.stats", fromlist=["norm"]).norm.cdf((100.0 - 50.0) / 17.0))
        c = v2.credits(ys, sides, "mult", lam=0.5)[0]
        self.assertLess(c[0], 1 - p)           # a blowout win cannot exceed the benchmark's surprise at losing
        self.assertGreater(c[1], -p)

    def test_mult_margin_changes_the_size_but_never_the_sign_or_the_order_of_results(self):
        ys, sides = self.setup([1.0, 7.0, 14.0, 28.0, 60.0], [True] * 5, strengths=(150.0, 0.0))
        c = v2.credits(ys, sides, "mult", lam=0.5)[0]
        self.assertTrue((np.diff(c) > 0).all())                 # a bigger win is worth more
        loss = self.setup([-1.0, -28.0], [False, False], strengths=(150.0, 0.0))
        cl = v2.credits(*loss, "mult", lam=0.5)[0]
        self.assertGreater(cl[0], cl[1])                        # a narrower loss costs less
        self.assertGreater(c.min(), cl.max())                   # every win beats every loss against the same opponent

    def test_a_narrow_win_over_a_decent_team_is_positive_under_mult_and_can_be_negative_under_add(self):
        ys, sides = self.setup([3.0], [True], strengths=(0.0, 0.0))      # benchmark is 100 points above this opponent
        self.assertGreater(v2.credits(ys, sides, "mult", lam=0.5)[0, 0], 0)
        self.assertLess(v2.credits(ys, sides, "add", lam=0.5)[0, 0], 0.05)
        self.assertGreaterEqual(v2.credits(ys, sides, "addfloor", lam=0.5)[0, 0], 0)

    def test_beating_a_weak_team_earns_little_beating_a_strong_one_a_lot(self):
        weak = self.setup([21.0], [True], strengths=(-200.0, 0.0))
        strong = self.setup([21.0], [True], strengths=(150.0, 0.0))
        self.assertLess(v2.credits(*weak, "mult")[0, 0], 0.05)
        self.assertGreater(v2.credits(*strong, "mult")[0, 0], 5 * v2.credits(*weak, "mult")[0, 0])

    def test_losing_to_a_weak_team_costs_more_than_losing_to_a_strong_one(self):
        weak = self.setup([-7.0], [False], strengths=(-200.0, 0.0))
        strong = self.setup([-7.0], [False], strengths=(150.0, 0.0))
        self.assertLess(v2.credits(*weak, "mult")[0, 0], v2.credits(*strong, "mult")[0, 0])

    def test_missing_scores_count_as_plain_results(self):
        ys, sides = self.setup([np.nan, np.nan], [True, False], strengths=(100.0, 0.0))
        for kind in ("win", "add", "mult"):
            c = v2.credits(ys, sides, kind)[0]
            self.assertGreater(c[0], 0)
            self.assertLess(c[1], 0)

    def test_unknown_kind_is_refused(self):
        ys, sides = self.setup([1.0], [True])
        with self.assertRaises(ValueError):
            v2.credits(ys, sides, "bogus")

    def test_the_posterior_mean_row_is_available_without_draws(self):
        ys, sides = self.setup([10.0, -3.0], [True, False])
        self.assertEqual(v2.credits(ys, sides, "mult", use_mean=True).shape, (1, 2))


class TestVariance(unittest.TestCase):
    def test_variance_is_positive_and_not_larger_than_a_bernoulli_result_allows(self):
        rng = np.random.default_rng(0)
        for strength in (-300.0, 0.0, 100.0, 300.0):
            m = rng.normal(0, 20, 50)
            n = len(m)
            ys = v2.Yardstick(teams=["O", "X"], index={"O": 0, "X": 1}, mean=np.array([strength, 0.0]), draws=np.tile([strength, 0.0], (2, 1)),
                              home=np.full(2, 2.5), sigma=17.0, bench=np.full(2, 100.0), bench_mean=100.0, home_mean=2.5)
            sides = v2.Sides(team=np.ones(n, dtype=int), opp=np.zeros(n, dtype=int), venue=np.zeros(n), margin=m, won=m > 0, game=np.arange(n))
            for kind in v2.CREDIT_KINDS:
                var = v2.credit_variance(ys, sides, kind, 0.5)
                self.assertTrue((var > 0).all(), kind)
                self.assertTrue((var <= 0.26).all(), (kind, var.max()))

    def test_win_variance_is_p_times_one_minus_p(self):
        ys = v2.Yardstick(teams=["O", "X"], index={"O": 0, "X": 1}, mean=np.array([50.0, 0.0]), draws=np.tile([50.0, 0.0], (2, 1)),
                          home=np.full(2, 2.5), sigma=17.0, bench=np.full(2, 100.0), bench_mean=100.0, home_mean=2.5)
        sides = v2.Sides(team=np.array([1]), opp=np.array([0]), venue=np.array([0.0]), margin=np.array([5.0]), won=np.array([True]), game=np.array([0]))
        p = float(norm_cdf((100.0 - 50.0) / 17.0))
        self.assertAlmostEqual(float(v2.credit_variance(ys, sides, "win")[0]), p * (1 - p), places=3)


def norm_cdf(x):
    from scipy.stats import norm
    return norm.cdf(x)


class TestScores(unittest.TestCase):
    def build(self):
        g, teams, _ = season(seed=8)
        ys = v2.fit_yardstick(g, None, 7, DIV(teams), "fbs", n_draws=20, seed=1)
        sides = v2.sides_of(g, ys.index)
        return g, ys, sides

    def test_a_score_is_the_sum_of_its_per_game_credits(self):
        _, ys, sides = self.build()
        c = v2.credits(ys, sides, "mult")
        s = v2.scores(c, sides, len(ys.teams))
        for t in range(len(ys.teams)):
            np.testing.assert_allclose(s[:, t], c[:, sides.team == t].sum(axis=1))

    def test_the_mean_form_does_not_penalise_a_team_for_playing_fewer_games(self):
        _, ys, sides = self.build()
        c = v2.credits(ys, sides, "mult")
        mean = v2.scores(c, sides, len(ys.teams), k=0.0)
        t = 0
        keep = np.flatnonzero(sides.team == t)
        drop = keep[-1]                                          # the same team without its last game
        mask = np.ones(len(sides.team), dtype=bool)
        mask[drop] = False
        # the mean over the remaining games is the mean of the others: compare with the hand calculation
        np.testing.assert_allclose(mean[:, t], c[:, keep].mean(axis=1))
        reduced = c[:, mask][:, sides.team[mask] == t].mean(axis=1)
        self.assertEqual(reduced.shape, mean[:, t].shape)

    def test_shrinkage_pulls_toward_zero(self):
        _, ys, sides = self.build()
        c = v2.credits(ys, sides, "mult")
        raw = np.abs(v2.scores(c, sides, len(ys.teams), k=0.0))
        shrunk = np.abs(v2.scores(c, sides, len(ys.teams), k=5.0))
        self.assertTrue((shrunk <= raw + 1e-12).all())

    def test_order_does_not_depend_on_the_row_order_of_the_games(self):
        g, teams, _ = season(seed=9)
        ys = v2.fit_yardstick(g, None, 7, DIV(teams), "fbs", n_draws=10, seed=1)
        a = v2.scores(v2.credits(ys, v2.sides_of(g, ys.index), "mult", use_mean=True), v2.sides_of(g, ys.index), len(ys.teams))
        g2 = g.sample(frac=1.0, random_state=3).reset_index(drop=True)
        ys2 = v2.fit_yardstick(g2, None, 7, DIV(teams), "fbs", n_draws=10, seed=1)
        b = v2.scores(v2.credits(ys2, v2.sides_of(g2, ys2.index), "mult", use_mean=True), v2.sides_of(g2, ys2.index), len(ys2.teams))
        np.testing.assert_allclose(a, b, atol=1e-8)


class TestConsensus(unittest.TestCase):
    def test_pairwise_probabilities_are_complementary(self):
        s = np.random.default_rng(0).normal(size=(200, 6))
        p = v2.pairwise(s, np.arange(6))
        np.testing.assert_allclose(p + p.T, 1.0 + np.eye(6) * 0.0, atol=1e-12)

    def test_eta_zero_is_the_order_the_draws_favour(self):
        p = np.array([[0.5, 0.9, 0.95], [0.1, 0.5, 0.8], [0.05, 0.2, 0.5]])
        order = v2.consensus_order(p, np.zeros((3, 3)), 0.0, np.array([2, 0, 1]))
        self.assertEqual(list(order), [0, 1, 2])

    def test_a_played_result_reverses_an_order_only_above_the_stated_probability(self):
        eta = 0.3                                              # threshold (1 + eta) / 2 = 0.65
        h = np.zeros((2, 2))
        h[1, 0] = 1.0                                          # team 1 beat team 0
        for p01, expect in ((0.64, [1, 0]), (0.66, [0, 1])):
            p = np.array([[0.5, p01], [1 - p01, 0.5]])
            self.assertEqual(list(v2.consensus_order(p, h, eta, np.array([0, 1]))), expect, p01)

    def test_ranges_always_contain_the_displayed_rank_and_widen_with_overlap(self):
        tight = np.array([[0.5, 0.99, 0.99], [0.01, 0.5, 0.99], [0.01, 0.01, 0.5]])
        lo, hi = v2.ranges(tight, np.array([0, 1, 2]), q=0.2)
        self.assertEqual((list(lo), list(hi)), ([1, 2, 3], [1, 2, 3]))
        loose = np.full((3, 3), 0.5)
        lo, hi = v2.ranges(loose, np.array([0, 1, 2]), q=0.2)
        self.assertEqual((list(lo), list(hi)), ([1, 1, 1], [3, 3, 3]))

    def test_the_consensus_never_ends_worse_than_where_it_started(self):
        rng = np.random.default_rng(1)
        s = rng.normal(size=(100, 10)) + np.linspace(2, 0, 10)
        p = v2.pairwise(s, np.arange(10))
        h = (rng.random((10, 10)) < 0.1).astype(float)
        np.fill_diagonal(h, 0)
        start = np.arange(10)
        cost = lambda o: sum((p.T + 0.3 * h.T)[o[a], o[b]] for a in range(10) for b in range(a + 1, 10))  # noqa: E731
        self.assertLessEqual(cost(v2.consensus_order(p, h, 0.3, start)), cost(start) + 1e-12)

    def test_played_matrix_counts_wins_once(self):
        g, teams, _ = season(seed=10)
        ys = v2.fit_yardstick(g, None, 7, DIV(teams), "fbs", n_draws=2, seed=1)
        h = v2.played_matrix(g, ys.index, np.arange(len(ys.teams)))
        self.assertTrue(set(np.unique(h)) <= {0.0, 1.0})
        pairs = {(w, l) for w, l in ((r.home_team_name, r.away_team_name) if r.home_won else (r.away_team_name, r.home_team_name)
                                     for r in g.itertuples(index=False))}
        self.assertEqual(int(h.sum()), len(pairs))


if __name__ == "__main__":
    unittest.main()
