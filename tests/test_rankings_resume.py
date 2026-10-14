"""The season-scale resume board: strength of record from win/loss-only ratings, head-to-head inside ties."""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rankings import resume  # noqa: E402


def game(home, away, home_won, neutral=0):
    return {"season": 2026, "week": 1, "game_date": pd.Timestamp("2026-09-05"), "home_team_name": home,
            "away_team_name": away, "home_won": home_won, "neutral_site": neutral, "division": "fbs",
            "home_division": "fbs", "away_division": "fbs", "margin": None, "home_score": None, "away_score": None}


def games(*rows):
    return pd.DataFrame([game(*r) for r in rows])


class TestOrder(unittest.TestCase):
    def test_head_to_head_decides_a_statistical_tie(self):
        sor = {"A": 4.0, "B": 3.9, "C": 1.0}
        var = {"A": 0.8, "B": 0.8, "C": 0.8}
        g = games(("A", "B", 0))                      # B won at A
        self.assertEqual(resume.order_division(["A", "B", "C"], g, sor, var, z=2.0), ["B", "A", "C"])

    def test_head_to_head_never_overrides_a_clear_gap(self):
        sor = {"A": 6.0, "B": 3.0}
        var = {"A": 0.8, "B": 0.8}                    # gap 3.0 > 2 * sqrt(1.6) = 2.53
        g = games(("A", "B", 0))
        self.assertEqual(resume.order_division(["A", "B"], g, sor, var, z=2.0), ["A", "B"])

    def test_tied_teams_that_never_met_keep_the_strength_of_record_order(self):
        sor = {"A": 4.0, "B": 3.9, "C": 3.8}
        var = dict.fromkeys(sor, 0.8)
        g = games(("A", "X", 1), ("B", "Y", 1))
        self.assertEqual(resume.order_division(["A", "B", "C"], g, sor, var, z=2.0), ["A", "B", "C"])

    def test_a_team_never_passes_one_with_a_clearly_different_record_even_through_a_chain(self):
        sor = {"A": 5.0, "B": 4.0, "C": 3.0}
        var = dict.fromkeys(sor, 0.5)                 # sd of a difference = 1.0; z=1.5 -> tied under 1.5
        g = games(("A", "C", 0))                      # C beat A, but A and C are 2.0 apart: not tied
        self.assertEqual(resume.order_division(["A", "B", "C"], g, sor, var, z=1.5), ["A", "B", "C"])

    def test_pairwise_tie_test(self):
        sor = {"A": 4.0, "B": 3.2, "C": 1.0}
        var = dict.fromkeys(sor, 0.5)
        self.assertTrue(resume.tied("A", "B", sor, var, 1.0))
        self.assertFalse(resume.tied("A", "C", sor, var, 1.0))

    def test_a_low_team_that_beat_a_high_one_climbs_only_as_far_as_its_ties_allow(self):
        sor = {"A": 5.0, "B": 4.6, "C": 4.2, "D": 1.0}
        var = dict.fromkeys(sor, 0.5)
        g = games(("D", "A", 1))                      # D beat A, but D is far below everyone
        self.assertEqual(resume.order_division(["A", "B", "C", "D"], g, sor, var, z=1.0)[-1], "D")


class TestBenchmark(unittest.TestCase):
    strengths = pd.Series({"Weak": -200.0, "Strong": 600.0, "X": 0.0, "Y": 0.0, "Z": 0.0})

    def sor(self, g, bench):
        return resume.record_scores(g, self.strengths, 0.0, bench)[0]

    def test_against_an_average_team_every_win_scores_the_same_sign(self):
        g = games(("X", "Weak", 1), ("Y", "Strong", 1))
        s = self.sor(g, 0.0)
        self.assertGreater(s["X"], 0.2)                  # a win over a weak team still earns real credit

    def test_against_a_top_team_beating_the_weak_earns_little_and_the_strong_a_lot(self):
        g = games(("X", "Weak", 1), ("Y", "Strong", 1))
        s = self.sor(g, 500.0)
        self.assertLess(s["X"], 0.1)
        self.assertGreater(s["Y"], 0.6)
        self.assertGreater(s["Y"], 10 * s["X"])

    def test_losing_to_a_weak_team_costs_far_more_than_losing_to_a_strong_one(self):
        g = games(("X", "Weak", 0), ("Y", "Strong", 0))
        s = self.sor(g, 500.0)
        self.assertLess(s["X"], -0.9)
        self.assertGreater(s["Y"], -0.4)

    def test_top_benchmark_is_the_mean_of_the_best_n(self):
        st = pd.Series({"a": 100.0, "b": 300.0, "c": 200.0, "d": -50.0})
        self.assertEqual(resume.top_benchmark(st, ["a", "b", "c", "d"], n=2), 250.0)
        self.assertEqual(resume.top_benchmark(st, [], n=2), 0.0)


class TestMarginCredit(unittest.TestCase):
    st = pd.Series({"Opp": 300.0, "X": 0.0, "Y": 0.0})

    def frame(self, margins):
        g = games(*[("X", "Opp", 1) for _ in margins])
        g["margin"] = margins
        return g

    def test_a_blowout_scores_more_than_a_narrow_win_but_a_win_always_beats_a_loss(self):
        narrow = resume.record_scores(self.frame([1.0]), self.st, 0.0, 300.0, lam=0.5)[0]["X"]
        blowout = resume.record_scores(self.frame([40.0]), self.st, 0.0, 300.0, lam=0.5)[0]["X"]
        loss = resume.record_scores(self.frame([-1.0]), self.st, 0.0, 300.0, lam=0.5)[0]["X"]
        self.assertGreater(blowout, narrow)
        self.assertGreater(narrow, loss)
        self.assertLess(blowout - narrow, 0.5 - 0.0)          # bounded: margin cannot outweigh a win/loss swing

    def test_zero_weight_is_the_win_only_resume(self):
        g = self.frame([40.0])
        a = resume.record_scores(g, self.st, 0.0, 300.0, lam=0.0)[0]["X"]
        b = resume.record_scores(g.assign(margin=1.0), self.st, 0.0, 300.0, lam=0.0)[0]["X"]
        self.assertAlmostEqual(a, b)

    def test_games_without_scores_count_as_wins_and_losses(self):
        g = self.frame([None])
        with_lam = resume.record_scores(g, self.st, 0.0, 300.0, lam=0.5)[0]["X"]
        self.assertAlmostEqual(with_lam, resume.record_scores(g, self.st, 0.0, 300.0, lam=0.0)[0]["X"])

    def test_variance_is_positive_and_not_larger_than_a_coin_flip_game(self):
        v = resume.record_scores(self.frame([10.0]), self.st, 0.0, 300.0, lam=0.5)[1]["X"]
        self.assertTrue(0.0 < v <= 0.25)


class TestRank(unittest.TestCase):
    def table(self, teams, division="fbs"):
        return pd.DataFrame({"team": teams, "division": division})

    def test_unbeaten_beats_one_loss_and_idle_team_is_null(self):
        g = games(("A", "B", 1), ("A", "C", 1), ("B", "C", 1), ("A", "D", 1), ("B", "D", 1), ("C", "D", 1))
        t = self.table(["A", "B", "C", "D", "Idle"])
        st = pd.Series(0.0, index=list("ABCD"))
        r = resume.rank_resume(t, g, st, 0.0, z=0.0)
        self.assertEqual(list(r.iloc[:4]), [1, 2, 3, 4])
        self.assertTrue(pd.isna(r.iloc[4]))

    def test_each_division_is_ranked_on_its_own(self):
        g = games(("A", "B", 1), ("C", "D", 1))
        t = pd.DataFrame({"team": list("ABCD"), "division": ["fbs", "fbs", "fcs", "fcs"]})
        r = resume.rank_resume(t, g, pd.Series(0.0, index=list("ABCD")), 0.0, z=0.0)
        self.assertEqual(list(r), [1, 2, 1, 2])

    def test_opponent_ratings_with_one_outcome_class_count_every_opponent_the_same(self):
        g = games(("A", "B", 1), ("A", "C", 1), ("D", "B", 1))   # only home wins: nothing to fit
        fit_kw = dict(C=2.0, w0=0.12, tau=8.0, major=None, divisions={}, model="margin", margin_alpha=0.3,
                      margin_cap=None, margin_scale=10.62)
        st, ha = resume.opponent_ratings(g, None, 1, fit_kw)
        self.assertEqual((set(st.index), set(st.unique()), ha), ({"A", "B", "C", "D"}, {0.0}, 0.0))

    def test_no_games_means_no_resume(self):
        t = self.table(["A", "B"])
        st = pd.Series(0.0, index=["A", "B"])
        self.assertTrue(resume.rank_resume(t, games(), st, 0.0).isna().all())
        self.assertTrue(resume.rank_resume(t, None, st, 0.0).isna().all())


if __name__ == "__main__":
    unittest.main()
