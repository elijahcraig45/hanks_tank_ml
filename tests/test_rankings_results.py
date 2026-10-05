"""The results-only order published beside the power rating (rankings.results)."""

import os
import sys
import unittest
import unittest.mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "cfb"))

from rankings import build, results, sources  # noqa: E402
from test_rankings import round_robin  # noqa: E402


def game(home, away, home_won, week=1, season=2026, margin=None, neutral=0,
         home_div="fbs", away_div="fbs"):
    return {
        "season": season, "week": week,
        "game_date": pd.Timestamp("2026-09-01") + pd.Timedelta(days=week),
        "home_team_name": home, "away_team_name": away, "home_won": int(home_won),
        "neutral_site": neutral, "home_division": home_div, "away_division": away_div,
        "margin": margin if margin is not None else (7 if home_won else -7),
        "home_score": None, "away_score": None,
    }


def frame(rows):
    return pd.DataFrame(rows)


class MinViolationOrderTests(unittest.TestCase):
    def test_a_transitive_chain_has_no_conflicts(self):
        g = frame([game("A", "B", 1), game("B", "C", 1), game("A", "C", 1, week=2)])
        order = results.min_violation_order(g, ["C", "B", "A"])  # worst possible start
        self.assertEqual(order, ["A", "B", "C"])
        self.assertEqual(results.count_conflicts(order, g), 0)

    def test_a_cycle_costs_exactly_one_conflict(self):
        g = frame([game("A", "B", 1), game("B", "C", 1), game("C", "A", 1)])
        order = results.min_violation_order(g, ["A", "B", "C"])
        self.assertEqual(results.count_conflicts(order, g), 1)

    def test_never_worse_than_the_starting_order(self):
        rng = np.random.default_rng(7)
        teams = [f"T{i}" for i in range(14)]
        rows = []
        for week in range(1, 7):
            shuffled = rng.permutation(teams)
            for h, a in zip(shuffled[0::2], shuffled[1::2]):
                rows.append(game(h, a, rng.random() < 0.5, week=week))
        g = frame(rows)
        start = list(rng.permutation(teams))
        order = results.min_violation_order(g, start)
        self.assertLessEqual(results.count_conflicts(order, g),
                             results.count_conflicts(start, g))
        self.assertCountEqual(order, teams)

    def test_equal_results_keep_the_starting_order(self):
        # Neither pair of games separates X from Y, so they stay where they started.
        g = frame([game("X", "Z", 1), game("Y", "Z", 1)])
        self.assertEqual(results.min_violation_order(g, ["Y", "X", "Z"]), ["Y", "X", "Z"])
        self.assertEqual(results.min_violation_order(g, ["X", "Y", "Z"]), ["X", "Y", "Z"])


class AttachTests(unittest.TestCase):
    def table(self, teams, division="fbs"):
        return pd.DataFrame({"team": teams, "division": division, "rank": range(1, len(teams) + 1)})

    def test_the_head_to_head_winner_is_ordered_above_the_loser(self):
        # The rating may prefer the loser (it is listed first); the results order may not.
        g = frame([game("Tech", "Colorado", 0, week=1)])
        out = results.attach(self.table(["Tech", "Colorado"]), g, {"Tech": "fbs", "Colorado": "fbs"},
                             "fbs", 2.0)
        ranks = dict(zip(out["team"], out["results_rank"]))
        self.assertLess(ranks["Colorado"], ranks["Tech"])

    def test_ranks_are_numbered_within_each_board(self):
        g = frame([
            game("F1", "F2", 1), game("C1", "C2", 1, home_div="fcs", away_div="fcs"),
            game("F1", "C1", 1, week=2, home_div="fbs", away_div="fcs"),
        ])
        t = pd.DataFrame({"team": ["F1", "F2", "C1", "C2"],
                          "division": ["fbs", "fbs", "fcs", "fcs"], "rank": [1, 2, 1, 2]})
        div = {"F1": "fbs", "F2": "fbs", "C1": "fcs", "C2": "fcs"}
        out = results.attach(t, g, div, "fbs", 2.0)
        for _, board in out.groupby("division"):
            self.assertEqual(sorted(board["results_rank"]), [1, 2])

    def test_a_team_with_no_games_gets_nulls_and_preseason_gets_all_nulls(self):
        g = frame([game("A", "B", 1)])
        out = results.attach(self.table(["A", "B", "Idle"]), g, {}, None, 2.0)
        self.assertTrue(pd.isna(out.set_index("team").loc["Idle", "results_rank"]))
        empty = results.attach(self.table(["A", "B"]), g.iloc[0:0], {}, None, 2.0)
        self.assertTrue(empty["results_rank"].isna().all())

    def test_conflicts_are_counted_on_both_teams(self):
        g = frame([game("A", "B", 1), game("B", "C", 1), game("C", "A", 1)])
        out = results.attach(self.table(["A", "B", "C"]), g, {}, None, 2.0).set_index("team")
        self.assertEqual(int(out["results_conflicts"].sum()), 2)  # one game, two teams


class BuildTests(unittest.TestCase):
    def games(self, season=2026):
        g = round_robin(["F1", "F2", "F3", "F4"], season=season, repeats=1)
        g["home_division"] = "fbs"
        g["away_division"] = "fbs"
        g["margin"] = np.where(g["home_won"] == 1, 7.0, -7.0)
        g["home_score"] = None
        g["away_score"] = None
        return g

    def test_only_college_publishes_the_results_order(self):
        self.assertTrue(sources.SPORTS["cfb"].extra_boards)
        self.assertFalse(sources.SPORTS["nfl"].extra_boards)
        self.assertFalse(sources.SPORTS["mlb"].extra_boards)

    def test_the_college_board_carries_the_columns(self):
        table, meta = build.build_board("cfb", 2026, n_boot=5, use_prior=False,
                                        games=self.games(), with_fpi=False,
                                        with_rationale=False)
        self.assertTrue(meta["has_results"])
        self.assertEqual(sorted(table["results_rank"]), [1, 2, 3, 4])

    def test_other_sports_do_not(self):
        g = round_robin(["A", "B", "C", "D"], season=2026, repeats=1)
        table, meta = build.build_board("nfl", 2026, n_boot=5, use_prior=False, games=g,
                                        with_fpi=False, with_rationale=False)
        self.assertFalse(meta["has_results"])
        self.assertNotIn("results_rank", table.columns)

    def test_the_four_boards_are_numbered_within_the_board_with_nulls_for_idle_teams(self):
        g = self.games()
        g = g[g["week"] <= 3]  # inside the membership grace period, so last season's teams stay boarded
        # Idle plays nobody this season; it only exists in last season's games.
        prior = round_robin(["F1", "F2", "F3", "F4", "Idle"], season=2025, repeats=1)
        prior["home_division"] = "fbs"
        prior["away_division"] = "fbs"
        prior["margin"] = np.where(prior["home_won"] == 1, 7.0, -7.0)
        prior["home_score"] = None
        prior["away_score"] = None
        table, meta = build.build_board("cfb", 2026, n_boot=5, games=pd.concat([prior, g]),
                                        with_fpi=False, with_rationale=False)
        self.assertEqual(meta["extra_boards"],
                         ["results_rank", "season_rank", "forecast_rank", "resume_rank"])
        t = table.set_index("team")
        played = ["F1", "F2", "F3", "F4"]
        for col in ("results_rank", "season_rank"):
            self.assertEqual(sorted(t.loc[played, col]), [1, 2, 3, 4], col)
        for col in ("results_rank", "season_rank", "resume_rank"):
            self.assertTrue(pd.isna(t.loc["Idle", col]), col)
        # Last season's evidence ranks it too, so the forecast board covers all five.
        self.assertEqual(sorted(t["forecast_rank"]), [1, 2, 3, 4, 5])

    def test_preseason_nulls_everything_but_the_forecast(self):
        prior = self.games(season=2025)
        table, meta = build.build_board("cfb", 2026, n_boot=5, games=prior, with_fpi=False,
                                        with_rationale=False)
        self.assertTrue(meta["is_preseason"])
        for col in ("results_rank", "season_rank", "resume_rank"):
            self.assertTrue(table[col].isna().all(), col)
        self.assertEqual(sorted(table["forecast_rank"]), [1, 2, 3, 4])
        self.assertEqual(meta["extra_boards"], ["forecast_rank"])
        self.assertFalse(meta["has_results"])

    def test_forecast_uses_its_own_weights_not_the_headline_ones(self):
        calls = []
        real = build.core.fit_with_prior

        def spy(current, prior, week, **kw):
            calls.append((prior is None, kw["w0"], kw["tau"]))
            return real(current, prior, week, **kw)

        prior = self.games(season=2025)
        with unittest.mock.patch.object(build.core, "fit_with_prior", spy):
            build.build_board("cfb", 2026, n_boot=0, games=pd.concat([prior, self.games()]),
                              with_fpi=False, with_rationale=False)
        self.assertIn((False, 0.12, 8.0), calls)   # headline
        self.assertIn((False, 0.25, 16.0), calls)  # forecast: the previous headline weights
        self.assertIn((True, 0.12, 8.0), calls)    # season only: no prior at all

    def test_resume_rank_follows_sor(self):
        g = self.games()
        prior = self.games(season=2025)
        table, _ = build.build_board("cfb", 2026, n_boot=5, games=pd.concat([prior, g]),
                                     with_fpi=False, with_rationale=False)
        t = table.dropna(subset=["resume_rank"]).sort_values("resume_rank")
        self.assertTrue(t["sor"].is_monotonic_decreasing)
        self.assertEqual(int(t["resume_rank"].iloc[0]), 1)

    def test_a_failed_season_fit_leaves_that_board_null_and_the_rest_intact(self):
        real = build.core.fit_with_prior

        def flaky(current, prior, week, **kw):
            if prior is None:
                raise ValueError("degenerate")
            return real(current, prior, week, **kw)

        prior = self.games(season=2025)
        with unittest.mock.patch.object(build.core, "fit_with_prior", flaky):
            table, meta = build.build_board("cfb", 2026, n_boot=0,
                                            games=pd.concat([prior, self.games()]),
                                            with_fpi=False, with_rationale=False)
        self.assertTrue(table["season_rank"].isna().all())
        self.assertNotIn("season_rank", meta["extra_boards"])
        self.assertEqual(sorted(table["results_rank"]), [1, 2, 3, 4])
        self.assertEqual(sorted(table["forecast_rank"]), [1, 2, 3, 4])

    def test_the_headline_weights_are_the_ones_the_comment_describes(self):
        spec = sources.SPORTS["cfb"]
        self.assertEqual((spec.prior_w0, spec.prior_tau), (0.12, 8.0))
        self.assertEqual((spec.forecast_w0, spec.forecast_tau), (0.25, 16.0))


if __name__ == "__main__":
    unittest.main()
