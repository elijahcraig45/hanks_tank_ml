"""Tests for the power-rankings rationale (rankings.explain).

The rationale makes numeric claims, so the tests check the numbers against the fit
itself rather than snapshotting text: the two rating parts must add up to the rating,
a game's leave-out contribution must equal an actual refit without it, and the summary
line must only say what the numbers support.
"""

import json
import os
import re
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rankings import build, core, explain  # noqa: E402
from test_rankings import round_robin  # noqa: E402

TEAMS = ["A", "B", "C", "D", "E"]
STRENGTH = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


def scored(season, seed, noise=10.0, repeats=1):
    """Round robin with noisy margins, so some games go against the strength order."""
    rng = np.random.default_rng(seed)
    g = round_robin(TEAMS, season=season, strength=STRENGTH, repeats=repeats)
    true = g["home_team_name"].map(STRENGTH) - g["away_team_name"].map(STRENGTH)
    margin = np.round(true * 4 + 2 + rng.normal(0, noise, len(g)))
    margin[margin == 0] = 1
    g["margin"] = margin
    g["home_won"] = (margin > 0).astype(int)
    g["home_score"] = np.where(margin > 0, 20 + margin, 20).astype(int)
    g["away_score"] = np.where(margin > 0, 20, 20 - margin).astype(int)
    return g


def fit_kw(model, **over):
    kw = dict(C=1.0, w0=0.5, tau=8.0, major=None, divisions={}, model=model,
              margin_alpha=2.0, margin_cap=21.0, margin_scale=5.0, blend=0.5)
    kw.update(over)
    return kw


class DecompositionTests(unittest.TestCase):
    def setUp(self):
        self.prior = scored(2025, seed=1)
        self.current = scored(2026, seed=2).iloc[:14].copy()
        self.week = int(self.current["week"].max())

    def _check_parts_sum(self, model, tol):
        kw = fit_kw(model)
        published, _, _ = core.fit_with_prior(self.current, self.prior, self.week, **kw)
        exp = explain.Explainer(self.current, self.prior, self.week, **kw)
        prior_part, current_part = exp.parts()
        total = (prior_part + current_part).reindex(published.index)
        np.testing.assert_allclose(total.to_numpy(), published.to_numpy(), atol=tol)
        # Both seasons genuinely contribute: this is not a trivial all-in-one split.
        self.assertGreater(prior_part.abs().max(), 1.0)
        self.assertGreater(current_part.abs().max(), 1.0)

    def test_margin_parts_sum_to_the_rating(self):
        self._check_parts_sum("margin", 1e-8)

    def test_win_loss_parts_sum_to_the_rating(self):
        self._check_parts_sum("bt", 1e-6)

    def test_blend_parts_sum_to_the_rating(self):
        self._check_parts_sum("blend", 1e-6)

    def test_preseason_rating_is_entirely_prior(self):
        kw = fit_kw("margin")
        exp = explain.Explainer(self.current.iloc[0:0], self.prior, 0, **kw)
        prior_part, current_part = exp.parts()
        self.assertTrue((current_part.abs() < 1e-9).all())
        np.testing.assert_allclose(prior_part.to_numpy(), exp.ratings.to_numpy(), atol=1e-8)


class LeaveOutTests(unittest.TestCase):
    def setUp(self):
        self.prior = scored(2025, seed=3)
        self.current = scored(2026, seed=4, repeats=2)
        self.week = int(self.current["week"].max())

    def _refit_delta(self, kw, drop_rows):
        full, _, _ = core.fit_with_prior(self.current, self.prior, self.week, **kw)
        keep = self.current.drop(self.current.index[drop_rows])
        refit, _, _ = core.fit_with_prior(keep, self.prior, self.week, **kw)
        return full - refit.reindex(full.index)

    def _closed(self, kw, drop_rows):
        exp = explain.Explainer(self.current, self.prior, self.week, **kw)
        offset = len(self.prior)
        return exp.leave_out([offset + r for r in drop_rows])

    def test_margin_leave_one_out_equals_a_refit(self):
        kw = fit_kw("margin")
        for row in (0, 5, 17, 33):
            actual = self._refit_delta(kw, [row])
            closed = self._closed(kw, [row]).reindex(actual.index)
            np.testing.assert_allclose(closed.to_numpy(), actual.to_numpy(), atol=1e-8)

    def test_margin_leave_block_out_equals_a_refit(self):
        kw = fit_kw("margin")
        rows = [1, 9, 22]
        actual = self._refit_delta(kw, rows)
        closed = self._closed(kw, rows).reindex(actual.index)
        np.testing.assert_allclose(closed.to_numpy(), actual.to_numpy(), atol=1e-8)

    def test_win_loss_leave_out_matches_a_refit_closely(self):
        # One Newton step from the optimum, so approximate; on real MLB series it is
        # within 0.012 rating points of the refit (see the PR). Here the games are few
        # and lopsided, so the tolerance is looser but still well below the effect.
        kw = fit_kw("bt", C=0.5)
        for rows in ([0], [3, 4], [10, 20, 30]):
            actual = self._refit_delta(kw, rows)
            closed = self._closed(kw, rows).reindex(actual.index)
            self.assertLess(np.abs(closed - actual).max(), 0.1 * max(actual.abs().max(), 1.0))


class LogisticPolishTests(unittest.TestCase):
    def test_bt_coef_is_the_stationary_point(self):
        from scipy.special import expit

        games = scored(2026, seed=5, repeats=3)
        teams = sorted(set(games["home_team_name"]) | set(games["away_team_name"]))
        x, y = core._design(games, teams, {}, None)
        w = np.linspace(0.3, 1.0, len(y))
        beta = core.bt_coef(x, y, 0.2, w)
        grad = x.T @ (w * (y - expit(x @ beta))) - beta / 0.2
        self.assertLess(np.abs(grad).max(), 1e-8)


class SummaryTests(unittest.TestCase):
    BANNED = ("poll", "ap", "coaches", "dominant", "elite", "best team", "clearly",
              "impressive", "juggernaut")

    def _board(self, sport="nfl", remaining=None):
        prior = scored(2025, seed=6)
        current = scored(2026, seed=7).iloc[:12].copy()
        games = pd.concat([prior, current], ignore_index=True)
        return build.build_board(sport, 2026, n_boot=30, games=games, with_fpi=False,
                                 remaining=remaining)

    def test_board_carries_the_rationale_columns(self):
        table, meta = self._board()
        self.assertTrue(meta["has_rationale"])
        for col in ("summary", "why_json", "games_json", "rating_from_prior",
                    "rating_from_current", "sched_strength", "sched_rank", "prior_share"):
            self.assertIn(col, table.columns)
        # The displayed parts add up to the displayed rating.
        np.testing.assert_allclose(
            (table["rating_from_prior"] + table["rating_from_current"]).to_numpy(),
            table["rating"].to_numpy(), atol=0.051)
        self.assertLess(meta["rationale_check"]["decomposition_max_abs_error"], 1e-6)

    def test_summary_is_deterministic_and_plain(self):
        first, _ = self._board()
        second, _ = self._board()
        self.assertEqual(list(first["summary"]), list(second["summary"]))
        for text, record, rank in zip(first["summary"], first["record"], first["rank"]):
            self.assertTrue(text.startswith(f"#{rank} because: {record}"), text)
            for word in self.BANNED:
                self.assertIsNone(re.search(rf"\b{word}\b", text.lower()), text)

    def test_best_win_is_a_win_and_worst_loss_a_loss(self):
        table, _ = self._board()
        for r in table.itertuples(index=False):
            games = json.loads(r.games_json)
            why = json.loads(r.why_json)
            for i in why["best"]:
                self.assertTrue(games[i]["won"])
                self.assertGreater(games[i]["contrib"], 0)
            for i in why["worst"]:
                self.assertFalse(games[i]["won"])
                self.assertLess(games[i]["contrib"], 0)

    def test_adjacent_pairs_split_the_gap_exactly(self):
        table, _ = self._board()
        ordered = table.sort_values("rank")
        for r in ordered.itertuples(index=False):
            if r.vs_next_json is None or pd.isna(r.vs_next_json):
                continue
            pair = json.loads(r.vs_next_json)
            self.assertAlmostEqual(pair["gap_from_prior"] + pair["gap_from_current"],
                                   pair["gap"], delta=0.15)
            self.assertEqual(pair["tied"], pair["p_order"] < explain.TIE_ORDER_P)
            if pair["tied"]:
                self.assertIn("Statistically tied", r.summary)

    def test_remaining_schedule_strength_uses_unplayed_games(self):
        remaining = pd.DataFrame([
            {"season": 2026, "week": 20, "game_date": None, "home_team_name": "E",
             "away_team_name": "A", "neutral_site": 0},
            {"season": 2026, "week": 20, "game_date": None, "home_team_name": "B",
             "away_team_name": "E", "neutral_site": 0},
        ])
        table, _ = self._board(remaining=remaining)
        by_team = table.set_index("team")
        self.assertEqual(int(by_team.loc["E", "sched_remaining_games"]), 2)
        self.assertAlmostEqual(
            by_team.loc["E", "sched_remaining"],
            (by_team.loc["A", "rating"] + by_team.loc["B", "rating"]) / 2, delta=0.2)
        self.assertEqual(int(by_team.loc["C", "sched_remaining_games"]), 0)
        self.assertTrue(pd.isna(by_team.loc["C", "sched_remaining"]))

    def test_mlb_summary_quotes_wins_not_run_differential(self):
        table, _ = self._board(sport="mlb")
        for text in table["summary"]:
            self.assertNotIn("margin", text)
            self.assertIn("wins", text)

    def test_preseason_summary_says_it_is_last_season(self):
        prior = scored(2025, seed=8)
        table, _ = build.build_board("nfl", 2026, n_boot=5, games=prior, with_fpi=False)
        for text in table["summary"]:
            self.assertIn("no 2026 games yet", text)


class HelperTests(unittest.TestCase):
    def test_prior_share_needs_both_parts_to_agree(self):
        self.assertAlmostEqual(explain.prior_share(30.0, 70.0), 0.3)
        self.assertAlmostEqual(explain.prior_share(-30.0, -70.0), 0.3)
        self.assertIsNone(explain.prior_share(40.0, -30.0))
        self.assertIsNone(explain.prior_share(0.0, 0.0))

    def test_pair_reports_head_to_head_and_common_opponents(self):
        g = lambda opp, won, pf, pa: {"opp": opp, "won": won, "pf": pf, "pa": pa,  # noqa: E731
                                      "margin": pf - pa, "over": 0.0, "site": "H"}
        a = {"team": "A", "rank": 1, "rating": 100.0, "rating_from_prior": 30.0,
             "rating_from_current": 70.0,
             "games": [g("B", True, 21, 14), g("C", True, 30, 3)]}
        b = {"team": "B", "rank": 2, "rating": 80.0, "rating_from_prior": 50.0,
             "rating_from_current": 30.0,
             "games": [g("A", False, 14, 21), g("C", False, 10, 13)]}
        pair = explain.pair_explanation(a, b, sport="nfl", points_per_elo=0.03,
                                        p_order=0.9)
        self.assertEqual((pair["h2h"]["w"], pair["h2h"]["l"]), (1, 0))
        self.assertEqual([c["opp"] for c in pair["common"]], ["C"])
        self.assertEqual(pair["gap_from_prior"], -20.0)
        self.assertEqual(pair["gap_from_current"], 40.0)
        self.assertFalse(pair["tied"])
        self.assertIn("head to head A went 1-0", pair["text"])

    def test_bootstrap_default_return_type_is_unchanged(self):
        games = scored(2026, seed=9)
        out = core.bootstrap_ranks(games, None, 0, n_boot=3, C=1.0)
        self.assertIsInstance(out, pd.DataFrame)
        ranks, draws = core.bootstrap_ranks(games, None, 0, n_boot=3, C=1.0,
                                            return_draws=True)
        self.assertEqual(len(draws), 3)
        self.assertEqual(set(draws.columns), set(TEAMS))

    def test_json_columns_are_written_as_text(self):
        for col in explain.TEXT_COLUMNS:
            self.assertIn(col, build.TEXT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
