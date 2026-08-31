"""Property tests for the sport-neutral power-rankings engine.

These pin the claims the engine actually makes, rather than snapshotting numbers:
order-independence (the whole reason this is Bradley-Terry and not Elo), home-field
recovery, division separation, and season-aware division handling.
"""

import json
import os
import sys
import unittest
import unittest.mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rankings import core, sources  # noqa: E402
from rankings.build import build_board  # noqa: E402


def round_robin(teams, season=2025, neutral=0, strength=None, repeats=2):
    """Every team plays every other, home and away. `strength` decides who wins."""
    strength = strength or {t: i for i, t in enumerate(teams)}
    rows = []
    week = 1
    for _ in range(repeats):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                rows.append({
                    "season": season,
                    "week": week,
                    "game_date": pd.Timestamp("2025-09-01") + pd.Timedelta(days=week),
                    "home_team_name": home,
                    "away_team_name": away,
                    "home_won": int(strength[home] > strength[away]),
                    "neutral_site": neutral,
                })
                week += 1
    return pd.DataFrame(rows)


class CoreTests(unittest.TestCase):
    def test_ratings_are_order_independent(self):
        """The premise that separates this from Elo: shuffling must change nothing."""
        games = round_robin(["A", "B", "C", "D"])
        first, home_a, _ = core.fit_ratings(games, C=1.0)

        shuffled = games.sample(frac=1.0, random_state=7).reset_index(drop=True)
        second, home_b, _ = core.fit_ratings(shuffled, C=1.0)

        pd.testing.assert_series_equal(
            first.sort_index(), second.sort_index(), atol=1e-6
        )
        self.assertAlmostEqual(home_a, home_b, places=6)

    def test_stronger_team_rates_higher(self):
        games = round_robin(["weak", "mid", "strong"])
        strengths, _, _ = core.fit_ratings(games, C=1.0)
        self.assertEqual(list(strengths.index), ["strong", "mid", "weak"])

    def test_home_field_is_recovered_and_absent_at_neutral_sites(self):
        teams = ["A", "B", "C", "D"]
        # Home team wins most of the time (not all — a single-class target has no
        # likelihood to maximize), so home field must come out positive.
        home_heavy = round_robin(teams)
        home_heavy["home_won"] = 1
        home_heavy.loc[home_heavy.index[::7], "home_won"] = 0
        _, home_adv, _ = core.fit_ratings(home_heavy, C=1.0)
        self.assertGreater(home_adv, 0)

        # The identical results at neutral sites carry no home-field information:
        # the home column is all zeros, so its coefficient must be exactly zero.
        neutral = home_heavy.copy()
        neutral["neutral_site"] = 1
        _, neutral_adv, _ = core.fit_ratings(neutral, C=1.0)
        self.assertAlmostEqual(neutral_adv, 0.0, places=6)

    def test_division_term_separates_weakly_connected_populations(self):
        """Two pools joined by a few games the majors always win should stay apart."""
        major = round_robin(["M1", "M2", "M3"], strength={"M1": 3, "M2": 2, "M3": 1})
        minor = round_robin(["m1", "m2", "m3"], strength={"m1": 3, "m2": 2, "m3": 1})
        major["home_division"] = "big"
        major["away_division"] = "big"
        minor["home_division"] = "small"
        minor["away_division"] = "small"

        bridge = pd.DataFrame([{
            "season": 2025, "week": 99,
            "game_date": pd.Timestamp("2025-11-01"),
            "home_team_name": "M3", "away_team_name": "m1",
            "home_won": 1, "neutral_site": 0,
            "home_division": "big", "away_division": "small",
        }])
        games = pd.concat([major, minor, bridge], ignore_index=True)

        strengths, _, gap = core.fit_ratings(games, C=1.0, major="big")
        self.assertGreater(gap, 0, "big division should rate above small")
        # Every major outranks every minor despite near-identical internal records.
        worst_major = min(strengths[t] for t in ("M1", "M2", "M3"))
        best_minor = max(strengths[t] for t in ("m1", "m2", "m3"))
        self.assertGreater(worst_major, best_minor)

    def test_team_divisions_prefers_per_side_columns(self):
        games = pd.DataFrame([{
            "season": 2026, "week": 1, "game_date": pd.Timestamp("2026-09-01"),
            "home_team_name": "Riser", "away_team_name": "Stayer",
            "home_won": 1, "neutral_site": 0,
            "home_division": "fbs", "away_division": "fcs",
        }])
        self.assertEqual(
            core.team_divisions(games), {"Riser": "fbs", "Stayer": "fcs"}
        )

    def test_preseason_uses_prior_undecayed(self):
        prior = round_robin(["A", "B", "C"], season=2025)
        empty = prior.iloc[0:0]
        strengths, _, _ = core.fit_with_prior(empty, prior, week=0, C=1.0)
        self.assertEqual(list(strengths.index), ["C", "B", "A"])

    def test_prior_weight_decays(self):
        self.assertAlmostEqual(core.prior_weight(0, 1.0, 8.0), 1.0)
        self.assertLess(core.prior_weight(8, 1.0, 8.0), core.prior_weight(4, 1.0, 8.0))
        self.assertGreater(core.prior_weight(8, 1.0, 8.0), 0.0)

    def test_win_prob_is_symmetric_and_centred(self):
        self.assertAlmostEqual(core.win_prob(100.0, 100.0), 0.5)
        self.assertAlmostEqual(
            core.win_prob(200.0, 0.0) + core.win_prob(0.0, 200.0), 1.0, places=9
        )

    def test_validate_rejects_incomplete_frames(self):
        with self.assertRaises(ValueError):
            core.validate(pd.DataFrame({"home_team_name": ["A"]}))


class BoardTests(unittest.TestCase):
    def _two_division_games(self):
        major = round_robin(["M1", "M2"], strength={"M1": 2, "M2": 1})
        minor = round_robin(["m1", "m2"], strength={"m1": 2, "m2": 1})
        major["home_division"] = major["away_division"] = "fbs"
        minor["home_division"] = minor["away_division"] = "fcs"
        bridge = pd.DataFrame([{
            "season": 2025, "week": 99, "game_date": pd.Timestamp("2025-11-01"),
            "home_team_name": "M2", "away_team_name": "m1",
            "home_won": 1, "neutral_site": 0,
            "home_division": "fbs", "away_division": "fcs",
        }])
        return pd.concat([major, minor, bridge], ignore_index=True)

    def test_board_ranks_within_division_but_keeps_a_global_rank(self):
        games = self._two_division_games()
        table, meta = build_board(
            "cfb", 2025, n_boot=5, use_prior=False, games=games
        )

        for division in ("fbs", "fcs"):
            board = table[table["division"] == division].sort_values("rank")
            self.assertEqual(
                list(board["rank"]), list(range(1, len(board) + 1)),
                f"{division} board should be numbered from 1",
            )

        # The joint fit is what makes the two ladders comparable, so the top FCS team
        # must still sit below every FBS team on the overall list.
        top_fcs = table[table["division"] == "fcs"]["overall_rank"].min()
        worst_fbs = table[table["division"] == "fbs"]["overall_rank"].max()
        self.assertGreater(top_fcs, worst_fbs)
        self.assertFalse(meta["is_preseason"])

    def test_preseason_board_is_flagged_and_uses_last_seasons_record(self):
        prior = round_robin(["A", "B", "C"], season=2025)
        table, meta = build_board("nfl", 2026, n_boot=5, games=prior)

        self.assertTrue(meta["is_preseason"])
        self.assertEqual(meta["record_season"], 2025)
        self.assertEqual(meta["games_current"], 0)
        # Records come from 2025, so they are not all 0-0.
        self.assertTrue(any(r != "0-0" for r in table["record"]))

    def test_last_row_has_no_next_team(self):
        games = round_robin(["A", "B", "C"], season=2025)
        table, _ = build_board("nfl", 2025, n_boot=5, use_prior=False, games=games)
        last = table.sort_values("rank").iloc[-1]
        self.assertTrue(pd.isna(last["gap_to_next"]))


class FranchiseIdentityTests(unittest.TestCase):
    """A club that gets renamed or relocated must stay ONE team.

    Keyed on the display name or the era's abbreviation instead, its earlier games
    belong to a different entity: last season cannot inform this season's rating and a
    phantom 0-0 row appears on the board. This happened for real — Oakland Athletics
    became Athletics in 2025, and the 2025 MLB board came out with 31 teams.
    """

    def test_mlb_rename_collapses_to_the_current_name(self):
        payload = {
            "dates": [
                {
                    "date": "2024-05-01",
                    "games": [{
                        "status": {"detailedState": "Final"},
                        "teams": {
                            "home": {"team": {"id": 133, "name": "Oakland Athletics"},
                                     "score": 5},
                            "away": {"team": {"id": 111, "name": "Boston Red Sox"},
                                     "score": 2},
                        },
                    }],
                },
                {
                    "date": "2025-05-01",
                    "games": [{
                        "status": {"detailedState": "Final"},
                        "teams": {
                            # Same franchise id, new name.
                            "home": {"team": {"id": 133, "name": "Athletics"}, "score": 1},
                            "away": {"team": {"id": 111, "name": "Boston Red Sox"},
                                     "score": 4},
                        },
                    }],
                },
            ]
        }

        with unittest.mock.patch.object(
            sources, "_curl", return_value=json.dumps(payload).encode()
        ), unittest.mock.patch.object(sources.Path, "exists", return_value=False), \
             unittest.mock.patch.object(sources.Path, "write_text"):
            games = sources.load_mlb((2024, 2025))

        teams = set(games["home_team_name"]) | set(games["away_team_name"])
        self.assertIn("Athletics", teams)
        self.assertNotIn(
            "Oakland Athletics", teams,
            "the old name must resolve to the current one, not sit alongside it",
        )

    def test_nfl_relocation_aliases_are_current_codes(self):
        # One entry per relocation, mapping the historical code to today's.
        self.assertEqual(sources.NFL_FRANCHISE_ALIASES["OAK"], "LV")
        self.assertEqual(sources.NFL_FRANCHISE_ALIASES["SD"], "LAC")
        self.assertEqual(sources.NFL_FRANCHISE_ALIASES["STL"], "LA")
        # A current code must never itself be aliased away.
        for current in ("LV", "LAC", "LA", "WAS"):
            self.assertNotIn(current, sources.NFL_FRANCHISE_ALIASES)


if __name__ == "__main__":
    unittest.main()
