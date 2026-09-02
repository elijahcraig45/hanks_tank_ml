"""Tests for the CollegeFootballData transforms.

Offline by design: fixtures in, frames out, no network and no BigQuery. These transforms
are where the bugs live, and two of the cases below are regressions for bugs that were
caught only because a dry run printed something odd.

CI runs `python -m unittest discover -s tests`, so these are unittest, not pytest.
"""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stats import cfb_advanced, cfbd  # noqa: E402


class TestSnake(unittest.TestCase):
    """Acronyms are the whole difficulty."""

    def test_all_caps_stat_types_stay_whole(self):
        # Regression: a naive "underscore before every capital" turned YDS into y_d_s,
        # so every pivoted player column was unusable and the leaderboards silently
        # fell back to the poorer ESPN feed.
        self.assertEqual(cfbd.snake("YDS"), "yds")
        self.assertEqual(cfbd.snake("TD"), "td")
        self.assertEqual(cfbd.snake("PCT"), "pct")

    def test_trailing_acronym_is_not_split(self):
        self.assertEqual(cfbd.snake("totalPPA"), "total_ppa")
        self.assertEqual(cfbd.snake("QBRating"), "qb_rating")

    def test_camel_case_splits_on_word_boundaries(self):
        self.assertEqual(cfbd.snake("passingYards"), "passing_yards")
        self.assertEqual(cfbd.snake("openFieldYardsTotal"), "open_field_yards_total")

    def test_dots_and_spaces_become_underscores(self):
        self.assertEqual(cfbd.snake("fieldPosition.averageStart"),
                         "field_position_average_start")
        self.assertEqual(cfbd.snake("QB HUR"), "qb_hur")
        self.assertEqual(cfbd.snake("IN 20"), "in_20")

    def test_no_output_has_a_dot_or_a_capital(self):
        for name in ("offense.passingPlays.successRate", "havoc.frontSeven", "IN 20"):
            out = cfbd.snake(name)
            self.assertNotIn(".", out)
            self.assertEqual(out, out.lower())


class TestFlatten(unittest.TestCase):
    def test_nested_blocks_become_prefixed_columns(self):
        record = {"ppa": 0.3, "havoc": {"frontSeven": 0.1, "total": 0.2},
                  "fieldPosition": {"averageStart": 71.2}}
        out = cfbd.flatten(record)
        self.assertEqual(out["ppa"], 0.3)
        self.assertEqual(out["havoc_front_seven"], 0.1)
        self.assertEqual(out["field_position_average_start"], 71.2)

    def test_defense_maps_to_the_opp_prefix(self):
        # The `opp_` convention already exists in cfb_season.team_season_stats for
        # ESPN's opponent split; CFBD's defense block means the same thing.
        record = {"offense": {"ppa": 0.4}, "defense": {"ppa": 0.1}}
        out = cfbd.split_off_def(record)
        self.assertEqual(out["ppa"], 0.4)
        self.assertEqual(out["opp_ppa"], 0.1)

    def test_absent_nested_block_yields_no_column_rather_than_a_wrong_one(self):
        out = cfbd.split_off_def({"offense": {"ppa": 0.4}})
        self.assertEqual(out["ppa"], 0.4)
        self.assertNotIn("opp_ppa", out)

    def test_both_spellings_of_the_upstream_typo_are_accepted(self):
        # CFBD spells it totalOpportunies. If they fix it, the column must not vanish.
        a = cfbd.split_off_def({"offense": {"totalOpportunies": 12}})
        b = cfbd.split_off_def({"offense": {"totalOpportunities": 12}})
        self.assertEqual(a["total_opportunities"], 12)
        self.assertEqual(b["total_opportunities"], 12)


def player_rows():
    """Long-form records in the shape /stats/player/season returns."""
    return [
        {"season": 2025, "playerId": "1", "player": "A Passer", "position": "QB",
         "team": "Ohio State", "conference": "Big Ten",
         "category": "passing", "statType": "YDS", "stat": "3000"},
        {"season": 2025, "playerId": "1", "player": "A Passer", "position": "QB",
         "team": "Ohio State", "conference": "Big Ten",
         "category": "passing", "statType": "TD", "stat": "30"},
        {"season": 2025, "playerId": "2", "player": "A Rusher", "position": "RB",
         "team": "Indiana", "conference": "Big Ten",
         "category": "rushing", "statType": "YDS", "stat": "1200"},
    ]


class TestPivot(unittest.TestCase):
    def test_one_row_per_player_with_category_prefixed_columns(self):
        out = cfbd.pivot_player_season(player_rows())
        self.assertEqual(len(out), 2)
        self.assertIn("passing_yds", out.columns)
        self.assertIn("rushing_yds", out.columns)
        passer = out[out.player_id == "1"].iloc[0]
        self.assertEqual(passer["passing_yds"], 3000)
        self.assertEqual(passer["passing_td"], 30)

    def test_category_prefix_prevents_collisions(self):
        # Both categories publish a YDS statType; they must not overwrite each other.
        out = cfbd.pivot_player_season(player_rows())
        rusher = out[out.player_id == "2"].iloc[0]
        self.assertTrue(pd.isna(rusher["passing_yds"]))
        self.assertEqual(rusher["rushing_yds"], 1200)

    def test_a_transfer_resolves_to_the_last_team_not_a_sum(self):
        rows = player_rows() + [
            {"season": 2025, "playerId": "1", "player": "A Passer", "position": "QB",
             "team": "Oregon", "conference": "Big Ten",
             "category": "passing", "statType": "YDS", "stat": "500"},
        ]
        out = cfbd.pivot_player_season(rows)
        passer = out[out.player_id == "1"].iloc[0]
        # Summing would give 3500, which is wrong for every rate stat.
        self.assertEqual(passer["passing_yds"], 500)
        self.assertEqual(passer["team"], "Oregon")

    def test_non_numeric_stat_becomes_null_not_an_exception(self):
        rows = player_rows() + [
            {"season": 2025, "playerId": "3", "player": "Odd", "position": "K",
             "team": "X", "conference": "Y",
             "category": "kicking", "statType": "PCT", "stat": "-"},
        ]
        out = cfbd.pivot_player_season(rows)
        self.assertTrue(pd.isna(out[out.player_id == "3"].iloc[0]["kicking_pct"]))

    def test_missing_columns_raise_rather_than_producing_a_wrong_frame(self):
        with self.assertRaises(ValueError):
            cfbd.pivot_player_season([{"season": 2025, "playerId": "1"}])


class TestLeaders(unittest.TestCase):
    def players(self):
        return pd.DataFrame([
            # A full-time starter.
            {"player_id": "1", "player_name": "Starter", "position": "QB",
             "team": "A", "conference": "C", "passing_att": 400, "passing_int": 8,
             "passing_yds": 3500},
            # Threw two passes, neither intercepted.
            {"player_id": "2", "player_name": "Backup", "position": "QB",
             "team": "B", "conference": "C", "passing_att": 2, "passing_int": 0,
             "passing_yds": 20},
        ])

    def test_volume_floor_keeps_a_backup_off_an_ascending_board(self):
        # Regression: without a qualifier, "fewest interceptions" is topped by whoever
        # barely played — the exact flaw that made ESPN's leaders feed unusable here.
        out = cfb_advanced.leaders_from_players(self.players(), 2025)
        fewest = out[out.category == "passing_int"]
        self.assertEqual(len(fewest), 1)
        self.assertEqual(fewest.iloc[0]["player_name"], "Starter")

    def test_the_qualifier_is_recorded_so_a_short_board_can_explain_itself(self):
        out = cfb_advanced.leaders_from_players(self.players(), 2025)
        fewest = out[out.category == "passing_int"].iloc[0]
        self.assertIn("passing_att", fewest["qualifier"])

    def test_counting_boards_need_no_floor(self):
        out = cfb_advanced.leaders_from_players(self.players(), 2025)
        yards = out[out.category == "passing_yds"]
        self.assertEqual(len(yards), 2)
        self.assertEqual(yards.iloc[0]["player_name"], "Starter")

    def test_direction_is_carried_so_the_ui_need_not_guess(self):
        out = cfb_advanced.leaders_from_players(self.players(), 2025)
        self.assertFalse(bool(out[out.category == "passing_int"].iloc[0]
                              ["higher_is_better"]))
        self.assertTrue(bool(out[out.category == "passing_yds"].iloc[0]
                             ["higher_is_better"]))


class TestSpreadSign(unittest.TestCase):
    """The check that stands between a real Vegas baseline and a silent 50% one."""

    def frames(self, invert=False):
        # Home wins by 10 when favoured; the stack's convention is that a positive
        # spread_line means the home side is favoured.
        rows, games = [], []
        for i in range(60):
            home_favoured = i % 2 == 0
            spread = 7.0 if home_favoured else -7.0
            if invert:
                spread = -spread
            rows.append({"game_id": str(i), "spread_line": spread})
            margin = 10 if home_favoured else -10
            games.append({"game_id": str(i), "home_score": 20 + margin,
                          "away_score": 20})
        return pd.DataFrame(rows), pd.DataFrame(games)

    def test_correct_convention_passes(self):
        lines, games = self.frames()
        corr = cfb_advanced.assert_spread_sign(lines, games)
        self.assertGreater(corr, 0)

    def test_inverted_convention_raises(self):
        lines, games = self.frames(invert=True)
        with self.assertRaises(ValueError) as ctx:
            cfb_advanced.assert_spread_sign(lines, games)
        self.assertIn("inverted", str(ctx.exception))

    def test_too_few_games_is_skipped_not_asserted(self):
        lines, games = self.frames()
        result = cfb_advanced.assert_spread_sign(lines.head(5), games.head(5))
        self.assertTrue(pd.isna(result))


class TestFramePinning(unittest.TestCase):
    def test_text_columns_stay_text_and_the_rest_go_numeric(self):
        # An all-null column inferring as float64 is what once made a load fail on a
        # clustering field; pinning dtypes makes autodetect deterministic.
        df = cfbd.as_frame(
            [{"team": "Ohio State", "ppa": "0.4", "blank": None}],
            text_columns=("team",),
        )
        self.assertEqual(df["team"].dtype.name, "string")
        self.assertEqual(df["ppa"].iloc[0], 0.4)

    def test_empty_input_yields_an_empty_frame_not_an_error(self):
        self.assertTrue(cfbd.as_frame([]).empty)


if __name__ == "__main__":
    unittest.main()
