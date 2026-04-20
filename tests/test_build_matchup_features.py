import os
import sys
import unittest
from datetime import date

import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from build_matchup_features import MatchupFeatureBuilder


class ProjectedLineupTests(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(MatchupFeatureBuilder)

    def test_projects_recent_confirmed_lineup_when_current_lineup_missing(self):
        game = {
            "game_pk": 999001,
            "game_date": "2026-04-20",
            "home_team_id": 144,
            "home_probable_pitcher_id": 900,
            "home_probable_pitcher": "Probable Arm",
        }

        current_team_rows = pd.DataFrame()
        recent_team_rows = pd.DataFrame(
            [
                {
                    "game_pk": 999000,
                    "game_date": "2026-04-19",
                    "team_id": 144,
                    "team_type": "home",
                    "player_id": player_id,
                    "player_name": f"Batter {batting_order}",
                    "batting_order": batting_order,
                    "position": "OF",
                    "bat_side": "R",
                    "pitch_hand": "",
                    "is_starter": True,
                    "is_probable_pitcher": False,
                    "lineup_confirmed": True,
                }
                for batting_order, player_id in enumerate(range(101, 110), start=1)
            ]
            + [
                {
                    "game_pk": 999000,
                    "game_date": "2026-04-19",
                    "team_id": 144,
                    "team_type": "home",
                    "player_id": 900,
                    "player_name": "Probable Arm",
                    "batting_order": None,
                    "position": "P",
                    "bat_side": "R",
                    "pitch_hand": "R",
                    "is_starter": True,
                    "is_probable_pitcher": True,
                    "lineup_confirmed": True,
                }
            ]
        )

        projected_rows = self.builder._project_team_lineup_rows(
            game,
            "home",
            date(2026, 4, 20),
            current_team_rows,
            recent_team_rows,
        )

        self.assertEqual(10, len(projected_rows))
        projected_batters = [row for row in projected_rows if row["batting_order"] is not None]
        self.assertEqual(list(range(1, 10)), [row["batting_order"] for row in projected_batters])
        self.assertTrue(all(row["lineup_confirmed"] is False for row in projected_rows))
        self.assertEqual(900, next(row for row in projected_rows if row["position"] == "P")["player_id"])


if __name__ == "__main__":
    unittest.main()
