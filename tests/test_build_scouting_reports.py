import os
import sys
import unittest
from datetime import date
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import build_scouting_reports


class FetchGamesOnDateTests(unittest.TestCase):
    @patch.object(build_scouting_reports, "_q")
    def test_falls_back_to_predictions_when_games_table_is_empty(self, mock_query):
        mock_query.side_effect = [
            [],
            [
                {
                    "game_pk": 824776,
                    "game_date": date(2026, 4, 20),
                    "home_team_id": 111,
                    "away_team_id": 116,
                    "home_team_name": "Boston Red Sox",
                    "away_team_name": "Detroit Tigers",
                    "home_score": None,
                    "away_score": None,
                    "status": "Preview",
                    "venue_name": None,
                }
            ],
        ]

        rows = build_scouting_reports.fetch_games_on_date(object(), date(2026, 4, 20))

        self.assertEqual(1, len(rows))
        self.assertEqual(824776, rows[0]["game_pk"])
        self.assertEqual(2, mock_query.call_count)
        self.assertIn(".games`", mock_query.call_args_list[0].args[1])
        self.assertIn(".game_predictions`", mock_query.call_args_list[1].args[1])


if __name__ == "__main__":
    unittest.main()
