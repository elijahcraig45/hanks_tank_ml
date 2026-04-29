import os
import sys
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import predict_today_games


class DummyV10Model:
    feature_names_in_ = np.array(["feature_a"])

    def __init__(self):
        self.last_input = None

    def predict_proba(self, X):
        self.last_input = X.copy()
        return np.array([[0.4, 0.6]])


class PredictTodayGamesTests(unittest.TestCase):
    def test_predict_game_uses_fill_values_for_missing_v10_features(self):
        predictor = object.__new__(predict_today_games.DailyPredictor)
        predictor.model = DummyV10Model()
        predictor.scaler = None
        predictor.feature_names = ["feature_a"]
        predictor.fill_values = {"feature_a": 0.33}
        predictor.model_version = "v10"
        predictor._is_v10 = True

        result = predictor.predict_game(
            game={"home_team_name": "Home", "away_team_name": "Away"},
            feat_df=pd.DataFrame([{"feature_a": np.nan}]),
        )

        self.assertEqual("medium", result["confidence_tier"])
        self.assertAlmostEqual(0.33, predictor.model.last_input.iloc[0]["feature_a"])

    @patch.object(predict_today_games.requests, "get")
    def test_fetch_schedule_can_include_final_games(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "dates": [
                {
                    "date": "2026-04-20",
                    "games": [
                        {
                            "gamePk": 824447,
                            "gameDate": "2026-04-20T23:10:00Z",
                            "status": {"abstractGameState": "Final"},
                            "teams": {
                                "home": {"team": {"id": 111, "name": "Red Sox"}},
                                "away": {"team": {"id": 116, "name": "Tigers"}},
                            },
                            "venue": {"name": "Fenway Park"},
                        }
                    ],
                }
            ]
        }
        mock_get.return_value = mock_response

        predictor = object.__new__(predict_today_games.DailyPredictor)
        excluded = predictor.fetch_schedule(date(2026, 4, 20))
        included = predictor.fetch_schedule(date(2026, 4, 20), include_final=True)

        self.assertEqual([], excluded)
        self.assertEqual(1, len(included))
        self.assertEqual(824447, included[0]["game_pk"])


if __name__ == "__main__":
    unittest.main()
