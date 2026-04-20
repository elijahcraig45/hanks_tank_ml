import json
import os
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cloud_function_main


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    def get_json(self, silent=True):
        return self.payload


class CloudFunctionMainTests(unittest.TestCase):
    @patch.object(cloud_function_main, "_run_scouting_reports", return_value={"step": "scouting_reports"})
    @patch.object(cloud_function_main, "_run_daily_prediction", return_value={"step": "predict_today"})
    @patch.object(cloud_function_main, "_run_v10_features", return_value={"step": "v10_features"})
    @patch.object(cloud_function_main, "_run_v8_features", return_value={"step": "v8_features"})
    @patch.object(cloud_function_main, "_run_v7_features", return_value={"step": "v7_features"})
    @patch.object(cloud_function_main, "_run_matchup_features", return_value={"step": "matchup_features"})
    @patch.object(cloud_function_main, "_run_lineup_fetch", return_value={"step": "lineups"})
    def test_pregame_v10_runs_v7_and_scouting_reports(
        self,
        _lineups_mock,
        _matchup_mock,
        v7_mock,
        _v8_mock,
        _v10_mock,
        _predict_mock,
        scouting_mock,
    ):
        response_body, status_code, _headers = cloud_function_main.daily_pipeline(
            FakeRequest(
                {
                    "mode": "pregame_v10",
                    "date": "2026-04-20",
                    "game_pks": [824776],
                    "dry_run": True,
                }
            )
        )

        response_json = json.loads(response_body)
        step_names = [step["step"] for step in response_json["steps"]]

        self.assertEqual(200, status_code)
        self.assertEqual(
            [
                "lineups",
                "matchup_features",
                "v7_features",
                "v8_features",
                "v10_features",
                "predict_today",
                "scouting_reports",
            ],
            step_names,
        )
        v7_mock.assert_called_once()
        scouting_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
