import os
import sys
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import build_v7_features


def _query_result(df: pd.DataFrame) -> MagicMock:
    result = MagicMock()
    result.to_dataframe.return_value = df
    return result


class V7FeatureBuilderTests(unittest.TestCase):
    @patch.object(build_v7_features.bigquery, "Client")
    def test_load_games_df_falls_back_to_predictions(self, mock_client):
        builder = build_v7_features.V7FeatureBuilder(dry_run=True)
        builder.bq.query = MagicMock(
            side_effect=[
                _query_result(pd.DataFrame(columns=["game_pk", "home_team_id", "away_team_id", "venue_id"])),
                _query_result(pd.DataFrame([
                    {"game_pk": 824776, "home_team_id": 111, "away_team_id": 116, "venue_id": None}
                ])),
            ]
        )

        games_df = builder._load_games_df(date(2026, 4, 20), [824776])

        self.assertEqual([824776], games_df["game_pk"].tolist())
        self.assertEqual(2, builder.bq.query.call_count)
        self.assertIn(".games`", builder.bq.query.call_args_list[0].args[0])
        self.assertIn(".game_predictions`", builder.bq.query.call_args_list[1].args[0])

    @patch.object(build_v7_features.bigquery, "Client")
    def test_get_starters_and_venue_uses_prediction_starters_when_lineups_missing(self, mock_client):
        builder = build_v7_features.V7FeatureBuilder(dry_run=True)
        builder.bq.query = MagicMock(
            return_value=_query_result(
                pd.DataFrame([
                    {"home_starter_id": 543243, "away_starter_id": 656427, "venue_id": None}
                ])
            )
        )

        with patch.object(builder, "_infer_starters_from_statcast", return_value=(None, None)) as infer_mock:
            home_starter_id, away_starter_id, venue_id, _ = builder._get_starters_and_venue(
                824776,
                date(2026, 4, 20),
            )

        self.assertEqual(543243, home_starter_id)
        self.assertEqual(656427, away_starter_id)
        self.assertIsNone(venue_id)
        infer_mock.assert_not_called()

    @patch.object(build_v7_features.bigquery, "Client")
    def test_finalize_rows_scopes_delete_to_requested_game_pks(self, mock_client):
        builder = build_v7_features.V7FeatureBuilder(dry_run=False)
        builder.bq.get_table = MagicMock()
        builder.bq.query = MagicMock(return_value=MagicMock(result=MagicMock()))
        builder.bq.load_table_from_dataframe = MagicMock(return_value=MagicMock(result=MagicMock()))

        result = builder._finalize_rows(
            [
                {
                    "game_pk": 824776,
                    "game_date": date(2026, 4, 20),
                    "home_team_id": 111,
                    "away_team_id": 116,
                }
            ],
            date(2026, 4, 20),
            [824776],
        )

        self.assertEqual("ok", result["status"])
        delete_sql = builder.bq.query.call_args_list[0].args[0]
        self.assertIn("AND game_pk IN (824776)", delete_sql)


if __name__ == "__main__":
    unittest.main()
