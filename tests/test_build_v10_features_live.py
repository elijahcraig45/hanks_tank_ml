import os
import sys
import unittest
from datetime import date
from unittest.mock import MagicMock

import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import build_v10_features_live


class V10LiveFeatureBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(build_v10_features_live.V10LiveFeatureBuilder)
        self.builder.dry_run = True
        self.builder.bq = MagicMock()

    def test_assemble_row_includes_matchup_features(self):
        game = {
            "game_pk": 824447,
            "game_date": "2026-04-20",
            "home_team_id": 111,
            "away_team_id": 116,
            "home_team_name": "Red Sox",
            "away_team_name": "Tigers",
            "venue_id": None,
            "series_game_number": 2,
            "games_in_series": 3,
        }
        v8_df = pd.DataFrame([{"game_pk": 824447, "home_elo": 1535.0, "away_elo": 1495.0}])
        matchup_df = pd.DataFrame(
            [
                {
                    "game_pk": 824447,
                    "lineup_confirmed": True,
                    "home_lineup_woba_vs_hand": 0.344,
                    "away_lineup_woba_vs_hand": 0.318,
                    "home_lineup_k_pct_vs_hand": 0.201,
                    "away_lineup_k_pct_vs_hand": 0.236,
                    "home_top3_woba_vs_hand": 0.366,
                    "away_top3_woba_vs_hand": 0.331,
                    "home_middle4_woba_vs_hand": 0.349,
                    "away_middle4_woba_vs_hand": 0.314,
                    "home_bottom2_woba_vs_hand": 0.309,
                    "away_bottom2_woba_vs_hand": 0.287,
                    "home_pct_same_hand": 0.44,
                    "away_pct_same_hand": 0.62,
                    "home_h2h_woba": 0.352,
                    "away_h2h_woba": 0.301,
                    "matchup_advantage_home": 0.026,
                }
            ]
        )

        row = self.builder._assemble_row(
            game=game,
            v8_df=v8_df,
            matchup_df=matchup_df,
            team_rolling={},
            team_quality={},
            sp_lookup={},
            team_game_count={111: 18, 116: 17},
            target_date=date(2026, 4, 20),
            season=2026,
            elo_direct={},
        )

        self.assertEqual(1, row["lineup_confirmed"])
        self.assertAlmostEqual(0.344, row["home_lineup_woba_vs_hand"])
        self.assertAlmostEqual(0.318, row["away_lineup_woba_vs_hand"])
        self.assertAlmostEqual(0.026, row["lineup_woba_differential"])
        self.assertAlmostEqual(0.035, row["lineup_k_pct_differential"])
        self.assertAlmostEqual(0.352, row["home_h2h_woba"])
        self.assertAlmostEqual(0.301, row["away_h2h_woba"])
        self.assertAlmostEqual(0.051, row["h2h_woba_differential"])
        self.assertAlmostEqual(0.026, row["matchup_advantage_home"])

    def test_assemble_row_uses_neutral_matchup_defaults_when_missing(self):
        row = self.builder._assemble_row(
            game={
                "game_pk": 824448,
                "game_date": "2026-04-21",
                "home_team_id": 111,
                "away_team_id": 116,
                "home_team_name": "Red Sox",
                "away_team_name": "Tigers",
                "venue_id": None,
                "series_game_number": 1,
                "games_in_series": 3,
            },
            v8_df=pd.DataFrame(),
            matchup_df=pd.DataFrame(),
            team_rolling={},
            team_quality={},
            sp_lookup={},
            team_game_count={},
            target_date=date(2026, 4, 21),
            season=2026,
            elo_direct={},
        )

        self.assertEqual(0, row["lineup_confirmed"])
        self.assertAlmostEqual(0.320, row["home_lineup_woba_vs_hand"])
        self.assertAlmostEqual(0.320, row["away_lineup_woba_vs_hand"])
        self.assertAlmostEqual(0.0, row["lineup_woba_differential"])
        self.assertAlmostEqual(0.220, row["home_lineup_k_pct_vs_hand"])
        self.assertAlmostEqual(0.220, row["away_lineup_k_pct_vs_hand"])
        self.assertAlmostEqual(0.0, row["lineup_k_pct_differential"])
        self.assertAlmostEqual(0.0, row["matchup_advantage_home"])

    def test_ensure_table_extends_schema_for_new_columns(self):
        self.builder.dry_run = False
        existing_table = MagicMock()
        existing_table.schema = build_v10_features_live.V10_FEATURES_SCHEMA[:3]
        self.builder.bq.get_table.return_value = existing_table

        self.builder._ensure_table()

        self.builder.bq.update_table.assert_called_once()
        self.assertEqual(
            len(build_v10_features_live.V10_FEATURES_SCHEMA),
            len(existing_table.schema),
        )


if __name__ == "__main__":
    unittest.main()
