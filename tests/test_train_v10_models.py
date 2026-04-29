import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import train_v10_models


class TrainV10ModelsTests(unittest.TestCase):
    def test_prepare_features_keeps_v10_matchup_columns(self):
        trainer = object.__new__(train_v10_models.V10Trainer)
        trainer.feature_cols = []
        trainer.fill_values = {}

        train_df = pd.DataFrame(
            [
                {
                    "home_won": 1,
                    "home_elo": 1520.0,
                    "lineup_woba_differential": 0.021,
                    "matchup_advantage_home": 0.018,
                    "lineup_confirmed": 1,
                },
                {
                    "home_won": 0,
                    "home_elo": 1480.0,
                    "lineup_woba_differential": -0.015,
                    "matchup_advantage_home": -0.012,
                    "lineup_confirmed": 1,
                },
            ]
        )
        val_df = pd.DataFrame(
            [
                {
                    "home_won": 1,
                    "home_elo": 1505.0,
                    "lineup_woba_differential": 0.010,
                    "matchup_advantage_home": 0.006,
                    "lineup_confirmed": 0,
                }
            ]
        )

        X_train, y_train, X_val, y_val = trainer.prepare_features(train_df, val_df)

        self.assertIn("lineup_woba_differential", trainer.feature_cols)
        self.assertIn("matchup_advantage_home", trainer.feature_cols)
        self.assertIn("lineup_confirmed", trainer.feature_cols)
        self.assertEqual(len(trainer.feature_cols), X_train.shape[1])
        self.assertEqual(2, len(y_train))
        self.assertEqual(1, len(y_val))
        self.assertEqual(set(trainer.feature_cols), set(X_val.columns))


if __name__ == "__main__":
    unittest.main()
