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

    # ---- 2026-09-25 builder fixes -------------------------------------------------
    def _row(self, game_date, team_game_count=None):
        return self.builder._assemble_row(
            game={"game_pk": 1, "game_date": game_date, "home_team_id": 111,
                  "away_team_id": 116, "home_team_name": "Red Sox",
                  "away_team_name": "Tigers", "venue_id": None,
                  "series_game_number": 1, "games_in_series": 4},
            v8_df=pd.DataFrame(), matchup_df=pd.DataFrame(), team_rolling={},
            team_quality={}, sp_lookup={}, team_game_count=team_game_count or {},
            target_date=date.fromisoformat(game_date), season=2026, elo_direct={},
        )

    def test_day_of_week_uses_training_encoding(self):
        # V8 training parquet: 1..7 with Sunday = 1
        self.assertEqual(1, self._row("2026-09-20")["day_of_week"])  # Sunday
        self.assertEqual(2, self._row("2026-09-21")["day_of_week"])  # Monday
        self.assertEqual(7, self._row("2026-09-26")["day_of_week"])  # Saturday
        self.assertEqual(1, self._row("2026-09-20")["is_weekend"])
        self.assertEqual(0, self._row("2026-09-21")["is_weekend"])

    def test_no_placeholder_team_quality_values(self):
        row = self._row("2026-09-21")
        for col in ("home_fg_xfip", "away_fg_xfip", "fg_xfip_differential",
                    "home_fg_whiff_pct", "home_fg_fbv_pct", "home_fg_ev_pct",
                    "home_fg_hh_pct", "home_fg_brl_pct"):
            self.assertIsNone(row[col], col)
        for col in ("fg_xfip", "fg_whiff_pct", "fg_fbv_pct", "fg_ev_pct", "fg_hh_pct", "fg_brl_pct"):
            self.assertNotIn(f"home_{col}", build_v10_features_live.V10_MODEL_FEATURES)
        self.assertEqual(4, row["games_in_series"])

    def test_season_game_count_ignores_prior_season(self):
        hist = pd.DataFrame({
            "game_pk": range(6),
            "game_date": pd.to_datetime(["2025-09-01", "2025-09-02", "2025-09-03",
                                         "2026-04-01", "2026-04-02", "2026-04-03"]),
            "home_team_id": [111] * 6, "away_team_id": [116] * 6,
            "home_score": [1] * 6, "away_score": [0] * 6,
        })
        b = self.builder
        b._load_v8_features = lambda pks: pd.DataFrame()
        b._load_matchup_features = lambda pks: pd.DataFrame()
        b._load_elo_direct = lambda: {}
        b._load_game_history = lambda d: hist
        b._compute_team_rolling = lambda h: {}
        b._fetch_team_quality = lambda season: {}
        b._load_sp_lookup = lambda: {}
        b._assemble_row = MagicMock(return_value={"game_pk": 9})
        b._build_and_write([{"game_pk": 9}], date(2026, 4, 5))
        counts = b._assemble_row.call_args.args[6]
        self.assertEqual({111: 3, 116: 3}, counts)
        row = self._row_with_counts(counts)
        self.assertEqual(3, row["season_game_number"])
        self.assertEqual(0, row["is_late_season"])
        self.assertEqual(1, row["is_early_season"])

    def _row_with_counts(self, counts):
        return build_v10_features_live.V10LiveFeatureBuilder._assemble_row(
            self.builder,
            {"game_pk": 9, "game_date": "2026-04-05", "home_team_id": 111,
             "away_team_id": 116, "home_team_name": "a", "away_team_name": "b",
             "venue_id": None}, pd.DataFrame(), pd.DataFrame(), {}, {}, {}, counts,
            date(2026, 4, 5), 2026, {})

    def test_woba_from_counting_stats(self):
        s = {"baseOnBalls": 500, "intentionalWalks": 20, "hitByPitch": 60, "hits": 1300,
             "doubles": 260, "triples": 20, "homeRuns": 180, "atBats": 5300, "sacFlies": 40}
        w = build_v10_features_live._woba(s)
        # (0.689*480 + 0.720*60 + 0.882*840 + 1.254*260 + 1.590*20 + 2.050*180) / 5880
        self.assertAlmostEqual(0.31320, w, places=4)
        self.assertIsNone(build_v10_features_live._woba({"hits": 1}))

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
