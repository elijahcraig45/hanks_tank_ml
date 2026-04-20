import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_scouting_reports import (  # noqa: E402
    _safe_float,
    _safe_int,
    _streak_label,
    build_hot_cold_section,
    generate_fun_facts,
)


class BuildScoutingReportsTests(unittest.TestCase):
    def test_safe_helpers_normalize_invalid_values(self):
        self.assertEqual(_safe_float("3.14159", 2), 3.14)
        self.assertIsNone(_safe_float("not-a-number"))
        self.assertEqual(_safe_int("12"), 12)
        self.assertIsNone(_safe_int("12.4"))
        self.assertEqual(_streak_label(4), "W4")
        self.assertEqual(_streak_label(-3), "L3")
        self.assertIsNone(_streak_label(None))

    def test_build_hot_cold_section_filters_thresholds(self):
        players = [
            {"player_name": "Hot One", "woba_delta": 0.03, "woba_14d": 0.42, "ab": 20, "hr": 2, "rbi": 5, "games_played": 7},
            {"player_name": "Hot Two", "woba_delta": 0.02, "woba_14d": 0.4, "ab": 18, "hr": 1, "rbi": 4, "games_played": 6},
            {"player_name": "Neutral", "woba_delta": 0.0, "woba_14d": 0.32, "ab": 15, "hr": 0, "rbi": 1, "games_played": 5},
            {"player_name": "Cold One", "woba_delta": -0.02, "woba_14d": 0.2, "ab": 19, "games_played": 7},
            {"player_name": "Cold Two", "woba_delta": -0.03, "woba_14d": 0.18, "ab": 22, "games_played": 8},
        ]

        section = build_hot_cold_section(players)

        self.assertEqual([p["name"] for p in section["hot"]], ["Hot One", "Hot Two"])
        self.assertEqual([p["name"] for p in section["cold"]], ["Cold One", "Cold Two"])

    def test_generate_fun_facts_uses_streaks_matchups_and_h2h(self):
        facts = generate_fun_facts(
            matchup={
                "home": [{"player_name": "Slugger", "woba": 0.43, "pa": 12, "hr": 2, "hits": 5}],
                "away": [{"player_name": "Cold Bat", "woba": 0.19, "pa": 16, "hr": 0, "hits": 2}],
            },
            hit_streaks={
                10: [{"player_name": "Hot Hitter", "hit_streak": 11}],
                20: [{"player_name": "Steady Bat", "hit_streak": 7}],
            },
            v8={"h2h_win_pct_3yr": 0.7, "h2h_games_3yr": 10},
            home_name="Atlanta Braves",
            away_name="New York Mets",
            home_tid=10,
            away_tid=20,
        )

        joined = " ".join(facts)
        self.assertIn("Hot Hitter", joined)
        self.assertIn("Slugger", joined)
        self.assertIn("Mets pitching", joined)
        self.assertIn("Braves have dominated", joined)
        self.assertLessEqual(len(facts), 8)


if __name__ == "__main__":
    unittest.main()
