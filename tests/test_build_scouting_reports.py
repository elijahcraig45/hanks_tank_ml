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


def _game(pk, home, away):
    return {"game_pk": pk, "game_date": date(2026, 9, 20), "home_team_id": home, "away_team_id": away,
            "home_team_name": f"Home {home}", "away_team_name": f"Away {away}", "status": "Preview"}


GAMES = [_game(1, 111, 116), _game(2, 112, 117), _game(3, 113, 118)]


def _report(pk):
    return {"game_pk": pk, "game_date": "2026-09-20", "home_team_id": 111, "away_team_id": 116,
            "home_team_name": "Home", "away_team_name": "Away", "generated_at": "2026-09-20T15:00:00.123456Z",
            "watch_list": [], "fun_facts": [], "news": {"home": [], "away": []}}


class ScopeGamesTests(unittest.TestCase):
    def test_no_game_pks_keeps_the_whole_slate(self):
        self.assertEqual(GAMES, build_scouting_reports.scope_games(GAMES, None))
        self.assertEqual(GAMES, build_scouting_reports.scope_games(GAMES, []))

    def test_keeps_only_the_requested_games_and_accepts_strings(self):
        kept = build_scouting_reports.scope_games(GAMES, ["2", 3])
        self.assertEqual([2, 3], [g["game_pk"] for g in kept])

    def test_unknown_game_keeps_nothing(self):
        self.assertEqual([], build_scouting_reports.scope_games(GAMES, [999]))


class RunScopingTests(unittest.TestCase):
    """A pregame task is for one game. It must build that game, write only that game, and leave the others."""

    def setUp(self):
        m = build_scouting_reports
        names = {
            "fetch_games_on_date": GAMES, "fetch_predictions": {}, "fetch_v8_features": {}, "fetch_matchup_features": {},
            "fetch_hot_cold_players": {}, "fetch_team_news": {},
            "fetch_team_abbrevs": {111: "AAA", 112: "BBB", 113: "CCC", 116: "DDD", 117: "EEE", 118: "FFF"},
            "fetch_batter_vs_team_matchups": {"home": [], "away": []}, "compute_hit_streaks": {},
            "fetch_yearly_h2h_records": [], "fetch_batter_vs_pitcher": {}, "fetch_venue_batter_stats": {"home": [], "away": []},
            "upsert_reports": {"reports_written": 0},
        }
        self.m = {}
        for n, ret in names.items():
            p = patch.object(m, n, return_value=ret)
            self.m[n] = p.start()
            self.addCleanup(p.stop)
        p = patch.object(m, "assemble_report", side_effect=lambda **kw: _report(kw["game"]["game_pk"]))
        self.m["assemble_report"] = p.start()
        self.addCleanup(p.stop)
        p = patch.object(m.bigquery, "Client")
        p.start()
        self.addCleanup(p.stop)

    def test_scoped_run_builds_one_game_and_merges_it(self):
        build_scouting_reports.run(date(2026, 9, 20), game_pks=[2])
        self.assertEqual(1, self.m["fetch_batter_vs_team_matchups"].call_count)
        self.assertEqual(1, self.m["assemble_report"].call_count)
        reports = self.m["upsert_reports"].call_args.args[1]
        self.assertEqual([2], [r["game_pk"] for r in reports])
        self.assertTrue(self.m["upsert_reports"].call_args.kwargs["merge"])

    def test_scoped_run_only_asks_about_that_games_teams(self):
        build_scouting_reports.run(date(2026, 9, 20), game_pks=[2])
        team_ids = self.m["fetch_hot_cold_players"].call_args.args[1]
        self.assertEqual({112, 117}, set(team_ids))

    def test_unscoped_run_builds_everything_and_overwrites_the_day(self):
        build_scouting_reports.run(date(2026, 9, 20))
        self.assertEqual(3, self.m["assemble_report"].call_count)
        self.assertEqual(3, len(self.m["upsert_reports"].call_args.args[1]))
        self.assertFalse(self.m["upsert_reports"].call_args.kwargs["merge"])

    def test_scoped_run_that_matches_nothing_writes_nothing(self):
        # The dangerous case: falling through to the overwrite here would empty the day.
        result = build_scouting_reports.run(date(2026, 9, 20), game_pks=[999])
        self.assertEqual(0, result["reports_written"])
        self.m["upsert_reports"].assert_not_called()
        self.m["assemble_report"].assert_not_called()


class FakeBq:
    def __init__(self):
        self.queries, self.loads = [], []

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        return type("Job", (), {"result": lambda s: None})()

    def load_table_from_json(self, rows, table_ref, job_config=None):
        self.loads.append((rows, table_ref, job_config))
        return type("Job", (), {"result": lambda s: None})()


class WritePathTests(unittest.TestCase):
    def test_merge_sends_one_atomic_statement_and_never_truncates(self):
        bq = FakeBq()
        out = build_scouting_reports.upsert_reports(bq, [_report(1), _report(2)], merge=True)
        self.assertEqual(2, out["reports_written"])
        self.assertEqual([], bq.loads)  # a load with WRITE_TRUNCATE is what would erase the other games
        sql, cfg = bq.queries[0]
        self.assertIn("MERGE", sql)
        self.assertNotIn("TRUNCATE", sql.upper())
        params = {p.name: p for p in cfg.query_parameters}
        self.assertEqual({"game_date", "rows"}, set(params))
        self.assertEqual(2, len(params["rows"].values))

    def test_merge_parameters_have_the_types_the_table_expects(self):
        bq = FakeBq()
        build_scouting_reports.merge_reports(bq, [_report(7)])
        rows = {p.name: p for p in bq.queries[0][1].query_parameters}["rows"].to_api_repr()
        fields = rows["parameterType"]["arrayType"]["structTypes"]
        self.assertEqual(
            {"game_pk": "INT64", "game_date": "DATE", "home_team_id": "INT64", "away_team_id": "INT64",
             "home_team_name": "STRING", "away_team_name": "STRING", "report": "STRING", "generated_at": "TIMESTAMP"},
            {f["name"]: f["type"]["type"] for f in fields})
        values = rows["parameterValue"]["arrayValues"][0]["structValues"]
        self.assertEqual("7", values["game_pk"]["value"])
        self.assertEqual("2026-09-20", values["game_date"]["value"])
        self.assertEqual(7, __import__("json").loads(values["report"]["value"])["game_pk"])

    def test_merge_with_no_reports_does_nothing(self):
        bq = FakeBq()
        self.assertEqual(0, build_scouting_reports.merge_reports(bq, [])["reports_written"])
        self.assertEqual([], bq.queries)

    def test_default_path_still_overwrites_the_days_partition(self):
        bq = FakeBq()
        build_scouting_reports.upsert_reports(bq, [_report(1)])
        self.assertEqual([], bq.queries)
        _rows, table_ref, cfg = bq.loads[0]
        self.assertTrue(table_ref.endswith("game_scouting_reports$20260920"))
        self.assertEqual("WRITE_TRUNCATE", cfg.write_disposition)

    def test_dry_run_writes_nothing_in_either_mode(self):
        bq = FakeBq()
        build_scouting_reports.upsert_reports(bq, [_report(1)], dry_run=True, merge=True)
        build_scouting_reports.upsert_reports(bq, [_report(1)], dry_run=True)
        self.assertEqual(([], []), (bq.queries, bq.loads))


if __name__ == "__main__":
    unittest.main()
