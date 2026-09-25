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

    def _daily_patches(self):
        names = ["_run_collection", "_run_validation", "_run_features",
                 "_run_v8_elo_update", "_run_v8_features", "_run_v10_features",
                 "_run_power_rankings", "_run_weekly_predictions", "_run_v7_features",
                 "_run_scouting_reports", "_run_weekly_training_v8", "_refresh_sp_gcs",
                 "_run_rosters"]
        patchers = {n: patch.object(cloud_function_main, n, return_value={"step": n}) for n in names}
        mocks = {n: p.start() for n, p in patchers.items()}
        for p in patchers.values():
            self.addCleanup(p.stop)
        return mocks

    def test_daily_skips_post_hoc_v7_scouting_and_training(self):
        mocks = self._daily_patches()
        # 2026-09-21 is a Monday, so target = Sunday: the old code trained here
        body, status, _ = cloud_function_main.daily_pipeline(
            FakeRequest({"mode": "daily", "date": "2026-09-20", "dry_run": True}))
        self.assertEqual(200, status)
        steps = json.loads(body)["steps"]
        self.assertTrue(all("seconds" in s for s in steps))
        for n in ("_run_v7_features", "_run_scouting_reports",
                  "_run_weekly_training_v8", "_refresh_sp_gcs"):
            mocks[n].assert_not_called()
        mocks["_run_collection"].assert_called_once()
        mocks["_run_power_rankings"].assert_called_once()

    def test_rosters_mode_runs_only_rosters(self):
        mocks = self._daily_patches()
        body, status, _ = cloud_function_main.daily_pipeline(
            FakeRequest({"mode": "rosters", "date": "2026-09-21", "dry_run": True}))
        self.assertEqual(200, status)
        self.assertEqual(["_run_rosters"], [s["step"] for s in json.loads(body)["steps"]])
        mocks["_run_collection"].assert_not_called()

    def test_scouting_reports_mode_still_available_on_demand(self):
        mocks = self._daily_patches()
        cloud_function_main.daily_pipeline(
            FakeRequest({"mode": "scouting_reports", "date": "2026-09-24", "dry_run": True}))
        mocks["_run_scouting_reports"].assert_called_once()

    def _pregame_patches(self):
        names = ["_run_lineup_fetch", "_run_matchup_features", "_run_v7_features",
                 "_run_v8_features", "_run_v10_features", "_run_daily_prediction",
                 "_run_scouting_reports", "_run_pa_sim", "_run_logit3", "_run_sim_blend"]
        patchers = {n: patch.object(cloud_function_main, n, return_value={"step": n}) for n in names}
        mocks = {n: p.start() for n, p in patchers.items()}
        for p in patchers.values():
            self.addCleanup(p.stop)
        return mocks

    def test_pregame_v10_shadows_are_off_by_default(self):
        mocks = self._pregame_patches()
        cloud_function_main.daily_pipeline(FakeRequest(
            {"mode": "pregame_v10", "date": "2026-09-20", "game_pks": [1], "dry_run": True}))
        mocks["_run_logit3"].assert_not_called()
        mocks["_run_sim_blend"].assert_not_called()
        mocks["_run_pa_sim"].assert_not_called()

    def test_pregame_v10_runs_shadows_when_flagged_before_scouting(self):
        mocks = self._pregame_patches()
        body, status, _ = cloud_function_main.daily_pipeline(FakeRequest(
            {"mode": "pregame_v10", "date": "2026-09-20", "game_pks": [7], "dry_run": True,
             "run_logit3": True, "run_sim_blend": True}))
        steps = [s["step"] for s in json.loads(body)["steps"]]
        self.assertEqual(200, status)
        self.assertLess(steps.index("_run_daily_prediction"), steps.index("_run_logit3"))
        self.assertLess(steps.index("_run_sim_blend"), steps.index("_run_scouting_reports"))
        args = mocks["_run_logit3"].call_args[0]
        self.assertEqual([7], args[1])

    def test_standalone_shadow_modes(self):
        mocks = self._pregame_patches()
        cloud_function_main.daily_pipeline(FakeRequest({"mode": "logit3", "date": "2026-09-20"}))
        cloud_function_main.daily_pipeline(FakeRequest({"mode": "sim_blend", "date": "2026-09-20"}))
        mocks["_run_logit3"].assert_called_once()
        mocks["_run_sim_blend"].assert_called_once()
        mocks["_run_daily_prediction"].assert_not_called()

    def test_sim_blend_refuses_in_a_small_container(self):
        with patch.dict(os.environ, {"SIM_BLEND_MEMORY_MB": "1024"}):
            out = cloud_function_main._run_sim_blend(
                cloud_function_main.date(2026, 9, 20), [], True, {})
        self.assertEqual("insufficient_memory", out["status"])

    def test_shadow_failure_is_non_fatal(self):
        def boom():
            raise RuntimeError("table exploded")
        out = cloud_function_main._shadow("logit3", boom)
        self.assertEqual("error", out["status"])
        self.assertIn("exploded", out["error"])


if __name__ == "__main__":
    unittest.main()
