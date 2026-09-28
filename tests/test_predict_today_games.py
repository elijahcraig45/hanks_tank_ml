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


class _Job:
    def __init__(self):
        self.state, self.error_result, self.errors = "DONE", None, None

    def result(self):
        return self


class _WriterBQ:
    """Job ids unique per project (409 on reuse), DELETEs recorded, thread-safe."""

    def __init__(self):
        import threading
        self.jobs, self.loaded, self.deletes, self.lock = {}, [], [], threading.Lock()

    def load_table_from_file(self, fh, table, job_id=None, job_config=None):
        from google.api_core import exceptions as gexc
        with self.lock:
            if job_id in self.jobs:
                raise gexc.Conflict(f"Already Exists: Job {job_id}")
            self.jobs[job_id] = _Job()
            self.loaded.append(fh.read().decode())
            return self.jobs[job_id]

    def get_job(self, job_id):
        return self.jobs[job_id]

    def query(self, sql, job_config=None):
        with self.lock:
            self.deletes.append((sql, {p.name: p for p in job_config.query_parameters}))
        return _Job()


class WritePredictionsIdempotencyTests(unittest.TestCase):
    def _predictor(self, bq):
        p = object.__new__(predict_today_games.DailyPredictor)
        p.bq, p.model_version, p.dry_run = bq, "v10", False
        return p

    def _rows(self, at, prob=0.55):
        return [{"game_pk": 823650, "game_date": "2026-09-27", "home_win_probability": prob,
                 "model_version": "v10", "predicted_at": at}]

    def test_twin_task_appends_nothing(self):
        bq = _WriterBQ()
        p = self._predictor(bq)
        d = date(2026, 9, 27)
        self.assertEqual(p._write_predictions(self._rows("2026-09-27T17:40:08.745392+00:00"), d), "ok")
        # the weekend's twin: same content, 13 ms later
        self.assertEqual(p._write_predictions(self._rows("2026-09-27T17:40:08.758921+00:00"), d), "duplicate")
        self.assertEqual(len(bq.loaded), 1)
        self.assertEqual(len(bq.deletes), 1)       # the twin must not delete anything

    def test_concurrent_twins_leave_exactly_one_row(self):
        import threading
        bq = _WriterBQ()
        d, out = date(2026, 9, 27), []
        go = threading.Barrier(6)

        def twin(i):
            go.wait()
            out.append(self._predictor(bq)._write_predictions(
                self._rows(f"2026-09-27T17:40:08.{700000 + i:06d}+00:00"), d))
        ts = [threading.Thread(target=twin, args=(i,)) for i in range(6)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(sorted(out), ["duplicate"] * 5 + ["ok"])
        self.assertEqual(len(bq.loaded), 1)

    def test_later_checkpoint_writes_and_supersedes_older_rows(self):
        bq = _WriterBQ()
        p = self._predictor(bq)
        d = date(2026, 9, 27)
        p._write_predictions(self._rows("2026-09-27T13:10:00+00:00"), d)
        # T-90 with the same inputs as T-360: a new bucket, so it is written (latest wins)
        self.assertEqual(p._write_predictions(self._rows("2026-09-27T17:40:00+00:00"), d), "ok")
        self.assertEqual(len(bq.loaded), 2)
        sql, params = bq.deletes[-1]
        self.assertIn("predicted_at < @run_start", sql)
        self.assertEqual(str(params["run_start"].value)[:19], "2026-09-27 17:40:00")
        self.assertEqual(params["pks"].values, [823650])
