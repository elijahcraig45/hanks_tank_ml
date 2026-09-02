"""Tests for the CFB BigQuery write paths.

These exist because of a real incident: the weekly ingest wiped 2021-2024 out of
cfb_historical.games. `load()` defaulted to WRITE_TRUNCATE, `fetch_history` in the
ingest mode is scoped to a single season, and on a cold Cloud Function container the
/tmp parquet cache is empty — so the frame that replaced the entire table held nothing
but the current season.

Two claims are pinned here: a season refresh replaces only that season, and a write
that would shrink a season already in BigQuery fails instead of succeeding quietly.
Both are asserted against a fake client, so nothing here touches BigQuery or ESPN.
"""

import os
import sys
import unittest
import unittest.mock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "cfb"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "nfl"))

import backfill_cfb  # noqa: E402


def games_frame(seasons):
    """One row per (season, index) — only the columns the write path touches."""
    rows = []
    for season, n in seasons.items():
        for i in range(n):
            rows.append({
                "game_id": f"{season}-{i}",
                "season": season,
                "division": "fbs",
                "game_date": pd.Timestamp(f"{season}-09-05"),
            })
    return pd.DataFrame(rows)


class FakeQueryJob:
    def __init__(self, rows=None):
        self._rows = rows or []

    def result(self):
        return self._rows


class FakeClient:
    """Records the SQL it is asked to run and the load-job configs it is handed."""

    def __init__(self, season_counts=None, query_raises=False):
        self.season_counts = season_counts or {}
        self.query_raises = query_raises
        self.queries = []
        self.loads = []

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        if self.query_raises:
            raise RuntimeError("404 Not found: Table hankstank:cfb_historical.games")
        if "COUNT(*)" in sql:
            return FakeQueryJob(
                [{"season": s, "n": n} for s, n in self.season_counts.items()]
            )
        return FakeQueryJob()

    def load_table_from_dataframe(self, df, table_id, job_config=None):
        self.loads.append({"df": df, "table_id": table_id, "cfg": job_config})
        return FakeQueryJob()


class TestLoadDefault(unittest.TestCase):
    def test_load_defaults_to_append_not_truncate(self):
        """The truncating default is the bug. It must not come back."""
        import inspect

        sig = inspect.signature(backfill_cfb.load)
        self.assertEqual(
            sig.parameters["write_disposition"].default, "WRITE_APPEND"
        )


class TestReplaceSeasons(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient(season_counts={2021: 1569, 2025: 1686})
        patcher = unittest.mock.patch.object(
            backfill_cfb, "_bq", return_value=self.client
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_deletes_only_the_seasons_present_in_the_frame(self):
        backfill_cfb.replace_seasons(
            games_frame({2026: 120}), "cfb_historical", "games"
        )

        deletes = [q for q, _ in self.client.queries if q.strip().startswith("DELETE")]
        self.assertEqual(len(deletes), 1)
        self.assertIn("season IN UNNEST(@seasons)", deletes[0])

        cfg = next(c for q, c in self.client.queries
                   if q.strip().startswith("DELETE"))
        self.assertEqual(cfg.query_parameters[0].values, [2026])

    def test_write_is_an_append(self):
        backfill_cfb.replace_seasons(
            games_frame({2026: 120}), "cfb_historical", "games"
        )
        self.assertEqual(len(self.client.loads), 1)
        self.assertEqual(
            self.client.loads[0]["cfg"].write_disposition, "WRITE_APPEND"
        )

    def test_refuses_a_write_that_would_shrink_a_complete_season(self):
        """The cold-cache signature: 40 rows offered against 1,686 already stored."""
        with self.assertRaises(RuntimeError) as ctx:
            backfill_cfb.replace_seasons(
                games_frame({2025: 40}), "cfb_historical", "games"
            )
        self.assertIn("cold-cache", str(ctx.exception))
        self.assertEqual(self.client.loads, [], "must not write after refusing")

    def test_allows_a_season_still_in_progress_to_grow(self):
        """2026 has few rows in BigQuery and few incoming; that is not a shrink."""
        self.client.season_counts = {2026: 76}
        backfill_cfb.replace_seasons(
            games_frame({2026: 99}), "cfb_historical", "games"
        )
        self.assertEqual(len(self.client.loads), 1)

    def test_missing_table_does_not_stop_the_first_load(self):
        client = FakeClient(query_raises=True)
        with unittest.mock.patch.object(backfill_cfb, "_bq", return_value=client):
            backfill_cfb.replace_seasons(
                games_frame({2026: 10}), "cfb_historical", "games"
            )
        self.assertEqual(len(client.loads), 1)

    def test_empty_frame_writes_nothing(self):
        backfill_cfb.replace_seasons(
            pd.DataFrame(columns=["season"]), "cfb_historical", "games"
        )
        self.assertEqual(self.client.loads, [])
        self.assertEqual(self.client.queries, [])


class TestReplaceGameIds(unittest.TestCase):
    """Predictions carry rows for unplayed games, so they scope by game, not season."""

    def setUp(self):
        self.client = FakeClient()
        patcher = unittest.mock.patch.object(
            backfill_cfb, "_bq", return_value=self.client
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_deletes_by_game_id_never_by_season(self):
        backfill_cfb.replace_game_ids(
            games_frame({2026: 3}), "cfb_season", "game_predictions"
        )
        deletes = [q for q, _ in self.client.queries if q.strip().startswith("DELETE")]
        self.assertEqual(len(deletes), 1)
        self.assertIn("game_id IN UNNEST(@ids)", deletes[0])
        # "season" appears in the dataset name, so assert on the filter itself: a
        # season-scoped delete here would drop predictions for unplayed weeks.
        where = deletes[0].split("WHERE", 1)[1]
        self.assertNotIn("season", where)

        cfg = next(c for q, c in self.client.queries
                   if q.strip().startswith("DELETE"))
        self.assertEqual(
            cfg.query_parameters[0].values, ["2026-0", "2026-1", "2026-2"]
        )


if __name__ == "__main__":
    unittest.main()
