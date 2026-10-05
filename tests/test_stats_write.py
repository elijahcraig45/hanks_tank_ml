"""The stats writer replaces a season without ever being able to erase it, and the nflverse *_list columns are read as the text BigQuery stores.

Regression for September 2026: pandas read fg_blocked_list (every value a single number) as float64, the load failed, and because the season's rows had already been deleted
the site served an empty NFL player table for weeks, while the Cloud Function returned 200."""
import os
import sys
import types

import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from google.api_core.exceptions import NotFound  # noqa: E402

from stats import build, nfl_stats  # noqa: E402

TARGET = "hankstank.nfl_season.player_season_stats"
STAGE = TARGET + "__stage"


class Field:
    def __init__(self, name, field_type="STRING"):
        self.name, self.field_type = name, field_type


class Job:
    def __init__(self, fail=None):
        self.fail = fail

    def result(self):
        if self.fail:
            raise self.fail
        return None


class FakeClient:
    def __init__(self, exists=True, target_cols=("player_id", "season"), stage_cols=None, fail_load=None, fail_script=None):
        self.log, self.exists, self.fail_load, self.fail_script = [], exists, fail_load, fail_script
        self.target_cols = [Field(c) for c in target_cols]
        self.stage_cols = [Field(c) for c in (stage_cols or target_cols)]

    def get_table(self, table_id):
        self.log.append(("get_table", table_id))
        if table_id == TARGET:
            if not self.exists:
                raise NotFound("no such table")
            return types.SimpleNamespace(schema=self.target_cols)
        return types.SimpleNamespace(schema=self.stage_cols)

    def query(self, sql):
        self.log.append(("query", sql))
        return Job(self.fail_script if "BEGIN TRANSACTION" in sql else None)

    def load_table_from_dataframe(self, df, table_id, job_config=None):
        self.log.append(("load", table_id, job_config))
        return Job(self.fail_load)

    def delete_table(self, table_id, not_found_ok=False):
        self.log.append(("delete_table", table_id))

    def kinds(self):
        return [e[0] for e in self.log]

    def sql(self):
        return [e[1] for e in self.log if e[0] == "query"]


class FakeBigQuery:
    class SchemaUpdateOption:
        ALLOW_FIELD_ADDITION = "ALLOW_FIELD_ADDITION"

    class LoadJobConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class TimePartitioning:
        def __init__(self, field):
            self.field = field


DF = pd.DataFrame({"player_id": ["a", "b"], "season": [2026, 2026]})


def run(client, df=DF, season=2026, layout=None):
    build._replace_season(client, FakeBigQuery, TARGET, df, season, layout)


# ---------------------------------------------------------------------------------------------------------------------- the swap is safe
def test_the_new_rows_are_loaded_into_a_scratch_copy_before_the_target_is_touched():
    c = FakeClient()
    run(c)
    loads = [e for e in c.log if e[0] == "load"]
    assert [e[1] for e in loads] == [STAGE]                                    # the target is never loaded into directly
    assert "LIKE" in c.sql()[0] and STAGE in c.sql()[0] and TARGET in c.sql()[0]
    first_delete = next(i for i, e in enumerate(c.log) if e[0] == "query" and "DELETE" in e[1])
    assert first_delete > c.log.index(loads[0])                               # the scratch load came first


def test_the_swap_is_one_transaction_so_a_failed_insert_rolls_the_delete_back():
    c = FakeClient()
    run(c)
    scripts = [s for s in c.sql() if "DELETE" in s or "INSERT" in s]
    assert len(scripts) == 1                                                  # DELETE and INSERT travel in the same job, never as two
    s = scripts[0]
    assert s.index("BEGIN TRANSACTION") < s.index("DELETE FROM") < s.index("INSERT INTO") < s.index("COMMIT TRANSACTION")
    assert f"DELETE FROM `{TARGET}` WHERE season = 2026" in s and f"FROM `{STAGE}`" in s and "`player_id`, `season`" in s


def test_a_load_that_fails_never_touches_the_target_and_the_scratch_table_is_dropped():
    c = FakeClient(fail_load=ValueError('Error converting Pandas column with name: "fg_blocked_list"'))
    with pytest.raises(ValueError, match="fg_blocked_list"):
        run(c)
    assert not any("DELETE" in s or "INSERT" in s for s in c.sql())           # the season is exactly as it was
    assert ("delete_table", STAGE) in c.log


def test_a_failed_transaction_propagates_and_still_cleans_up():
    c = FakeClient(fail_script=RuntimeError("Invalid value"))
    with pytest.raises(RuntimeError, match="Invalid value"):
        run(c)
    assert ("delete_table", STAGE) in c.log


def test_a_column_the_feed_added_is_added_to_the_target_before_the_transaction():
    c = FakeClient(stage_cols=("player_id", "season", "new_metric"))
    c.stage_cols[2].field_type = "FLOAT"
    run(c)
    sql = c.sql()
    alter = next(i for i, s in enumerate(sql) if s.startswith("ALTER TABLE"))
    script = next(i for i, s in enumerate(sql) if "BEGIN TRANSACTION" in s)
    assert alter < script and "ADD COLUMN IF NOT EXISTS `new_metric` FLOAT64" in sql[alter]
    assert "`new_metric`" in sql[script]


def test_a_table_that_does_not_exist_yet_is_simply_loaded_with_its_layout():
    c = FakeClient(exists=False)
    run(c, layout={"partition": "game_date", "cluster": ["season"]})
    loads = [e for e in c.log if e[0] == "load"]
    assert len(loads) == 1 and loads[0][1] == TARGET
    assert loads[0][2].time_partitioning.field == "game_date" and loads[0][2].clustering_fields == ["season"]
    assert not c.sql()                                                         # no scratch table, no delete: there was nothing to protect


def test_write_bq_skips_empty_frames_and_reports_each_table(monkeypatch):
    c = FakeClient()
    fake_bq = types.SimpleNamespace(Client=lambda project=None: c, LoadJobConfig=FakeBigQuery.LoadJobConfig, SchemaUpdateOption=FakeBigQuery.SchemaUpdateOption,
                                    TimePartitioning=FakeBigQuery.TimePartitioning)
    import google.cloud
    monkeypatch.setattr(google.cloud, "bigquery", fake_bq, raising=False)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bq)
    out = build.write_bq("nfl", 2026, {"player_season_stats": DF, "stat_leaders": pd.DataFrame()})
    assert out == [{"table": TARGET, "rows": 2}]


# ------------------------------------------------------------------------------------------------------------------- nflverse list columns
def test_list_columns_are_read_as_text_even_when_every_value_is_a_single_number(tmp_path, monkeypatch):
    monkeypatch.setattr(nfl_stats, "CACHE", tmp_path)
    (tmp_path / "nfl_player_stats_2026.csv").write_text(
        "player_id,season,fg_made_list,fg_blocked_list,gwfg_distance_list,attempts\n"
        "a,2026,51;43;41,23,40,10\n"
        "b,2026,,54,53,20\n"
        "c,2026,,,,30\n")
    df = nfl_stats.fetch_player_stats(2026)
    for col in ("fg_made_list", "fg_blocked_list", "gwfg_distance_list"):
        assert df[col].dtype == object, col                                   # float64 here is what broke the load
    assert df["fg_blocked_list"].tolist()[:2] == ["23", "54"] and df["gwfg_distance_list"].tolist()[:2] == ["40", "53"]
    assert df["fg_blocked_list"].isna().tolist() == [False, False, True]     # a blank stays null; it is not the text "nan"
    assert df["attempts"].dtype != object                                     # only the list columns are forced to text


def test_a_column_with_no_values_at_all_is_still_text_not_float(tmp_path, monkeypatch):
    monkeypatch.setattr(nfl_stats, "CACHE", tmp_path)
    (tmp_path / "nfl_player_stats_2026.csv").write_text("player_id,fg_blocked_list\na,\nb,\n")
    df = nfl_stats.fetch_player_stats(2026)
    assert df["fg_blocked_list"].isna().all() and df["fg_blocked_list"].dtype == object
