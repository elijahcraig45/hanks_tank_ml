"""sim_blend operational fixes from the 2026-09-26/27 weekend: idempotent writes,
content-keyed load jobs (dedupe under concurrency), the previous-game lineup fallback,
and the per-instance warm engine."""
import os
import sys
import threading
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from google.api_core import exceptions as gexc  # noqa: E402

from pa_sim import blend  # noqa: E402
from test_sim_distributions import FakeBQ, _patch_run, _slate  # noqa: E402

NOW = datetime(2026, 9, 20, 18, tzinfo=timezone.utc)


# ------------------------------------------------------------------ fake BigQuery load jobs
class Job:
    def __init__(self, error=None):
        self.state = "DONE"
        self.error_result = error

    def result(self):
        if self.error_result:
            raise RuntimeError(self.error_result["message"])
        return self


class LoadBQ:
    """Job ids are unique per project, as in BigQuery: a second load with an id already
    used raises 409 Conflict whatever its content. Thread-safe, like the real service."""

    def __init__(self, columns=None, fail_first=False):
        self.jobs, self.tables, self.lock = {}, {}, threading.Lock()
        self.columns, self.fail_first = columns, fail_first

    def get_table(self, ref):
        if self.columns is None:
            raise gexc.NotFound("no table")

        class T:
            schema = [type("F", (), {"name": c}) for c in self.columns]
        return T()

    def get_job(self, job_id):
        return self.jobs[job_id]

    def load_table_from_dataframe(self, df, ref, job_id=None, job_config=None):
        with self.lock:
            if job_id in self.jobs:
                raise gexc.Conflict(f"Already Exists: Job hankstank:US.{job_id}")
            if self.fail_first and not self.jobs:
                self.jobs[job_id] = Job({"message": "backend error"})
                return self.jobs[job_id]
            self.jobs[job_id] = Job()
            self.tables.setdefault(ref, []).append(df.copy())
            return self.jobs[job_id]

    def rows(self, ref):
        return sum(len(d) for d in self.tables.get(ref, []))


def _rows(pk=100, p=0.55, at=NOW):
    return [dict(game_pk=pk, game_date=date(2026, 9, 20), home_win_probability=p,
                 model_version="sim_blend_v2", predicted_at=at)]


def test_content_key_ignores_predicted_at_only():
    a = blend.content_key(_rows(at=NOW))
    assert a == blend.content_key(_rows(at=datetime(2026, 9, 20, 18, 6, tzinfo=timezone.utc)))
    assert a != blend.content_key(_rows(p=0.56))
    assert a != blend.content_key(_rows(pk=101))


def test_identical_recomputation_is_not_appended_twice():
    bq = LoadBQ()
    ref = f"{blend.PROJECT}.{blend.DATASET}.{blend.PRED_TABLE}"
    assert blend.append(bq, blend.PRED_TABLE, _rows(at=NOW)) == "ok"
    # the weekend's case: the twin task, 1-6 minutes later, same rows, new predicted_at
    later = datetime(2026, 9, 20, 18, 4, tzinfo=timezone.utc)
    assert blend.append(bq, blend.PRED_TABLE, _rows(at=later)) == "duplicate"
    assert bq.rows(ref) == 1
    assert blend.append(bq, blend.PRED_TABLE, _rows(p=0.61, at=later)) == "ok"   # new lineup
    assert bq.rows(ref) == 2


def test_concurrent_twins_write_exactly_once():
    bq = LoadBQ()
    ref = f"{blend.PROJECT}.{blend.DATASET}.{blend.DIST_TABLE}"
    results, start = [], threading.Barrier(8)

    def twin(i):
        start.wait()
        results.append(blend.append(bq, blend.DIST_TABLE, _rows(at=datetime(2026, 9, 20, 18, 0, i,
                                                                            tzinfo=timezone.utc))))
    ts = [threading.Thread(target=twin, args=(i,)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sorted(results) == ["duplicate"] * 7 + ["ok"]
    assert bq.rows(ref) == 1


def test_failed_prior_job_is_retried_under_a_new_id():
    bq = LoadBQ(fail_first=True)
    ref = f"{blend.PROJECT}.{blend.DATASET}.{blend.PRED_TABLE}"
    with pytest.raises(RuntimeError):
        blend.append(bq, blend.PRED_TABLE, _rows())
    assert bq.rows(ref) == 0
    assert blend.append(bq, blend.PRED_TABLE, _rows()) == "ok"       # Cloud Tasks retry
    assert bq.rows(ref) == 1


def test_columns_the_table_lacks_are_dropped():
    bq = LoadBQ(columns=["game_pk", "game_date", "home_win_probability", "model_version", "predicted_at"])
    rows = [dict(_rows()[0], lineup_source="previous_game")]
    assert blend.append(bq, blend.PRED_TABLE, rows) == "ok"
    df = bq.tables[f"{blend.PROJECT}.{blend.DATASET}.{blend.PRED_TABLE}"][0]
    assert "lineup_source" not in df.columns and len(df) == 1


def test_empty_rows_do_nothing():
    assert blend.append(LoadBQ(), blend.PRED_TABLE, []) == "empty"


# ------------------------------------------------------------------ run_slate skips written games
def test_game_already_in_every_table_is_skipped_before_any_work(monkeypatch):
    calls = _patch_run(monkeypatch, _slate(1))
    every = {t: {100} for t in (blend.PRED_TABLE, blend.PROPS_TABLE, blend.DIST_TABLE, blend.PLAYER_TABLE)}
    monkeypatch.setattr(blend, "existing_games", lambda *a: every)
    monkeypatch.setattr(blend, "warm_state", lambda *a: pytest.fail("must not build the engine"))
    out = blend.run_slate(date(2026, 9, 20), bq=FakeBQ(), n_episodes=300, game_pks=[100], now=NOW)
    assert out["status"] == "already_written" and calls == []


def test_force_resimulates_written_games(monkeypatch):
    calls = _patch_run(monkeypatch, _slate(1))
    monkeypatch.setattr(blend, "existing_games", lambda *a: pytest.fail("force skips the check"))
    out = blend.run_slate(date(2026, 9, 20), bq=FakeBQ(), n_episodes=300, game_pks=[100], now=NOW,
                          force=True)
    assert out["status"] == "ok" and len(calls) == 4


def test_a_run_that_died_between_tables_finishes_only_the_rest(monkeypatch):
    calls = _patch_run(monkeypatch, _slate(2))
    part = {blend.PRED_TABLE: {100, 101}, blend.PROPS_TABLE: {100, 101}, blend.DIST_TABLE: {100},
            blend.PLAYER_TABLE: set()}
    monkeypatch.setattr(blend, "existing_games", lambda *a: part)
    out = blend.run_slate(date(2026, 9, 20), bq=FakeBQ(), n_episodes=300, game_pks=[100, 101], now=NOW)
    written = {t: {r["game_pk"] for r in rows} for t, rows in calls}
    assert written == {blend.PRED_TABLE: set(), blend.PROPS_TABLE: set(), blend.DIST_TABLE: {101},
                       blend.PLAYER_TABLE: {100, 101}}
    assert out["writes"][blend.PRED_TABLE] == "empty" and out["writes"][blend.PLAYER_TABLE] == "ok"


def test_dry_run_never_checks_or_writes(monkeypatch):
    calls = _patch_run(monkeypatch, _slate(1))
    monkeypatch.setattr(blend, "existing_games", lambda *a: pytest.fail("dry run is read-only"))
    out = blend.run_slate(date(2026, 9, 20), dry_run=True, bq=FakeBQ(), n_episodes=300,
                          game_pks=[100], now=NOW)
    assert out["status"] == "dry_run" and calls == []


# ------------------------------------------------------------------ warm engine
def test_engine_is_built_once_per_date(monkeypatch):
    _patch_run(monkeypatch, _slate(1))
    built = []
    import pa_sim.v2_inputs as v2i
    monkeypatch.setattr(v2i, "load_inputs", lambda bq, t: built.append(t) or {})
    for pk in (100, 100, 100):
        out = blend.run_slate(date(2026, 9, 20), dry_run=True, bq=FakeBQ(), n_episodes=300,
                              game_pks=[pk], now=NOW)
    assert built == [date(2026, 9, 20)] and out["warm"] is True
    blend.run_slate(date(2026, 9, 21), dry_run=True, bq=FakeBQ(), n_episodes=300, now=NOW)
    assert built == [date(2026, 9, 20), date(2026, 9, 21)]
    monkeypatch.setenv("SIM_BLEND_WARM", "0")
    blend.run_slate(date(2026, 9, 21), dry_run=True, bq=FakeBQ(), n_episodes=300, now=NOW)
    assert len(built) == 3


def test_release_memory_is_safe_to_call():
    blend.release_memory()


# ------------------------------------------------------------------ lineup fallback
def _games_today(**kw):
    r = dict(game_pk=7, game_date=date(2026, 9, 27), game_time_utc=pd.Timestamp("2026-09-27T19:10Z"),
             home_team_id=1, away_team_id=2, home_team_name="H", away_team_name="A",
             home_starter_id=11, away_starter_id=22, home_starter_name="hs", away_starter_name="as")
    r.update(kw)
    return pd.DataFrame([r])


def _posted(side, n, pk=7):
    return pd.DataFrame([dict(game_pk=pk, team_type=side, batting_order=j + 1, player_id=(100 if side == "home" else 200) + j,
                              player_name=f"{side}{j}") for j in range(n)])


def _prev(team, base, n=9):
    return pd.DataFrame([dict(team_id=team, source_game_pk=6, batting_order=j + 1, player_id=base + j,
                              player_name=f"p{base + j}") for j in range(n)])


def test_fallback_fills_only_the_incomplete_side():
    long = pd.concat([_posted("home", 9), _posted("away", 4)])
    prev = pd.concat([_prev(1, 500), _prev(2, 600)])
    fb = blend.fallback_slate(_games_today(), long, prev, {7: 15}, {"H": "HH", "A": "AA"})
    r = fb.iloc[0]
    assert r.home_lineup == list(range(100, 109))            # posted, kept
    assert r.away_lineup == list(range(600, 609))            # previous game's
    assert r.lineup_source == "previous_game_away" and r.venue_id == 15 and r.home_ab == "HH"


def test_fallback_with_no_posted_lineup_uses_both_previous():
    fb = blend.fallback_slate(_games_today(), pd.DataFrame(), pd.concat([_prev(1, 500), _prev(2, 600)]), {}, {})
    assert fb.iloc[0].lineup_source == "previous_game" and fb.iloc[0].venue_id == -1


def test_fallback_never_invents_a_starter_or_a_short_lineup():
    prev = pd.concat([_prev(1, 500), _prev(2, 600)])
    assert blend.fallback_slate(_games_today(home_starter_id=np.nan), pd.DataFrame(), prev, {}, {}).empty
    assert blend.fallback_slate(_games_today(), pd.DataFrame(), _prev(1, 500), {}, {}).empty
    assert blend.fallback_slate(_games_today(), pd.DataFrame(),
                                pd.concat([_prev(1, 500), _prev(2, 600, n=8)]), {}, {}).empty


def test_fallback_rows_carry_their_source_into_the_prediction_row(monkeypatch):
    s = _slate(1)
    s["lineup_source"] = "previous_game"
    calls = _patch_run(monkeypatch, s)
    out = blend.run_slate(date(2026, 9, 20), bq=FakeBQ(), n_episodes=300, game_pks=[100], now=NOW,
                          lineup_fallback=True)
    pred = dict(calls)[blend.PRED_TABLE]
    assert pred[0]["lineup_source"] == "previous_game" and out["lineup_source"] == {100: "previous_game"}


# ------------------------------------------------------------------ starter K in game_props_sim
def test_props_starter_k_pmf_gets_the_starter_k_calibration():
    from test_sim_blend import _summary
    from pa_sim import dists
    c = blend.load_coefs()
    cal = blend.load_player_calibration()
    sm = _summary()
    _, raw = blend.game_rows(sm, _slate(), np.array([0.6, 0.45]), c, 3000, NOW)
    _, calp = blend.game_rows(sm, _slate(), np.array([0.6, 0.45]), c, 3000, NOW, cal=cal)
    want = dists.apply_calibration(sm["spk"][0, 1], blend._stat_cal(cal, "starter", "K"))
    assert calp[0]["home_starter_k_pmf"] == pytest.approx(list(want))
    assert sum(calp[0]["home_starter_k_pmf"]) == pytest.approx(1.0)
    assert calp[0]["home_starter_k_mean"] < raw[0]["home_starter_k_mean"]    # thinning x0.96
