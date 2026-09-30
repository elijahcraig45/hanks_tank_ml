"""Model control plane (ML side): parsing, fail-open read, pauses, pin, feature_set,
model_sha256 and tier overrides. No network or credentials: everything is injected."""
import hashlib
import json
import logging
import os
import pickle
import sys
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from google.cloud import bigquery

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cloud_function_main  # noqa: E402
import model_control  # noqa: E402
import predict_today_games as ptg  # noqa: E402

SHA = "a" * 64
URI = "gs://bucket/models/v10.pkl"


def row(target, **kw):
    base = {"sport": "mlb", "target": target}
    base.update(kw)
    return base


def state(*rows, available=True):
    return model_control.ControlState(list(rows), available=available)


# ---------------------------------------------------------------- parsing / validation
def test_defaults_for_empty_state():
    s = model_control.empty_state()
    assert s.available is False
    assert s.is_paused("v10") is False
    assert s.pin("v10") is None
    assert s.tiers("v10") is None


@pytest.mark.parametrize("value,expected", [("paused", True), (" PAUSED ", True), ("active", False),
                                            ("", False), (None, False), ("stop", False)])
def test_is_paused_only_on_explicit_paused(value, expected):
    assert state(row("v10", run_state=value)).is_paused("v10") is expected


def test_pause_is_per_key_and_star_rows_ignored():
    s = state(row("logit3", run_state="paused"), row("*", run_state="paused", banner="x"))
    assert s.is_paused("logit3") and not s.is_paused("v10") and not s.is_paused("*")


def test_pin_requires_role_live_uri_and_sha():
    ok = row("v10", role="live", artifact_uri=URI, artifact_sha256=SHA)
    assert state(ok).pin("v10") == (URI, SHA)
    assert state({**ok, "artifact_sha256": SHA.upper()}).pin("v10") == (URI, SHA)   # normalised
    for bad in (
        {**ok, "role": "shadow"}, {**ok, "role": None},
        {**ok, "artifact_uri": None}, {**ok, "artifact_sha256": None},
        {**ok, "artifact_uri": "https://x/y.pkl"}, {**ok, "artifact_uri": "gs://bucket"},
        {**ok, "artifact_sha256": "abc"}, {**ok, "artifact_sha256": "g" * 64},
    ):
        assert state(bad).pin("v10") is None, bad


@pytest.mark.parametrize("high,medium,expected", [
    ("0.7", "0.6", (0.7, 0.6)),
    (0.66, 0.55, (0.66, 0.55)),
    ("0.6", "0.7", None),       # medium above high
    ("0.6", "0.6", None),       # equal
    ("0.6", "0.5", None),       # medium must be > 0.5
    ("1", "0.6", None),         # high must be < 1
    ("0.7", None, None),        # both required
    (None, "0.6", None),
    ("abc", "0.6", None),
    ("nan", "0.6", None),
])
def test_tiers_validation(high, medium, expected):
    assert state(row("v10", tier_high=high, tier_medium=medium)).tiers("v10") == expected


def test_bad_row_types_do_not_poison_state():
    s = model_control.ControlState([object(), None, row("v10", run_state="paused")], available=True)
    assert s.is_paused("v10")


# ---------------------------------------------------------------- get_state (the read)
class _Job:
    def __init__(self, rows):
        self._rows = rows

    def result(self, timeout=None):
        _Job.timeout = timeout
        return iter(self._rows)


class _Client:
    def __init__(self, rows=None, exc=None):
        self.rows, self.exc, self.calls = rows or [], exc, []

    def query(self, sql, job_config=None):
        self.calls.append((sql, job_config))
        if self.exc:
            raise self.exc
        return _Job(self.rows)


def test_get_state_reads_once_with_parameterised_query_and_timeout():
    c = _Client([row("v10", run_state="paused"), row("*", banner="hi")])
    s = model_control.get_state("mlb", client=c)
    assert s.available and s.is_paused("v10")
    assert len(c.calls) == 1
    sql, cfg = c.calls[0]
    assert "`hankstank.control.model_control_current`" in sql
    assert "@sport" in sql and "mlb" not in sql
    assert cfg.query_parameters[0].name == "sport" and cfg.query_parameters[0].value == "mlb"
    assert _Job.timeout == model_control.QUERY_TIMEOUT_S


def test_get_state_dataset_from_env(monkeypatch):
    monkeypatch.setenv("CONTROL_DATASET", "control_test")
    c = _Client([])
    model_control.get_state("nfl", client=c)
    assert "hankstank.control_test.model_control_current" in c.calls[0][0]


def test_get_state_ignores_other_sports_rows():
    c = _Client([{"sport": "nfl", "target": "xgb", "run_state": "paused"}])
    assert not model_control.get_state("mlb", client=c).is_paused("xgb")


def test_get_state_fail_open_logs_one_warning(caplog):
    c = _Client(exc=RuntimeError("boom"))
    with caplog.at_level(logging.WARNING, logger="model_control"):
        s = model_control.get_state("mlb", client=c)
    assert s.available is False and not s.is_paused("v10") and s.pin("v10") is None
    assert [r.levelno for r in caplog.records].count(logging.WARNING) == 1


def test_get_state_empty_view_is_available_with_defaults():
    s = model_control.get_state("mlb", client=_Client([]))
    assert s.available and not s.is_paused("v10") and s.tiers("v10") is None


def test_get_state_result_error_fails_open():
    class Bad(_Client):
        def query(self, sql, job_config=None):
            class J:
                def result(self, timeout=None):
                    raise TimeoutError("slow")
            return J()
    assert model_control.get_state("mlb", client=Bad()).available is False


def test_get_state_kill_switch_never_builds_a_client():
    with patch.object(bigquery, "Client", side_effect=AssertionError("must not be built")):
        assert model_control.get_state("mlb").available is False   # MODEL_CONTROL_DISABLED=1


def test_get_state_client_construction_failure_fails_open(monkeypatch):
    monkeypatch.delenv("MODEL_CONTROL_DISABLED")
    with patch.object(bigquery, "Client", side_effect=RuntimeError("no credentials")):
        s = model_control.get_state("mlb")
    assert s.available is False and not s.is_paused("v10")


# ---------------------------------------------------------------- MLB cloud function pauses
class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    def get_json(self, silent=True):
        return self.payload


def call(payload):
    body, status, _ = cloud_function_main.daily_pipeline(FakeRequest(payload))
    return json.loads(body), status


def paused(*keys):
    return state(*[row(k, run_state="paused") for k in keys])


def test_predict_today_paused_returns_200_paused_and_touches_nothing():
    with patch.object(model_control, "get_state", return_value=paused("v10")), \
         patch.object(ptg, "DailyPredictor") as dp:
        out, status = call({"mode": "predict_today", "date": "2026-09-29", "game_pks": [1]})
    assert status == 200 and out["status"] == "ok"
    assert [{k: v for k, v in s.items() if k != "seconds"} for s in out["steps"]] == \
        [{"step": "predict_today", "status": "paused"}]
    dp.assert_not_called()          # no BigQuery client, no table check, no writes


def test_paused_step_writes_one_structured_log_line(capsys):
    with patch.object(model_control, "get_state", return_value=paused("v10")):
        call({"mode": "predict_today", "date": "2026-09-29"})
    events = [json.loads(l) for l in capsys.readouterr().out.splitlines()
              if '"model_paused"' in l]
    assert len(events) == 1 and events[0]["model"] == "v10"


def test_predict_today_pause_never_raises_even_with_broken_dependencies():
    with patch.object(model_control, "get_state", return_value=paused("v10")), \
         patch.object(ptg, "DailyPredictor", side_effect=RuntimeError("would 500")):
        _out, status = call({"mode": "predict_today", "date": "2026-09-29"})
    assert status == 200


def test_pa_sim_paused_writes_nothing():
    import pa_sim.pipeline as pipe
    with patch.object(model_control, "get_state", return_value=paused("pa_sim")), \
         patch.object(pipe, "run_slate") as rs:
        out, status = call({"mode": "pa_sim", "date": "2026-09-29"})
    assert status == 200 and out["steps"][0]["status"] == "paused"
    rs.assert_not_called()


def test_pa_sim_unpaused_success_behaviour_unchanged():
    import pa_sim.pipeline as pipe
    with patch.object(model_control, "get_state", return_value=state()), \
         patch.object(pipe, "run_slate", return_value={"step": "pa_sim", "games": 3}) as rs:
        out, status = call({"mode": "pa_sim", "date": "2026-09-29"})
    assert status == 200 and out["steps"][0]["games"] == 3
    rs.assert_called_once()


def test_logit3_and_sim_blend_paused_are_skipped_like_disabled():
    import logit3_shadow
    import pa_sim.blend as blend
    with patch.object(model_control, "get_state", return_value=paused("logit3", "sim_blend")), \
         patch.object(logit3_shadow, "run_slate") as l3, \
         patch.object(blend, "run_slate") as sb, patch.object(blend, "memory_ok") as mem:
        o1, s1 = call({"mode": "logit3", "date": "2026-09-29"})
        o2, s2 = call({"mode": "sim_blend", "date": "2026-09-29"})
    assert s1 == s2 == 200
    assert o1["steps"][0] == {"step": "logit3", "status": "paused", "seconds": o1["steps"][0]["seconds"]}
    assert o2["steps"][0]["status"] == "paused"
    l3.assert_not_called(); sb.assert_not_called(); mem.assert_not_called()


def test_only_the_paused_shadow_is_skipped():
    import logit3_shadow
    with patch.object(model_control, "get_state", return_value=paused("sim_blend")), \
         patch.object(logit3_shadow, "run_slate", return_value={"step": "logit3", "rows": 2}) as l3:
        out, status = call({"mode": "logit3", "date": "2026-09-29"})
    assert status == 200 and out["steps"][0]["rows"] == 2
    l3.assert_called_once()


def test_shadow_helper_still_swallows_failures_when_not_paused():
    with patch.object(model_control, "get_state", return_value=state()):
        r = cloud_function_main._shadow("logit3", lambda: 1 / 0)
    assert r["status"] == "error"


def test_control_read_failure_is_fail_open_and_predictor_gets_no_control():
    with patch.object(model_control, "get_state", side_effect=RuntimeError("bq down")), \
         patch.object(ptg, "DailyPredictor") as dp:
        dp.return_value.run_for_game_pks.return_value = {"games_predicted": 1}
        out, status = call({"mode": "predict_today", "date": "2026-09-29", "game_pks": [1]})
    assert status == 200 and out["steps"][0]["games_predicted"] == 1
    assert dp.call_args.kwargs["control"] is None


def test_control_is_read_once_per_request_across_all_steps():
    import logit3_shadow
    import pa_sim.blend as blend
    import pa_sim.pipeline as pipe
    gs = MagicMock(return_value=state())
    quiet = {n: patch.object(cloud_function_main, n, return_value={"step": n})
             for n in ("_run_lineup_fetch", "_run_matchup_features", "_run_v7_features",
                       "_run_v8_features", "_run_v10_features", "_run_scouting_reports")}
    started = [p.start() for p in quiet.values()]
    try:
        with patch.object(model_control, "get_state", gs), patch.object(ptg, "DailyPredictor") as dp, \
             patch.object(pipe, "run_slate", return_value={"step": "pa_sim"}), \
             patch.object(logit3_shadow, "run_slate", return_value={"step": "logit3"}), \
             patch.object(blend, "memory_ok", return_value=(True, 2048, 1024)), \
             patch.object(blend, "run_slate", return_value={"step": "sim_blend"}):
            dp.return_value.run_for_game_pks.return_value = {"games_predicted": 1}
            out, status = call({"mode": "pregame_v10", "date": "2026-09-29", "game_pks": [1],
                                "run_pa_sim": True, "run_logit3": True, "run_sim_blend": True})
    finally:
        for p in quiet.values():
            p.stop()
    assert status == 200
    assert {"predict_today", "pa_sim", "logit3", "sim_blend"} <= {s["step"] for s in out["steps"]}
    assert gs.call_count == 1
    # the predictor is handed that same state
    assert dp.call_args.kwargs["control"] is gs.return_value


# ---------------------------------------------------------------- predictor: pin, sha, feature_set, tiers
class M:
    """Picklable stand-in for a fitted estimator."""
    feature_names_in_ = np.array(["f"])

    def __init__(self, p=0.6):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]] * len(X))


def blob(payload) -> bytes:
    return pickle.dumps(payload)


def payload(**kw):
    base = {"model": M(), "features": ["f"], "fill_values": {}, "version": "v10-pinned"}
    base.update(kw)
    return base


def predictor(control=None, fallback_v4=False):
    p = object.__new__(ptg.DailyPredictor)
    p._control, p.fallback_v4 = control, fallback_v4
    p.model = p.scaler = p.feature_names = None
    p.fill_values, p.model_version, p.model_sha256 = {}, None, None
    p._is_v8 = p._is_v10 = False
    return p


class FakeGCS:
    """Stand-in for storage.Client: objects keyed by (bucket, path); missing -> raises."""
    def __init__(self, objects):
        self.objects, self.reads = objects, []

    def __call__(self, project=None):
        return self

    def bucket(self, name):
        self._b = name
        return self

    def blob(self, path):
        b = self._b

        class B:
            def download_as_bytes(_s):
                self.reads.append((b, path))
                if (b, path) not in self.objects:
                    raise FileNotFoundError(f"gs://{b}/{path}")
                return self.objects[(b, path)]
        return B()


@pytest.fixture
def chain(tmp_path, monkeypatch):
    """Point the whole local/GCS chain at a tmp dir holding a 'chain' artifact (v10 label)."""
    raw = blob(payload(version="v10", model_name="chain"))
    f = tmp_path / "v10.pkl"
    f.write_bytes(raw)
    monkeypatch.setattr(ptg, "V10_LOCAL", f)
    return raw


def pin_state(sha=SHA, uri=URI):
    return state(row("v10", role="live", artifact_uri=uri, artifact_sha256=sha))


def test_pin_match_loads_pinned_first_and_records_sha(chain, monkeypatch):
    raw = blob(payload(version="v10-pinned", model_name="ignored", feature_set="v10"))
    gcs = FakeGCS({("bucket", "models/v10.pkl"): raw})
    monkeypatch.setattr(ptg.storage, "Client", gcs)
    p = predictor(pin_state(sha=hashlib.sha256(raw).hexdigest()))
    p.load_model()
    assert gcs.reads == [("bucket", "models/v10.pkl")]
    assert p.model_version == "v10-pinned"                 # from the payload's `version`
    assert p.model_sha256 == hashlib.sha256(raw).hexdigest()
    assert p._is_v10 is True


def test_pinned_artifact_without_feature_set_warns(chain, monkeypatch, caplog):
    raw = blob(payload(version="v10"))
    monkeypatch.setattr(ptg.storage, "Client", FakeGCS({("bucket", "models/v10.pkl"): raw}))
    with caplog.at_level(logging.WARNING, logger=ptg.logger.name):
        predictor(pin_state(sha=hashlib.sha256(raw).hexdigest())).load_model()
    assert any("no valid `feature_set`" in r.getMessage() for r in caplog.records)


def test_pin_mismatch_raises_and_never_falls_back(chain, monkeypatch):
    raw = blob(payload())
    monkeypatch.setattr(ptg.storage, "Client", FakeGCS({("bucket", "models/v10.pkl"): raw}))
    p = predictor(pin_state(sha="b" * 64))
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        p.load_model()
    assert p.model is None                                  # chain artifact was NOT loaded


def test_pin_hash_checked_before_unpickle(chain, monkeypatch):
    evil = b"not a pickle at all"
    monkeypatch.setattr(ptg.storage, "Client", FakeGCS({("bucket", "models/v10.pkl"): evil}))
    with patch.object(ptg.pickle, "loads", side_effect=AssertionError("unpickled unverified bytes")):
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            predictor(pin_state(sha="c" * 64)).load_model()


def test_pin_missing_object_logs_error_and_uses_normal_chain(chain, monkeypatch, caplog):
    monkeypatch.setattr(ptg.storage, "Client", FakeGCS({}))
    p = predictor(pin_state())
    with caplog.at_level(logging.ERROR, logger=ptg.logger.name):
        p.load_model()
    assert any(r.levelno == logging.ERROR and "pinned model" in r.getMessage() for r in caplog.records)
    assert p.model_sha256 == hashlib.sha256(chain).hexdigest()     # chain artifact, its own sha
    assert p.model_version == "v10"


def test_no_pin_or_invalid_pin_or_fallback_v4_never_touch_gcs(chain, monkeypatch):
    gcs = FakeGCS({})
    monkeypatch.setattr(ptg.storage, "Client", gcs)
    for ctl, fb in ((None, False), (state(), False),
                    (state(row("v10", role="live", artifact_uri=URI, artifact_sha256="bad")), False)):
        p = predictor(ctl, fallback_v4=fb)
        p.load_model()
        assert p.model_version == "v10" and p._is_v10
    assert gcs.reads == []


def test_chain_load_is_identical_to_before_and_hashes_the_file(chain):
    p = predictor(None)
    p.load_model()
    assert (p.model_version, p._is_v10, p._is_v8) == ("v10", True, False)
    assert p.model_sha256 == hashlib.sha256(chain).hexdigest()


def test_gcs_chain_load_caches_original_bytes_so_warm_sha_matches(tmp_path, monkeypatch):
    raw = blob(payload(version="v10"))
    local = tmp_path / "sub" / "v10.pkl"
    monkeypatch.setattr(ptg, "V10_LOCAL", local)
    monkeypatch.setattr(ptg.storage, "Client", FakeGCS({("hanks_tank_data", ptg.V10_MODEL_GCS): raw}))
    cold = predictor(None)
    cold.load_model()
    warm = predictor(None)
    warm.load_model()                       # now served from the local cache
    assert cold.model_sha256 == warm.model_sha256 == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("extra,is_v10,is_v8", [
    ({"feature_set": "v8"}, False, True),                              # overrides version == v10
    ({"feature_set": "V10", "version": "custom", "model_name": "mystery"}, True, False),
    ({"feature_set": "v10", "version": "v8_final"}, True, False),
    ({"feature_set": "bogus"}, True, False),                           # invalid -> old detection
    ({}, True, False),                                                 # absent -> old detection
])
def test_feature_set_overrides_label_and_absence_keeps_old_behaviour(tmp_path, monkeypatch, extra, is_v10, is_v8):
    f = tmp_path / "v10.pkl"
    base = {"model": M(), "features": ["f"], "version": "v10"}
    f.write_bytes(blob({**base, **extra}))
    monkeypatch.setattr(ptg, "V10_LOCAL", f)
    p = predictor(None)
    p.load_model()
    assert (p._is_v10, p._is_v8) == (is_v10, is_v8)


def test_old_label_detection_unchanged_without_feature_set(tmp_path, monkeypatch):
    f = tmp_path / "v10.pkl"
    monkeypatch.setattr(ptg, "V10_LOCAL", f)
    f.write_bytes(blob({"model": M(), "features": ["f"], "model_name": "V8_nocat"}))
    p = predictor(None); p.load_model()
    assert (p._is_v10, p._is_v8, p.model_version) == (False, True, "V8_nocat")
    f.write_bytes(blob({"model": M(), "features": ["f"], "model_name": "v5_stack"}))
    p = predictor(None); p.load_model()
    assert (p._is_v10, p._is_v8) == (False, False)
    f.write_bytes(blob({"model": M(), "features": ["f"], "model_name": "my_v10_model", "version": "x"}))
    p = predictor(None); p.load_model()
    assert p._is_v10 is True and p.model_version == "x"


def test_plain_estimator_still_detected_as_v10(tmp_path, monkeypatch):
    f = tmp_path / "v10.pkl"
    f.write_bytes(blob(M()))
    monkeypatch.setattr(ptg, "V10_LOCAL", f)
    p = predictor(None); p.load_model()
    assert p._is_v10 and p.model_version == "v10"


def predict(p, prob):
    p.model, p.scaler, p.fill_values, p._is_v10 = M(prob), None, {}, True
    return p.predict_game({"home_team_name": "H", "away_team_name": "A"}, pd.DataFrame([{"f": 1.0}]))


@pytest.mark.parametrize("prob,tier", [(0.66, "high"), (0.60, "medium"), (0.55, "low"),
                                       (0.36, "high")])
def test_tiers_use_constants_without_control(prob, tier):
    assert predict(predictor(None), prob)["confidence_tier"] == tier


def test_tier_override_applies_for_the_production_key():
    ctl = state(row("v10", tier_high="0.70", tier_medium="0.60"))
    assert predict(predictor(ctl), 0.66)["confidence_tier"] == "medium"   # was high
    assert predict(predictor(ctl), 0.72)["confidence_tier"] == "high"
    assert predict(predictor(ctl), 0.58)["confidence_tier"] == "low"      # was medium
    # another model's tiers never apply to v10
    other = state(row("logit3", tier_high="0.70", tier_medium="0.60"))
    assert predict(predictor(other), 0.66)["confidence_tier"] == "high"


def test_invalid_tier_override_is_ignored():
    for bad in (dict(tier_high="0.55", tier_medium="0.60"), dict(tier_high="0.9"),
                dict(tier_high="x", tier_medium="y"), dict(tier_high="0.99", tier_medium="0.5")):
        assert predict(predictor(state(row("v10", **bad))), 0.66)["confidence_tier"] == "high"


def test_broken_control_object_falls_back_to_constants():
    class Boom:
        def tiers(self, k):
            raise RuntimeError("x")
        def pin(self, k):
            raise RuntimeError("x")
    p = predictor(Boom())
    assert predict(p, 0.66)["confidence_tier"] == "high"
    p.load_model  # noqa: B018  (pin errors are swallowed in _pin)
    assert p._pin() is None


# ---------------------------------------------------------------- model_sha256 column handling
class _LoadJob:
    state, error_result, errors = "DONE", None, None

    def result(self):
        return self


class WriterBQ:
    def __init__(self, columns=None, get_table_raises=False):
        self.columns, self.get_table_raises = columns, get_table_raises
        self.loads, self.deletes, self.get_table_calls = [], [], 0

    def get_table(self, ref):
        self.get_table_calls += 1
        if self.get_table_raises:
            raise RuntimeError("nope")
        t = MagicMock()
        t.schema = [bigquery.SchemaField(c, "STRING") for c in self.columns]
        return t

    def load_table_from_file(self, fh, table, job_id=None, job_config=None):
        self.loads.append({"job_id": job_id, "rows": [json.loads(l) for l in fh.read().decode().splitlines()],
                           "schema": [f.name for f in job_config.schema]})
        return _LoadJob()

    def query(self, sql, job_config=None):
        self.deletes.append(sql)
        return _LoadJob()


def rows(with_sha=True):
    r = {"game_pk": 1, "game_date": "2026-09-29", "home_win_probability": 0.55,
         "model_version": "v10", "predicted_at": "2026-09-29T17:40:08+00:00"}
    if with_sha:
        r["model_sha256"] = SHA
    return [r]


def writer(bq):
    p = object.__new__(ptg.DailyPredictor)
    p.bq, p.model_version, p.dry_run = bq, "v10", False
    return p


D = date(2026, 9, 29)


def test_schema_declares_nullable_model_sha256():
    f = {f.name: f for f in ptg.PREDICTIONS_SCHEMA}["model_sha256"]
    assert f.field_type == "STRING" and f.mode == "NULLABLE"


def test_sha_written_when_column_exists_rows_and_schema_consistent():
    bq = WriterBQ(columns=["game_pk", "model_sha256"])
    writer(bq)._write_predictions(rows(), D)
    (load,) = bq.loads
    assert "model_sha256" in load["schema"] and load["rows"][0]["model_sha256"] == SHA
    assert bq.get_table_calls == 1                                   # checked once per write


def test_sha_dropped_from_rows_and_schema_when_column_absent():
    bq = WriterBQ(columns=["game_pk"])
    writer(bq)._write_predictions(rows(), D)
    (load,) = bq.loads
    assert "model_sha256" not in load["schema"] and "model_sha256" not in load["rows"][0]
    assert len(load["schema"]) == len(ptg.PREDICTIONS_SCHEMA) - 1


def test_sha_dropped_when_get_table_raises():
    bq = WriterBQ(get_table_raises=True)
    writer(bq)._write_predictions(rows(), D)
    assert "model_sha256" not in bq.loads[0]["rows"][0] and "model_sha256" not in bq.loads[0]["schema"]


def test_sha_dropped_when_client_has_no_get_table():
    class NoGetTable:
        loads = []

        def load_table_from_file(self, fh, table, job_id=None, job_config=None):
            self.loads.append(([f.name for f in job_config.schema], fh.read().decode()))
            return _LoadJob()

        def query(self, sql, job_config=None):
            return _LoadJob()
    bq = NoGetTable()
    writer(bq)._write_predictions(rows(), D)
    names, body = bq.loads[0]
    assert "model_sha256" not in names and "model_sha256" not in body


def test_job_id_unchanged_for_rows_without_the_column():
    with_key, without_key = WriterBQ(columns=["game_pk"]), WriterBQ(columns=["game_pk"])
    writer(with_key)._write_predictions(rows(True), D)
    writer(without_key)._write_predictions(rows(False), D)
    assert with_key.loads[0]["job_id"] == without_key.loads[0]["job_id"]
    # and it differs once the column is really being written (content changed)
    present = WriterBQ(columns=["model_sha256"])
    writer(present)._write_predictions(rows(True), D)
    assert present.loads[0]["job_id"] != with_key.loads[0]["job_id"]


def test_ensure_table_never_alters_an_existing_table():
    p = object.__new__(ptg.DailyPredictor)
    p.dry_run = False
    p.bq = MagicMock()
    p._ensure_table()
    p.bq.create_table.assert_not_called()
    p.bq.query.assert_not_called()
    p.bq.update_table.assert_not_called()
