"""Tests for the ESPN FPI game-prediction snapshot.

What is pinned: the parser reads a real (trimmed) core-API predictor payload; the
projection is renormalised over no-tie; side swaps flip every home-oriented field; and —
the property the model comparison page's scoreboard depends on — nothing captured at or
after kickoff can ever carry a pregame timestamp. Nothing here touches ESPN or BigQuery.
"""

import json
import os
import sys
import unittest.mock
from datetime import datetime, timezone

import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from rankings import fpi_games as fg  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "espn_predictor_nfl_401872656.json")
NOW = datetime(2026, 9, 25, 18, 0, tzinfo=timezone.utc)


@pytest.fixture
def payload():
    with open(FIXTURE) as fh:
        return json.load(fh)


def fake_parsed(p=0.6, margin=3.0, home_id="26", away_id="17"):
    return {
        "home_win_probability": p, "predicted_home_margin": margin,
        "home_game_projection": 100 * p, "away_game_projection": 100 * (1 - p),
        "matchup_quality": 50.0, "espn_home_team_id": home_id,
        "espn_away_team_id": away_id, "espn_last_modified": "2026-09-24T10:00Z",
    }


def slate(kickoffs):
    return pd.DataFrame({
        "game_id": [f"g{i}" for i in range(len(kickoffs))],
        "espn_event_id": [str(1000 + i) for i in range(len(kickoffs))],
        "season": 2026, "week": 4, "division": None,
        "home_team": "SEA", "away_team": "NE",
        "kickoff": pd.to_datetime(kickoffs, utc=True),
    })


# ------------------------------------------------------------------- parsing
def test_parses_real_payload(payload):
    out = fg.parse_predictor(payload)
    # 61.112 / (61.112 + 38.6...) — renormalised, so home + away == 1 exactly.
    home = next(s["value"] for s in payload["homeTeam"]["statistics"]
                if s["name"] == "gameProjection")
    away = next(s["value"] for s in payload["awayTeam"]["statistics"]
                if s["name"] == "gameProjection")
    assert out["home_win_probability"] == pytest.approx(home / (home + away))
    assert out["espn_home_team_id"] == "26"
    assert out["espn_away_team_id"] == "17"
    assert out["predicted_home_margin"] is not None
    # Home favoured -> positive margin, the stack's spread convention.
    assert (out["predicted_home_margin"] > 0) == (out["home_win_probability"] > 0.5)


def test_renormalises_tie_share():
    p = {"homeTeam": {"statistics": [{"name": "gameProjection", "value": 60.0}]},
         "awayTeam": {"statistics": [{"name": "gameProjection", "value": 39.6}]}}
    assert fg.parse_predictor(p)["home_win_probability"] == pytest.approx(60 / 99.6)


def test_half_filled_block_is_rejected():
    p = {"homeTeam": {"statistics": [{"name": "gameProjection", "value": 60.0}]},
         "awayTeam": {"statistics": []}}
    assert fg.parse_predictor(p) is None
    assert fg.parse_predictor({}) is None


def test_orient_swaps_every_home_field():
    out = fg.orient(fake_parsed(0.7, 5.0, home_id="26", away_id="17"), expected_home_id="17")
    assert out["home_win_probability"] == pytest.approx(0.3)
    assert out["predicted_home_margin"] == pytest.approx(-5.0)
    assert out["home_game_projection"] == pytest.approx(30.0)
    assert out["espn_home_team_id"] == "17"


def test_orient_leaves_matching_or_unknown_alone():
    base = fake_parsed(0.7)
    assert fg.orient(base, "26") is base
    assert fg.orient(base, None) is base
    assert fg.orient(base, "999") is base  # neither side matches: do not guess


# ------------------------------------------------------ the pregame guarantee
def test_snapshot_drops_started_and_finished_games():
    s = slate(["2026-09-25 17:00", "2026-09-25 18:00", "2026-09-26 16:00", None])
    with unittest.mock.patch.object(fg, "fetch_predictor", return_value=fake_parsed()):
        rows = fg.snapshot("nfl", s, now=NOW)
    # 17:00 has kicked off, 18:00 is kicking off now (not strictly before), no kickoff
    # means we cannot prove it is pregame. Only the Saturday game survives.
    assert rows["game_id"].tolist() == ["g2"]
    assert (rows["predicted_at"] < rows["kickoff"]).all()
    assert (rows["source"] == fg.SOURCE_PREGAME).all()


def test_backfill_rows_can_never_pass_as_pregame():
    s = slate(["2026-09-13 17:00", "2026-09-20 17:00", "2026-10-01 17:00"])
    with unittest.mock.patch.object(fg, "fetch_predictor", return_value=fake_parsed()):
        rows = fg.backfill("nfl", s, now=NOW)
    assert rows["game_id"].tolist() == ["g0", "g1"]  # the future game is not backfilled
    assert (rows["source"] == fg.SOURCE_BACKFILL).all()
    # The scoreboard's rule is predicted_at < kickoff; these fail it by construction.
    assert (rows["predicted_at"] >= rows["kickoff"]).all()


def test_missing_predictor_is_skipped_not_filled():
    s = slate(["2026-09-26 16:00", "2026-09-26 19:00"])
    with unittest.mock.patch.object(fg, "fetch_predictor",
                                    side_effect=[fake_parsed(), None]):
        rows = fg.snapshot("nfl", s, now=NOW)
    assert len(rows) == 1
    assert rows["home_win_probability"].notna().all()
    assert list(rows.columns) == fg.COLUMNS


# ------------------------------------------------------------------- slates
def test_nfl_slate_converts_eastern_kickoff_to_utc():
    sched = pd.DataFrame({
        "game_id": ["2026_04_PIT_CLE", "2026_03_ATL_GB", "2026_05_X_Y"],
        "season": 2026, "week": [4, 3, 5],
        "gameday": ["2026-10-01", "2026-09-24", "2026-10-11"],
        "gametime": ["20:15", "20:15", "13:00"],
        "home_team": ["CLE", "GB", "Y"], "away_team": ["PIT", "ATL", "X"],
        "espn": [401872999.0, 401872948.0, 401873100.0],
    })
    out = fg.nfl_slate(sched, now=NOW, horizon_days=8, team_ids={"CLE": "5"})
    # Thursday 20:15 EDT is 00:15 UTC Friday; last night's game is gone; the 11th is
    # beyond the 8-day horizon.
    assert out["game_id"].tolist() == ["2026_04_PIT_CLE"]
    assert out["kickoff"].iloc[0] == pd.Timestamp("2026-10-02 00:15", tz="UTC")
    assert out["espn_event_id"].iloc[0] == "401872999"
    assert out["espn_home_team_id"].iloc[0] == "5"


def test_cfb_slate_uses_espn_event_id_and_filters_played():
    sched = pd.DataFrame({
        "game_id": ["401858467", "401856766"], "season": 2026, "week": 4,
        "division": "fbs", "home_team": ["PUR", "TCU"], "away_team": ["ND", "UNC"],
        "game_date": pd.to_datetime(["2026-09-26 18:00", "2026-08-29 16:00"]),
    })
    out = fg.cfb_slate(sched, now=NOW)
    assert out["game_id"].tolist() == ["401858467"]
    assert out["espn_event_id"].tolist() == ["401858467"]
    assert str(out["kickoff"].dt.tz) == "UTC"


# ------------------------------------------------------------------- writing
def test_off_by_default(monkeypatch):
    monkeypatch.delenv("FPI_SNAPSHOT", raising=False)
    assert not fg.enabled({})
    assert fg.enabled({"fpi_snapshot": True})
    monkeypatch.setenv("FPI_SNAPSHOT", "1")
    assert fg.enabled({})


def test_write_never_creates_the_table():
    rows = pd.DataFrame([{c: None for c in fg.COLUMNS}])
    rows["predicted_at"] = pd.Timestamp(NOW)
    fake = unittest.mock.MagicMock()
    with unittest.mock.patch("google.cloud.bigquery.Client", return_value=fake):
        assert fg.write_bq(rows, "cfb") == 1
    _, table_id = fake.load_table_from_dataframe.call_args.args[:2]
    cfg = fake.load_table_from_dataframe.call_args.kwargs["job_config"]
    assert table_id == "hankstank.cfb_season.fpi_game_predictions"
    assert cfg.create_disposition == "CREATE_NEVER"
    assert cfg.write_disposition == "WRITE_APPEND"


def test_run_snapshot_is_never_fatal():
    steps = {}
    with unittest.mock.patch.object(fg, "snapshot", side_effect=RuntimeError("espn down")):
        fg.run_snapshot("nfl", slate(["2026-09-26 16:00"]), steps)
    assert "error" in steps["fpi_snapshot"]
