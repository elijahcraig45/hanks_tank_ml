"""Regression tests for the 2026-09-25 power-rankings audit.

One class per finding, each pinning the specific failure that was measured:
double-counted suspended games, bootstrap bands that disagreed with their own point
ranks, non-D1 teams on college boards, division movers rated with last season's
division, caches that never refreshed, and the empty NFL strength of record.
"""

import json
import os
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "cfb"))

from rankings import build, core, sources  # noqa: E402
from test_rankings import round_robin  # noqa: E402


def mlb_game(pk, home, away, hs, as_, state="Final", code="F"):
    return {
        "gamePk": pk,
        "status": {"detailedState": state, "codedGameState": code},
        "teams": {
            "home": {"team": {"id": home[0], "name": home[1]}, "score": hs},
            "away": {"team": {"id": away[0], "name": away[1]}, "score": as_},
        },
    }


ATL, SF, PHI = (144, "Atlanta Braves"), (137, "San Francisco Giants"), (143, "Philadelphia Phillies")


class MlbGameIdentityTests(unittest.TestCase):
    def payload(self):
        return {"dates": [
            # Suspended on the 16th, resumed and finished on the 17th: StatsAPI lists
            # the same gamePk on both dates, each showing the final score.
            {"date": "2026-06-16", "games": [mlb_game(824912, SF, ATL, 2, 7)]},
            {"date": "2026-06-17", "games": [
                mlb_game(824912, SF, ATL, 2, 7),
                mlb_game(824913, SF, ATL, 3, 1),
            ]},
            # Called after becoming official — counts in the standings.
            {"date": "2026-06-18", "games": [
                mlb_game(824950, PHI, ATL, 4, 2, state="Completed Early"),
                mlb_game(824951, PHI, SF, 0, 0, state="Postponed", code="D"),
            ]},
        ]}

    def test_suspended_game_counts_once_on_its_resumption_date(self):
        rows = sources.mlb_rows(self.payload(), 2026)
        pks = [r["game_pk"] for r in rows]
        self.assertEqual(len(pks), len(set(pks)), "gamePk must be unique")
        suspended = [r for r in rows if r["game_pk"] == 824912]
        self.assertEqual(len(suspended), 1)
        self.assertEqual(suspended[0]["game_date"], "2026-06-17")

    def test_completed_early_is_a_decided_game(self):
        rows = sources.mlb_rows(self.payload(), 2026)
        self.assertIn(824950, [r["game_pk"] for r in rows])
        self.assertNotIn(824951, [r["game_pk"] for r in rows])

    def test_records_match_the_standings_arithmetic(self):
        rows = sources.mlb_rows(self.payload(), 2026)
        games = pd.DataFrame(rows).assign(week=1, neutral_site=0)
        rec = build._records(games)
        # ATL: won 824912, lost 824913 and 824950. Double counting made it 2-2.
        self.assertEqual(rec["Atlanta Braves"], [1, 2])
        self.assertEqual(rec["San Francisco Giants"], [1, 1])
        self.assertEqual(rec["Philadelphia Phillies"], [1, 0])

    def test_margin_is_home_minus_away(self):
        rows = {r["game_pk"]: r for r in sources.mlb_rows(self.payload(), 2026)}
        self.assertEqual(rows[824912]["margin"], -5)


class CacheRefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _age(self, path, hours):
        t = time.time() - hours * 3600
        os.utime(path, (t, t))

    def test_mlb_in_progress_season_refetches_when_stale(self):
        cached = self.dir / "mlb_games_2026.json"
        old = {"dates": [{"date": "2026-04-01", "games": [
            mlb_game(1, SF, ATL, 1, 0), mlb_game(2, SF, ATL, None, None, "Scheduled", "S"),
        ]}]}
        cached.write_text(json.dumps(old))
        self._age(cached, 48)
        new = {"dates": [{"date": "2026-04-01", "games": [
            mlb_game(1, SF, ATL, 1, 0), mlb_game(2, SF, ATL, 0, 5),
        ]}]}
        with unittest.mock.patch.object(sources, "CACHE", self.dir), \
             unittest.mock.patch.object(sources, "_curl",
                                        return_value=json.dumps(new).encode()) as curl:
            payload = sources._mlb_payload(2026)
        curl.assert_called_once()
        self.assertEqual(payload, new)

    def test_mlb_finished_season_is_never_refetched(self):
        cached = self.dir / "mlb_games_2024.json"
        done = {"dates": [{"date": "2024-09-29", "games": [mlb_game(1, SF, ATL, 1, 0)]}]}
        cached.write_text(json.dumps(done))
        self._age(cached, 24 * 300)  # fetched months after the season ended
        with unittest.mock.patch.object(sources, "CACHE", self.dir), \
             unittest.mock.patch.object(sources, "_curl") as curl:
            sources._mlb_payload(2024)
        curl.assert_not_called()

    def test_mlb_fetch_failure_falls_back_to_cache(self):
        cached = self.dir / "mlb_games_2026.json"
        cached.write_text(json.dumps({"dates": []}))
        self._age(cached, 48)
        with unittest.mock.patch.object(sources, "CACHE", self.dir), \
             unittest.mock.patch.object(sources, "_curl", side_effect=RuntimeError("down")):
            self.assertEqual(sources._mlb_payload(2026), {"dates": []})

    def test_nfl_stale_cache_is_refetched(self):
        cache = self.dir / "nfl_games.csv"
        cache.write_text("game_type,result,location,home_team,away_team,gameday,season,week\n")
        self._age(cache, 48)
        body = (b"game_type,result,location,home_team,away_team,gameday,season,week\n"
                b"REG,3,Home,KC,BUF,2026-09-10,2026,1\n")
        with unittest.mock.patch.object(sources, "NFL_CACHE", cache), \
             unittest.mock.patch.object(sources, "_curl", return_value=body) as curl:
            games = sources.load_nfl()
        curl.assert_called_once()
        self.assertEqual(len(games), 1)
        self.assertEqual(games["margin"].iloc[0], 3)

    def test_nfl_playoff_rounds_are_loaded(self):
        cache = self.dir / "nfl_games.csv"
        rows = ["game_type,result,location,home_team,away_team,gameday,season,week"]
        for i, kind in enumerate(("REG", "WC", "DIV", "CON", "SB")):
            loc = "Neutral" if kind == "SB" else "Home"
            rows.append(f"{kind},{i + 1},{loc},KC,BUF,2026-01-{10 + i},2025,{18 + i}")
        cache.write_text("\n".join(rows) + "\n")
        with unittest.mock.patch.object(sources, "NFL_CACHE", cache), \
             unittest.mock.patch.object(sources, "_curl") as curl:
            games = sources.load_nfl()
        curl.assert_not_called()
        self.assertEqual(sorted(games["week"]), [18, 19, 20, 21, 22])
        self.assertEqual(int(games.loc[games["week"] == 22, "neutral_site"].iloc[0]), 1)

    def test_nfl_fresh_cache_is_reused(self):
        cache = self.dir / "nfl_games.csv"
        cache.write_text("game_type,result,location,home_team,away_team,gameday,season,week\n"
                         "REG,3,Home,KC,BUF,2026-09-10,2026,1\n")
        with unittest.mock.patch.object(sources, "NFL_CACHE", cache), \
             unittest.mock.patch.object(sources, "_curl") as curl:
            sources.load_nfl()
        curl.assert_not_called()

    def test_cfb_in_progress_season_is_refetched_from_a_stale_cache(self):
        import cfb_config
        import espn_data

        season = cfb_config.CTX.season
        cache = self.dir / "cfb_games_v2.parquet"
        pd.DataFrame({
            "game_id": [str(i) for i in range(150)], "season": season, "week": 1,
            "division": "fbs", "game_date": pd.Timestamp(f"{season}-09-01"),
            "cross_division": 0,
        }).to_parquet(cache)
        self._age(cache, 48)
        calls = []

        def fake_week(s, w, group, seasontype):
            calls.append(s)
            return []

        with unittest.mock.patch.object(espn_data, "GAMES_CACHE", cache), \
             unittest.mock.patch.object(espn_data, "RAW_CACHE", self.dir), \
             unittest.mock.patch.object(espn_data, "_fetch_week", side_effect=fake_week), \
             unittest.mock.patch.object(espn_data, "_probe_transport", return_value=True):
            espn_data.fetch_history(first_season=season, last_season=season,
                                    divisions=("fbs",), workers=1)
        self.assertTrue(calls, "the in-progress season must be refetched")


class CrossDivisionFlagTests(unittest.TestCase):
    def test_cached_cross_flags_survive_a_retag(self):
        """Appending fresh games re-ran the both-feeds test over cached rows that had
        already been reduced to one sighting, zeroing every cached cross-division
        flag. That is what put 24 FCS teams, Delaware State among them, on the 2026
        FBS board once the rankings refresh appended 2025 to a cached 2026."""
        import espn_data

        cached = pd.DataFrame([
            {"game_id": "1", "division": "fbs", "cross_division": 1},
            {"game_id": "2", "division": "fcs", "cross_division": 0},
        ])
        fresh = pd.DataFrame([
            {"game_id": "3", "division": "fbs"}, {"game_id": "3", "division": "fcs"},
        ])
        out = espn_data.tag_cross_division(pd.concat([cached, fresh], ignore_index=True))
        flags = dict(zip(out["game_id"], out["cross_division"]))
        self.assertEqual(flags, {"1": 1, "2": 0, "3": 1})


class CollegeMembershipTests(unittest.TestCase):
    CONF = {"24": "fcs", "12": "fbs", "48": "fcs"}

    def chunk(self):
        return pd.DataFrame([
            # FBS-feed-only listing of an FBS v FCS game (the Delaware State case).
            {"home_team": "DEL", "away_team": "DSU", "home_conference_id": "12",
             "away_conference_id": "24"},
            {"home_team": "DSU", "away_team": "ALB", "home_conference_id": "24",
             "away_conference_id": "48"},
            # D2 opponent, listed in the FCS feed.
            {"home_team": "DSU", "away_team": "BOWIE", "home_conference_id": "24",
             "away_conference_id": "104"},
            {"home_team": "DSU", "away_team": "SCSU", "home_conference_id": "24",
             "away_conference_id": None},
        ])

    def test_division_comes_from_conference_membership(self):
        div = sources.cfb_team_divisions(self.chunk(), self.CONF)
        self.assertEqual(div["DEL"], "fbs")
        self.assertEqual(div["DSU"], "fcs")
        self.assertEqual(div["ALB"], "fcs")
        self.assertIsNone(div["BOWIE"])
        self.assertIsNone(div["SCSU"])

    def test_non_d1_teams_are_fitted_but_never_boarded(self):
        games = round_robin(["F1", "F2", "C1", "C2"], season=2026)
        side = {"F1": "fbs", "F2": "fbs", "C1": "fcs", "C2": "fcs"}
        games["home_division"] = games["home_team_name"].map(side)
        games["away_division"] = games["away_team_name"].map(side)
        d2 = pd.DataFrame([{
            "season": 2026, "week": 50, "game_date": pd.Timestamp("2026-11-01"),
            "home_team_name": "C1", "away_team_name": "D2 School", "home_won": 1,
            "neutral_site": 0, "home_division": "fcs", "away_division": None,
        }])
        games = pd.concat([games, d2], ignore_index=True)
        table, _ = build.build_board("cfb", 2026, n_boot=5, use_prior=False,
                                     games=games, with_fpi=False)
        self.assertNotIn("D2 School", set(table["team"]))
        self.assertEqual(set(table["team"]), set(side))

    def test_current_season_decides_the_board(self):
        prior = pd.DataFrame([{"home_team_name": "X", "away_team_name": "Y",
                               "home_division": "fcs", "away_division": "fcs"}])
        current = pd.DataFrame([{"home_team_name": "X", "away_team_name": "Z",
                                 "home_division": "fbs", "away_division": None}])
        m = build.board_membership(prior, current)
        self.assertEqual(m, {"X": "fbs", "Y": "fcs", "Z": None})


class DivisionMoverTests(unittest.TestCase):
    def seasons(self):
        prior = round_robin(["Riser", "c1", "c2"], season=2025,
                            strength={"Riser": 3, "c1": 2, "c2": 1})
        prior["home_division"] = prior["away_division"] = "fcs"
        current = round_robin(["Riser", "F1", "F2"], season=2026,
                              strength={"F1": 3, "Riser": 2, "F2": 1})
        current["home_division"] = current["away_division"] = "fbs"
        bridge = pd.DataFrame([{
            "season": 2026, "week": 90, "game_date": pd.Timestamp("2026-11-01"),
            "home_team_name": "F2", "away_team_name": "c1", "home_won": 1,
            "neutral_site": 0, "home_division": "fbs", "away_division": "fcs",
        }])
        return prior, pd.concat([current, bridge], ignore_index=True)

    def test_mover_is_tagged_by_its_current_division(self):
        prior, current = self.seasons()
        self.assertEqual(core.season_divisions(prior, current)["Riser"], "fbs")
        # The old setdefault over prior-then-current kept "fcs".
        self.assertEqual(core.team_divisions(pd.concat([prior, current]))["Riser"], "fcs")

    def test_mover_rating_uses_current_division_regardless_of_frame_order(self):
        prior, current = self.seasons()
        kw = dict(C=1.0, w0=1.0, tau=8.0, major="fbs")
        s_default, _, gap = core.fit_with_prior(current, prior, 5, **kw)
        s_explicit, _, _ = core.fit_with_prior(
            current, prior, 5, divisions={"Riser": "fbs", "F1": "fbs", "F2": "fbs",
                                          "c1": "fcs", "c2": "fcs"}, **kw)
        pd.testing.assert_series_equal(s_default.sort_index(), s_explicit.sort_index())
        s_wrong, _, _ = core.fit_with_prior(
            current, prior, 5, divisions={"Riser": "fcs", "F1": "fbs", "F2": "fbs",
                                          "c1": "fcs", "c2": "fcs"}, **kw)
        self.assertGreater(gap, 0)
        self.assertGreater(s_default["Riser"] - s_default.mean(),
                           s_wrong["Riser"] - s_wrong.mean())


class BootstrapConsistencyTests(unittest.TestCase):
    def test_bands_bracket_point_ranks_with_a_decayed_prior(self):
        rng = np.random.default_rng(3)
        teams = [f"T{i}" for i in range(10)]
        true = {t: i * 0.25 for i, t in enumerate(teams)}
        # Last season ran in the OPPOSITE order: at full weight it would drag the
        # bootstrap far away from a point estimate that has mostly decayed it.
        flipped = {t: -v for t, v in true.items()}

        def season(year, strength, reps):
            g = round_robin(teams, season=year, repeats=reps)
            diff = g["home_team_name"].map(strength) - g["away_team_name"].map(strength)
            g["home_won"] = (rng.random(len(g)) < 1 / (1 + np.exp(-diff))).astype(int)
            g["week"] = np.arange(len(g)) // 10 + 1
            return g

        prior, current = season(2025, flipped, 2), season(2026, true, 2)
        week = int(current["week"].max()) + 1
        kw = dict(C=1.0, w0=1.0, tau=4.0)
        point, _, _ = core.fit_with_prior(current, prior, week, **kw)
        boot = core.bootstrap_ranks(current, prior, week, n_boot=60, **kw)
        point_rank = {t: i for i, t in enumerate(point.index, start=1)}
        off = [abs(point_rank[t] - boot.loc[t, "rank_p50"]) for t in teams]
        self.assertLessEqual(np.mean(off), 1.0)
        for t in teams:
            self.assertLessEqual(boot.loc[t, "rank_p05"], point_rank[t])
            self.assertGreaterEqual(boot.loc[t, "rank_p95"], point_rank[t])

    def test_board_bands_are_numbered_within_the_board(self):
        games = round_robin(["F1", "F2", "C1", "C2"], season=2026)
        side = {"F1": "fbs", "F2": "fbs", "C1": "fcs", "C2": "fcs"}
        games["home_division"] = games["home_team_name"].map(side)
        games["away_division"] = games["away_team_name"].map(side)
        table, _ = build.build_board("cfb", 2026, n_boot=10, use_prior=False,
                                     games=games, with_fpi=False)
        self.assertTrue((table["rank_p95"] <= 2).all())


class BoardColumnTests(unittest.TestCase):
    def test_board_carries_computed_at_date_and_strength_of_record(self):
        games = round_robin(["A", "B", "C", "D"], season=2026)
        table, meta = build.build_board("nfl", 2026, n_boot=5, use_prior=False,
                                        games=games, with_fpi=False)
        self.assertIn("computed_at", table.columns)
        self.assertEqual(str(table["as_of_date"].iloc[0]),
                         str(games["game_date"].max().date()))
        self.assertTrue(table["sor_rank"].notna().all())
        self.assertEqual(meta["sor_source"], "model")
        # D beats everyone, so it has the best record against this schedule.
        self.assertEqual(table.loc[table["sor_rank"] == 1, "team"].iloc[0], "D")


class MarginModelTests(unittest.TestCase):
    def games(self):
        g = round_robin(["A", "B", "C"], strength={"A": 1, "B": 2, "C": 3})
        # Everyone beats A by a lot and C beats B narrowly.
        g["margin"] = np.where(g["home_won"] == 1, 1, -1) * np.where(
            (g["home_team_name"] == "A") | (g["away_team_name"] == "A"), 20, 2)
        return g

    def test_margin_model_orders_by_margin_and_scale_sets_confidence(self):
        g = self.games()
        s, _, _ = core.fit_ratings(g, model="margin", margin_alpha=1.0, margin_scale=10.0)
        self.assertEqual(list(s.index), ["C", "B", "A"])
        wide, _, _ = core.fit_ratings(g, model="margin", margin_alpha=1.0,
                                      margin_scale=20.0)
        self.assertAlmostEqual(wide["C"] * 2, s["C"], places=6)
        # B-A gap (a blowout) is far larger than C-B (a narrow edge).
        self.assertGreater(s["B"] - s["A"], 3 * (s["C"] - s["B"]))

    def test_cap_limits_blowouts(self):
        g = self.games()
        free, _, _ = core.fit_ratings(g, model="margin", margin_alpha=1.0)
        capped, _, _ = core.fit_ratings(g, model="margin", margin_alpha=1.0,
                                        margin_cap=5.0)
        self.assertLess(capped["B"] - capped["A"], free["B"] - free["A"])

    def test_blend_sits_between_its_parts(self):
        g = self.games()
        kw = dict(C=1.0, margin_alpha=1.0, margin_scale=10.0)
        bt, _, _ = core.fit_ratings(g, model="bt", **kw)
        mg, _, _ = core.fit_ratings(g, model="margin", **kw)
        bl, _, _ = core.fit_ratings(g, model="blend", blend=0.25, **kw)
        for t in "ABC":
            self.assertAlmostEqual(bl[t], 0.25 * bt[t] + 0.75 * mg[t], places=6)


if __name__ == "__main__":
    unittest.main()
