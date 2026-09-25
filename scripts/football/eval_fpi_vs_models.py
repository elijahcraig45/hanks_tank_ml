"""FPI vs margin ridge vs production XGBoost vs the market, on the same games. Read-only.

    python scripts/football/eval_fpi_vs_models.py --games-csv /path/to/nflverse/games.csv

Reads BigQuery (SELECT only), nflverse's games.csv and ESPN's core API. Writes nothing
except a local JSON cache of ESPN responses (--cache, default ./.fpi_eval_cache).

What each column is:
  xgb     production rows in {nfl,cfb}_season.game_predictions. For 2024-25 these are
          walk-forward BACKFILL rows (trained only on earlier weeks, but written after
          the fact); for 2026 they are the live rows, and only those with
          predicted_at < kickoff are kept.
  ridge   margin_ridge.walk_forward with the shipped config, refit per week on earlier
          games only. Information-honest, but computed now, not captured live.
  fpi     ESPN's predictor gameProjection, renormalised over no-tie. Fetched after the
          games; see the pregame check below for why it is usable anyway.
  market  NFL: de-vigged closing moneylines. CFB: Phi(spread / 15.5) from
          cfb_season.betting_lines (15.5 is the ridge's CFB sigma, tuned on 2022, not on
          these games; it is slightly under-confident for the market, so conservative).

Pregame check for FPI: compare gameProjection with ESPN's own in-game win probability at
the first play. A value revised after the game would drift away from it.

Paired bootstrap (4000 resamples of games) on per-game log-loss differences.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "nfl"))
sys.path.insert(0, str(SRC / "cfb"))

from rankings import fpi_games  # noqa: E402
from rankings.http import get_bytes  # noqa: E402

CORE = ("https://sports.core.api.espn.com/v2/sports/football/leagues/{lg}/events/{e}"
        "/competitions/{e}/{what}")
CFB_MARKET_SIGMA = 15.5
NFL_MARKET_SIGMA = 13.45  # spread fallback only, where a moneyline is missing


def per_game_ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_ci(d, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(bs, [2.5, 97.5])


def report(df, models, label, pairs):
    b = df.dropna(subset=models + ["home_won"])
    y = b["home_won"].to_numpy(float)
    print(f"\n### {label}  n={len(b)} (dropped {len(df) - len(b)} missing a model)")
    for m in models:
        p = b[m].to_numpy(float)
        print(f"  {m:8s} acc {np.mean((p > .5) == y):.3f}  logloss "
              f"{per_game_ll(y, p).mean():.4f}  brier {np.mean((p - y) ** 2):.4f}")
    for a, r in pairs:
        d = per_game_ll(y, b[a]) - per_game_ll(y, b[r])
        lo, hi = boot_ci(d)
        print(f"  {a} - {r}: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")


def fetch_json(cache: Path, lg: str, event: str, what: str, query: str = "") -> dict | None:
    path = cache / f"{lg}_{event}_{what}.json"
    if path.exists() and path.stat().st_size > 50:
        return json.loads(path.read_text())
    try:
        raw = get_bytes(CORE.format(lg=lg, e=event, what=what) + query, 30)
    except Exception:
        return None
    path.write_bytes(raw)
    return json.loads(raw)


def espn_frame(cache: Path, lg: str, events: list[str], first_play_every: int) -> pd.DataFrame:
    def one(i_e):
        i, e = i_e
        pred = fetch_json(cache, lg, e, "predictor")
        parsed = fpi_games.parse_predictor(pred) if pred else None
        row = {"eid": e}
        if parsed:
            row["fpi"] = parsed["home_win_probability"]
            row["fpi_margin"] = parsed["predicted_home_margin"]
        if first_play_every and i % first_play_every == 0:
            pr = fetch_json(cache, lg, e, "probabilities", "?limit=1") or {}
            items = pr.get("items") or []
            if items:
                row["first_play_wp"] = items[0].get("homeWinPercentage")
        return row

    with ThreadPoolExecutor(6) as ex:
        return pd.DataFrame(list(ex.map(one, enumerate(events))))


def pregame_check(df: pd.DataFrame, label: str) -> None:
    d = df.dropna(subset=["fpi", "first_play_wp"])
    diff = d["fpi"] - d["first_play_wp"]
    print(f"\nFPI vs ESPN first-play WP, {label}: n={len(d)}, median diff "
          f"{diff.median() * 100:+.2f} pts, |diff|>3 pts in "
          f"{(diff.abs() > .03).mean() * 100:.1f}% of games")


def devig(home_ml, away_ml):
    def imp(ml):
        ml = np.asarray(ml, float)
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
    h, a = imp(home_ml), imp(away_ml)
    return h / (h + a)


def run_nfl(bq, games_csv: str, cache: Path):
    import margin_ridge as mr
    import predict_nfl as pn

    g = pd.read_csv(games_csv, low_memory=False)
    g = g[g.season >= 1999].reset_index(drop=True)
    rg = pn.ridge_frame(g)
    test = (rg.season >= 2024).to_numpy()
    rg["ridge"] = mr.win_prob(mr.walk_forward(rg, test, mr.NFL_RIDGE), mr.NFL_RIDGE.sigma)

    n = g[(g.season >= 2024) & g.result.notna() & (g.result != 0)].copy()
    n["home_won"] = (n.result > 0).astype(int)
    n["market"] = devig(n.home_moneyline, n.away_moneyline)
    n["market"] = n["market"].fillna(pd.Series(mr.win_prob(n.spread_line, NFL_MARKET_SIGMA),
                                               index=n.index))
    local = pd.to_datetime(n.gameday + " " + n.gametime.fillna("13:00"))
    n["kickoff"] = local.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    n["eid"] = n.espn.astype("Int64").astype(str)
    n = n.merge(rg[["game_id", "ridge"]], on="game_id", how="left")
    n = n.merge(espn_frame(cache, "nfl", n.eid.tolist(), 1), on="eid", how="left")

    xp = bq.query("SELECT game_id, home_win_probability AS xgb, predicted_at "
                  "FROM `hankstank.nfl_season.game_predictions`").to_dataframe()
    n = n.merge(xp, on="game_id", how="left")
    live = n.season == 2026
    n.loc[live & ~(n.predicted_at < n.kickoff), "xgb"] = np.nan  # strict pregame

    pregame_check(n, "NFL")
    models = ["xgb", "ridge", "fpi", "market"]
    pairs = [("fpi", "xgb"), ("fpi", "ridge"), ("ridge", "xgb"), ("market", "fpi"),
             ("market", "ridge")]
    report(n[n.season.isin([2024, 2025])], models, "NFL 2024-25 (xgb = backfill)", pairs)
    report(n[n.season == 2025], models, "NFL 2025", pairs)
    report(n[live], models, "NFL 2026 live (xgb pregame only)", pairs)


def run_cfb(bq, cache: Path):
    import margin_ridge as mr
    import pipeline as cp

    cg = bq.query("SELECT * FROM `hankstank.cfb_historical.games`").to_dataframe()
    cg = cg[cg.home_won.notna()].reset_index(drop=True)
    crg = cp.ridge_frame(cg)
    test = (crg.season >= 2025).to_numpy()
    crg["ridge"] = mr.win_prob(mr.walk_forward(crg, test, mr.CFB_RIDGE),
                               mr.CFB_RIDGE.sigma)

    c = cg[(cg.season >= 2025) & (cg.division == "fbs")][
        ["game_id", "season", "week", "home_won", "game_date"]].copy()
    c = c.merge(crg[["game_id", "ridge"]], on="game_id", how="left")
    c = c.merge(espn_frame(cache, "college-football", c.game_id.tolist(), 4)
                .rename(columns={"eid": "game_id"}), on="game_id", how="left")
    lines = bq.query("SELECT game_id, ANY_VALUE(spread_line) AS spread_line "
                     "FROM `hankstank.cfb_season.betting_lines` GROUP BY game_id"
                     ).to_dataframe()
    c = c.merge(lines, on="game_id", how="left")
    c["market"] = mr.win_prob(c.spread_line, CFB_MARKET_SIGMA)
    xp = bq.query("SELECT game_id, home_win_probability AS xgb, predicted_at, "
                  "game_date AS kickoff FROM `hankstank.cfb_season.game_predictions`"
                  ).to_dataframe()
    c = c.merge(xp, on="game_id", how="left")
    live = c.season == 2026
    c.loc[live & ~(c.predicted_at < c.kickoff), "xgb"] = np.nan

    pregame_check(c, "CFB FBS")
    models = ["xgb", "ridge", "fpi", "market"]
    pairs = [("fpi", "xgb"), ("fpi", "ridge"), ("ridge", "xgb"), ("market", "fpi"),
             ("market", "ridge")]
    report(c[c.season == 2025], models, "CFB FBS 2025 (xgb = backfill)", pairs)
    for lo, hi in ((1, 4), (5, 20)):
        report(c[(c.season == 2025) & c.week.between(lo, hi)], models,
               f"CFB FBS 2025 weeks {lo}-{hi}", pairs[:3])
    report(c[live], models, "CFB FBS 2026 live (xgb pregame only)", pairs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-csv", required=True,
                    help="nflverse games.csv (github.com/nflverse/nfldata/raw/master/"
                         "data/games.csv) — carries the ESPN event ids and moneylines")
    ap.add_argument("--cache", default=".fpi_eval_cache")
    ap.add_argument("--sport", choices=["nfl", "cfb", "both"], default="both")
    args = ap.parse_args()

    from google.cloud import bigquery

    bq = bigquery.Client(project="hankstank")
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    if args.sport in ("nfl", "both"):
        run_nfl(bq, args.games_csv, cache)
    if args.sport in ("cfb", "both"):
        run_cfb(bq, cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
