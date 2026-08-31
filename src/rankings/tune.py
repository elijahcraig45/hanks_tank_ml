"""Measure the ridge strength and prior decay per sport instead of guessing them.

Walk-forward: for each week of a season, fit on everything strictly before it and score
that week's games by log loss. That is the honest test — it never lets the model see a
result it is being graded on.

The college constants were chosen this way; NFL and MLB have very different sample
sizes (17 and 162 games per team against college's 12, over 32 and 30 teams against
~380), so copying college's values across would be an assumption, not a measurement.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rankings import core, sources  # noqa: E402
from rankings.sources import SPORTS  # noqa: E402

logger = logging.getLogger(__name__)


def walk_forward(games: pd.DataFrame, season: int, C: float, tau: float,
                 w0: float = 1.0, major: str | None = None,
                 min_train: int = 40) -> tuple[float | None, int]:
    """Total log loss over a season's weeks, fitting only on earlier games."""
    current = games[games["season"] == season]
    prior = games[games["season"] == season - 1]
    if current.empty:
        return None, 0
    if prior.empty:
        prior = None

    losses, weights = [], []
    for week in sorted(current["week"].unique()):
        test = current[current["week"] == week]
        seen = current[current["week"] < week]
        if len(test) < 10:
            continue
        if prior is None and len(seen) < min_train:
            continue
        try:
            strengths, home_adv, _ = core.fit_with_prior(
                seen, prior, int(week), C=C, w0=w0, tau=tau, major=major
            )
        except Exception:
            continue
        loss, n = core.score_games(strengths, home_adv, test)
        if loss is not None:
            losses.append(loss * n)
            weights.append(n)

    if not weights:
        return None, 0
    return float(sum(losses) / sum(weights)), int(sum(weights))


def sweep(sport: str, seasons: list[int], cs: list[float], taus: list[float]) -> pd.DataFrame:
    spec = SPORTS[sport]
    load_seasons = tuple(sorted({s for y in seasons for s in (y, y - 1)}))
    games = sources.load(sport, load_seasons if sport == "mlb" else None)

    rows = []
    for C in cs:
        for tau in taus:
            per_season, total_n = [], 0
            for season in seasons:
                loss, n = walk_forward(games, season, C, tau, major=spec.major_division)
                if loss is not None:
                    per_season.append(loss * n)
                    total_n += n
            if total_n:
                rows.append({
                    "C": C, "tau": tau,
                    "log_loss": round(sum(per_season) / total_n, 4),
                    "games": total_n,
                })
                logger.info("C=%s tau=%s -> %.4f (%d games)",
                            C, tau, rows[-1]["log_loss"], total_n)
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", required=True, choices=sorted(SPORTS))
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--cs", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--taus", type=float, nargs="+", default=[2.0, 4.0, 8.0, 12.0, 40.0])
    args = ap.parse_args()

    out = sweep(args.sport, args.seasons, args.cs, args.taus)
    print()
    print(f"WALK-FORWARD LOG LOSS — {SPORTS[args.sport].label}, seasons {args.seasons}")
    print(out.to_string(index=False))
    if not out.empty:
        best = out.iloc[0]
        print(f"\nbest: C={best.C} tau={best.tau} -> {best.log_loss}")
        cur = SPORTS[args.sport]
        print(f"currently configured: C={cur.ridge_C} tau={cur.prior_tau}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
