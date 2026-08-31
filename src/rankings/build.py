"""Build power-ranking boards for any supported sport.

    python -m rankings.build --sport nfl --season 2026 --dry-run
    python -m rankings.build --sport cfb --season 2026 --write-bq

One joint Bradley-Terry fit per sport produces a rating for every team; the board is
then split per division where the sport has them. Ranks are numbered WITHIN a board so
the FCS page reads 1..N rather than starting at 140, while `overall_rank` and the shared
`rating` keep the two ladders comparable — which is the whole point of fitting them
together.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rankings import core, fpi, sources  # noqa: E402
from rankings.sources import SPORTS  # noqa: E402

logger = logging.getLogger(__name__)


def _records(games: pd.DataFrame) -> dict[str, list[int]]:
    rec: dict[str, list[int]] = {}
    for g in games.itertuples(index=False):
        winner, loser = (
            (g.home_team_name, g.away_team_name) if g.home_won
            else (g.away_team_name, g.home_team_name)
        )
        rec.setdefault(winner, [0, 0])[0] += 1
        rec.setdefault(loser, [0, 0])[1] += 1
    return rec


def build_board(sport: str, season: int, week: int | None = None,
                n_boot: int = 200, use_prior: bool = True,
                games: pd.DataFrame | None = None,
                with_fpi: bool = True) -> tuple[pd.DataFrame, dict]:
    spec = SPORTS[sport]
    if games is None:
        # MLB is fetched per season, so ask for this one and last for the prior.
        seasons = (season, season - 1) if sport == "mlb" else None
        games = sources.load(sport, seasons)

    if games.empty:
        raise SystemExit(f"no games available for {sport}")

    current = games[games["season"] == season].copy()
    if week is not None:
        current = current[current["week"] < week]
        effective_week = week
    elif not current.empty:
        effective_week = int(current["week"].max())
    else:
        # Preseason: nothing played yet, so the board is week 0 of this season.
        effective_week = 0

    prior = games[games["season"] == season - 1] if use_prior else None
    if prior is not None and prior.empty:
        prior = None

    if prior is None and current.empty:
        raise SystemExit(
            f"no games for {sport} {season} and no prior season to fall back on"
        )

    major = spec.major_division
    strengths, home_adv, div_gap = core.fit_with_prior(
        current, prior, effective_week,
        C=spec.ridge_C, w0=spec.prior_w0, tau=spec.prior_tau, major=major,
    )

    # Bootstrap over the same evidence the point estimate used, prior included, so the
    # ranges describe the actual information rather than a different sample.
    pool = pd.concat(
        [f for f in (prior, current) if f is not None and not f.empty], ignore_index=True
    )
    divisions = core.team_divisions(pool)
    boot = core.bootstrap_ranks(
        pool, n_boot=n_boot, C=spec.ridge_C, divisions=divisions, major=major
    )

    # In preseason there is no current-season record to show, and "0-0" against a
    # fully-formed rating reads as a bug. Fall back to last season's, which is exactly
    # what the rating is built from; `record_season` tells the UI which one it got.
    record = _records(current) if not current.empty else _records(prior)
    record_season = season if not current.empty else season - 1

    # Board membership needs a division for EVERY team in the fit, but early in a
    # season only a handful have played. Start from the prior season so nobody is
    # missing, then let this season's games overwrite it — that ordering is what makes
    # a team who moved up appear on the right board from week one.
    division_of = dict(core.team_divisions(prior)) if prior is not None else {}
    division_of.update(core.team_divisions(current))
    if not division_of:
        division_of = divisions

    overall = {team: i for i, team in enumerate(strengths.index, start=1)}

    boards = spec.board_divisions or (None,)
    rows: list[dict] = []
    for board in boards:
        members = [
            t for t in strengths.index
            if board is None or division_of.get(t) == board
        ]
        if not members:
            continue
        ladder = strengths.loc[members]
        for rank, team in enumerate(ladder.index, start=1):
            rating = float(ladder.loc[team])
            nxt = float(ladder.iloc[rank]) if rank < len(ladder) else None
            wins, losses = record.get(team, [0, 0])
            rows.append({
                "season": season,
                "as_of_week": effective_week,
                "division": board,
                "rank": rank,
                "overall_rank": overall[team],
                "team": team,
                "record": f"{wins}-{losses}",
                "record_season": record_season,
                "rating": round(rating, 1),
                "gap_to_next": round(rating - nxt, 1) if nxt is not None else None,
                "p_beat_next": (
                    round(core.win_prob(rating, nxt), 3) if nxt is not None else None
                ),
                "p_beat_last": round(
                    core.win_prob(rating, float(ladder.iloc[-1])), 3
                ),
                "rank_p05": int(boot.loc[team, "rank_p05"]) if team in boot.index else None,
                "rank_p50": int(boot.loc[team, "rank_p50"]) if team in boot.index else None,
                "rank_p95": int(boot.loc[team, "rank_p95"]) if team in boot.index else None,
            })

    is_preseason = current.empty
    meta = {
        "sport": sport,
        "season": season,
        "as_of_week": effective_week,
        "games_current": int(len(current)),
        "games_prior": int(len(prior)) if prior is not None else 0,
        "teams": int(len(strengths)),
        "home_field_points": round(home_adv, 1),
        "division_gap_points": round(div_gap, 1) if major else None,
        "ridge_C": spec.ridge_C,
        "prior_tau": spec.prior_tau,
        "n_boot": n_boot,
        "prior_weight": (
            1.0 if is_preseason
            else round(core.prior_weight(effective_week, spec.prior_w0, spec.prior_tau), 3)
            if prior is not None else 0.0
        ),
        "is_preseason": bool(is_preseason),
        "record_season": record_season,
    }

    out = pd.DataFrame(rows)
    for key in ("prior_weight", "home_field_points"):
        out[key] = meta[key]

    # Strength of record, strength of schedule and unit efficiency come from ESPN's
    # FPI. They answer questions this rating deliberately does not model, so they are
    # attached for display rather than fed back into the fit — the board stays
    # independent of ESPN's own rating.
    if with_fpi:
        out = fpi.attach(out, sport, season)
        meta["has_fpi"] = bool("fpi" in out.columns and out["fpi"].notna().any())
    else:
        meta["has_fpi"] = False

    return out, meta


def print_board(table: pd.DataFrame, meta: dict, top: int = 25) -> None:
    spec = SPORTS[meta["sport"]]
    print()
    header = f"POWER RANKINGS — {spec.label} {meta['season']}"
    if meta["is_preseason"]:
        header += "  [PRESEASON: no games played, rating is entirely last season]"
    print(header)
    print(
        f"{meta['games_current']} games this season, {meta['games_prior']} carried from "
        f"last at weight {meta['prior_weight']:.2f} | home field "
        f"{meta['home_field_points']} pts"
        + (f" | division gap {meta['division_gap_points']} pts"
           if meta["division_gap_points"] else "")
    )
    print("=" * 96)

    for board, group in table.groupby("division", dropna=False):
        if pd.notna(board):
            print(f"\n-- {str(board).upper()} --")
        print(f"{'#':>3} {'ovr':>4} {'team':<34} {'rec':>7} {'rating':>8} "
              f"{'gap':>6} {'P(beat next)':>12} {'rank 5-95%':>11}")
        print("-" * 96)
        for r in group.head(top).itertuples(index=False):
            # The last team on a board has no "next", so these are genuinely absent
            # rather than zero — print them as such instead of a misleading 0.0.
            rng = (
                f"{r.rank_p05}-{r.rank_p95}"
                if r.rank_p05 is not None and pd.notna(r.rank_p05) else "—"
            )
            gap = f"{r.gap_to_next:>6.1f}" if pd.notna(r.gap_to_next) else f"{'—':>6}"
            nxt = f"{r.p_beat_next:>12.1%}" if pd.notna(r.p_beat_next) else f"{'—':>12}"
            print(f"{r.rank:>3} {r.overall_rank:>4} {str(r.team)[:34]:<34} {r.record:>7} "
                  f"{r.rating:>8.1f} {gap} {nxt} {rng:>11}")
        if len(group) > top:
            print(f"    ... {len(group) - top} more")


# Columns that are text even when empty. Without this, a column that is null for a
# whole sport — `division` for the NFL and MLB — comes out of pandas as float64, lands
# in BigQuery as FLOAT, and the load fails outright because FLOAT cannot be a
# clustering field.
TEXT_COLUMNS = ("team", "division", "record")


def write_bq(table: pd.DataFrame, meta: dict) -> dict:
    """Publish to <dataset>.power_rankings, replacing this season's slice."""
    from google.cloud import bigquery

    spec = SPORTS[meta["sport"]]
    project = os.environ.get("GCP_PROJECT", "hankstank")
    dataset = os.environ.get(spec.dataset_env, spec.default_dataset)
    table_id = f"{project}.{dataset}.power_rankings"

    out = table.copy()
    for column in TEXT_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype("string")

    # No clustering or partitioning. These boards top out in the hundreds of rows, so
    # clustering buys nothing measurable, and specifying it makes the load brittle:
    # a table's clustering spec is fixed at creation, so any table created without it
    # rejects an appending load that asks for it.
    client = bigquery.Client(project=project)
    # DELETE-then-append rather than WRITE_TRUNCATE so re-running one season does not
    # destroy the others already in the table.
    client.query(
        f"DELETE FROM `{table_id}` WHERE season = {meta['season']}"
    ).result()

    client.load_table_from_dataframe(
        out, table_id,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()
    return {"table": table_id, "rows": len(out)}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", required=True, choices=sorted(SPORTS))
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, default=None,
                    help="rank as of this week (default: latest played)")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--no-prior", action="store_true")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print, write nothing")
    ap.add_argument("--write-bq", action="store_true")
    ap.add_argument("--csv", type=str, default=None, help="also save the board here")
    ap.add_argument("--no-fpi", action="store_true",
                    help="skip the ESPN strength-of-record / schedule join")
    args = ap.parse_args()

    table, meta = build_board(
        args.sport, args.season, week=args.week,
        n_boot=args.boot, use_prior=not args.no_prior, with_fpi=not args.no_fpi,
    )
    print_board(table, meta, top=args.top)

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv} ({len(table)} rows)")

    if args.write_bq and not args.dry_run:
        print("\n" + str(write_bq(table, meta)))
    else:
        print(f"\n[dry run] would write {len(table)} rows to "
              f"{SPORTS[args.sport].default_dataset}.power_rankings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
