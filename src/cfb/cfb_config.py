"""Configuration for the college football pipeline.

CFB is not NFL with different logos. Three settings differ for real reasons:

  * MOV-damped Elo. Blowouts are routine in college (60-point wins happen); without
    damping, ratings run away from a handful of lopsided results.
  * Heavier season regression (0.50 vs NFL's 0.33). Rosters turn over far more, so
    last year's rating carries less information.
  * Larger home-field advantage. College HFA runs ~3.5 points against the NFL's ~1.9.

FBS and FCS share this config and the same BigQuery tables, separated by a `division`
column. Models are trained and evaluated per division and never pooled — but Elo runs
as a single pool across both, because cross-division games are the edges that put the
two populations on a common rating scale.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Locally this file is src/<sport>/config.py, so the repo root is two levels up. In the
# deployed Cloud Function the tree is flattened to /workspace, which has only one
# parent, and parents[2] raised IndexError — the reason every scheduled run failed on
# import. Fall back to the module's own directory when the tree is shallower.
_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent


def _writable_base(preferred: Path) -> Path:
    """A directory we can actually write to.

    /workspace is read-only in Cloud Functions, so a cache rooted there fails at import
    time. /tmp is the writable scratch space there and a fine cache anywhere.
    """
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.touch()
        probe.unlink()
        return preferred
    except OSError:
        return Path("/tmp")


@dataclass(frozen=True)
class CfbContext:
    project: str
    season_dataset: str
    hist_dataset: str
    season: int

    @classmethod
    def from_env(cls) -> "CfbContext":
        return cls(
            project=os.environ.get("GCP_PROJECT", "hankstank"),
            season_dataset=os.environ.get("CFB_DATASET", "cfb_season"),
            hist_dataset=os.environ.get("CFB_HIST_DATASET", "cfb_historical"),
            season=int(os.environ.get("CFB_SEASON", "2026")),
        )


CTX = CfbContext.from_env()

DATA_DIR = _writable_base(REPO_ROOT / "data") / "cfb"
RAW_CACHE = DATA_DIR / "raw"

# ESPN's public scoreboard API — no key required, unlike CollegeFootballData.
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
DIVISION_GROUPS = {"fbs": 80, "fcs": 81}

FIRST_SEASON = int(os.environ.get("CFB_FIRST_SEASON", "2014"))
HOLDOUT_SEASON = 2025

# Elo — see the module docstring for why these differ from NFL's.
ELO_START = 1500.0
ELO_K = 24.0                   # higher than NFL: ~12 games/season to learn from
ELO_HOME_BONUS = 65.0          # ~3.5 points
ELO_SEASON_REGRESSION = 0.50   # heavy roster turnover
ELO_MOV_DAMPING = True

PYTHAG_EXPONENT = 2.37
