"""Configuration for the NFL pipeline.

Deliberately env-driven rather than module-level literals — the MLB modules hardcode
PROJECT/DATASET at import time, which is the single biggest barrier to a second sport.
Nothing here imports from the MLB modules, and nothing in the MLB modules imports this.

When CFB arrives, add SPORT and derive the dataset names from it.
"""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NflContext:
    project: str
    season_dataset: str
    hist_dataset: str
    bucket: str
    season: int

    @classmethod
    def from_env(cls) -> "NflContext":
        return cls(
            project=os.environ.get("GCP_PROJECT", "hankstank"),
            season_dataset=os.environ.get("NFL_DATASET", "nfl_season"),
            hist_dataset=os.environ.get("NFL_HIST_DATASET", "nfl_historical"),
            bucket=os.environ.get("GCS_BUCKET", "hanks_tank_data"),
            season=int(os.environ.get("NFL_SEASON", "2026")),
        )


CTX = NflContext.from_env()

# nflverse history starts in 1999. Betting lines are populated from 1999 too
# (verified in the M0 spike: spread_line is 100% non-null across all seasons).
FIRST_SEASON = 1999

# 2025 is held out as a true out-of-sample season. Touch it exactly once, at the end.
HOLDOUT_SEASON = 2025

DATA_DIR = REPO_ROOT / "data" / "nfl"
MODEL_DIR = REPO_ROOT / "models"
RAW_CACHE = DATA_DIR / "raw"

# Corporate TLS interception breaks Python's certifi bundle on this machine while curl
# (system keychain) works. If a CA bundle is configured, honor it. Harmless in GCP.
CA_BUNDLE = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")

# --- Elo -------------------------------------------------------------------
# NOT the MLB constants. Measured in the M0 spike: NFL home-field advantage has
# decayed materially, from ~53 Elo (1999-2007) to ~32 Elo (2021-2025). A static 48
# (the 2008-2015 value) would systematically over-favor home teams in modern games.
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_BONUS = 32.0          # current-era value; sweep {25, 32, 40, 48}
ELO_SEASON_REGRESSION = 0.33   # sweep {0.25, 0.33, 0.40}

# Football Pythagorean exponent (Football Outsiders); MLB uses 1.83.
PYTHAG_EXPONENT = 2.37
