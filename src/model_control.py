"""Read-only client for the model control plane (docs/MODEL_CONTROL.md).

The lab is the only writer. The pipeline reads `<CONTROL_DATASET or 'control'>.model_control_current`
ONCE per call to `get_state` and asks it three questions per model key:

    is_paused(key) -> bool
    pin(key)       -> (gs:// uri, sha256) | None
    tiers(key)     -> (high, medium) | None

FAIL OPEN, always: a missing dataset, an empty view, a slow or failing query, a bad value, or
MODEL_CONTROL_DISABLED=1 all yield an empty state whose answers are the defaults (not paused, no
pin, no tier override), i.e. exactly how the pipeline behaved before this module existed.
Exactly one WARNING is logged per failed read. This module never writes anything.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT = "hankstank"
QUERY_TIMEOUT_S = 5.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GCS_RE = re.compile(r"^gs://[^/\s]+/\S+$")


def _dataset() -> str:
    return os.environ.get("CONTROL_DATASET") or "control"


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _row_dict(row) -> dict:
    try:
        return dict(row.items())
    except Exception:  # noqa: BLE001 - plain mappings / odd row types
        return dict(row)


def _frac(value) -> Optional[float]:
    try:
        x = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


class ControlState:
    """Validated snapshot of the control view for one sport. Immutable by convention."""

    def __init__(self, rows: Optional[list] = None, available: bool = False):
        self.available = bool(available)
        self._targets: dict[str, dict] = {}
        for raw in rows or []:
            try:
                row = _row_dict(raw)
            except Exception:  # noqa: BLE001 - one bad row must not poison the rest
                continue
            target = _clean(row.get("target"))
            if target is None or target == "*":
                continue  # '*' carries sport-wide banner fields only; not used here
            self._targets[target] = row

    def _row(self, key: str) -> dict:
        return self._targets.get(key) or {}

    def is_paused(self, key: str) -> bool:
        """Only an explicit run_state of 'paused' pauses; anything else is active."""
        value = _clean(self._row(key).get("run_state"))
        return value is not None and value.lower() == "paused"

    def pin(self, key: str) -> Optional[tuple[str, str]]:
        """(uri, sha256) when the key is `live` with BOTH a valid gs:// uri and a 64-hex sha256.

        role=live is required, as the contract says: an artifact left over from an earlier pin
        must not silently steer production once the operator has moved the role on."""
        row = self._row(key)
        role = _clean(row.get("role"))
        uri = _clean(row.get("artifact_uri"))
        sha = _clean(row.get("artifact_sha256"))
        if role is None or role.lower() != "live" or uri is None or sha is None:
            return None
        sha = sha.lower()
        if not _GCS_RE.match(uri) or not _SHA256_RE.match(sha):
            logger.warning("model control: ignoring invalid pin for %s (uri/sha256 malformed)", key)
            return None
        return uri, sha

    def tiers(self, key: str) -> Optional[tuple[float, float]]:
        """(high, medium) when both parse and 0.5 < medium < high < 1, else None."""
        row = self._row(key)
        high, medium = _frac(row.get("tier_high")), _frac(row.get("tier_medium"))
        if high is None or medium is None:
            if row.get("tier_high") is not None or row.get("tier_medium") is not None:
                logger.warning("model control: ignoring incomplete/invalid tiers for %s", key)
            return None
        if not (0.5 < medium < high < 1.0):
            logger.warning("model control: ignoring out-of-range tiers for %s (%s, %s)",
                           key, high, medium)
            return None
        return high, medium


def empty_state() -> ControlState:
    return ControlState([], available=False)


def get_state(sport: str, client=None) -> ControlState:
    """Read the control view once. Never raises; returns an empty state on any problem.

    `client` is an injectable BigQuery client (tests). Without one, a client is built lazily
    unless MODEL_CONTROL_DISABLED=1 (a kill switch that skips the read entirely)."""
    try:
        if client is None:
            if os.environ.get("MODEL_CONTROL_DISABLED") == "1":
                return empty_state()
            from google.cloud import bigquery

            client = bigquery.Client(project=PROJECT)
        from google.cloud import bigquery

        table = f"{PROJECT}.{_dataset()}.model_control_current"
        job = client.query(
            f"SELECT * FROM `{table}` WHERE sport = @sport OR sport = '*'",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("sport", "STRING", sport)]
            ),
        )
        rows = [r for r in job.result(timeout=QUERY_TIMEOUT_S)]
        # A row for another sport can never steer this one.
        rows = [r for r in rows if _clean(_row_dict(r).get("sport")) in (sport, "*")]
        return ControlState(rows, available=True)
    except Exception as exc:  # noqa: BLE001 - fail open by contract
        logger.warning("model control unavailable for %s, using defaults: %s", sport,
                       str(exc)[:200])
        return empty_state()


def tier_label(p: float, high: float, medium: float) -> str:
    """'high' / 'medium' / 'low' for the winning-side probability of `p` (home win prob)."""
    top = max(p, 1.0 - p)
    if top >= high:
        return "high"
    if top >= medium:
        return "medium"
    return "low"
