"""HTTP fetch that works both on a TLS-inspected corporate laptop and in Cloud Functions.

Lifted from the reasoning in cfb/espn_data.py: this machine sits behind a Netskope /
Home Depot TLS-inspecting proxy whose root certificate predates the Authority Key
Identifier requirement, so OpenSSL 3 rejects it and `requests` fails — while curl
validates the same chain against the macOS trust store.

Both paths verify certificates; this is a trust-store difference, not a bypass.

Reaching for curl unconditionally would be the easy fix and the wrong one: the Cloud
Functions runtime is not guaranteed to ship a curl binary, so a subprocess call that
works locally can fail in production with a bare FileNotFoundError. So probe requests
first, fall back only when TLS interception is actually detected, and cache the result —
paying a 30s TLS timeout on every one of several hundred requests turns a two-minute
job into a 45-minute one.
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

_USE_CURL: bool | None = None
_PROBE_URL = "https://sports.core.api.espn.com/v2/sports/football"


def _probe() -> bool:
    global _USE_CURL
    if _USE_CURL is None:
        try:
            import requests

            requests.get(_PROBE_URL, timeout=15)
            _USE_CURL = False
        except Exception as exc:
            if "SSL" in type(exc).__name__ or "SSL" in str(exc):
                logger.info("TLS interception detected — using curl transport")
                _USE_CURL = True
            else:
                # Anything else (timeout, DNS) is not a trust problem; requests is
                # still the right transport and the caller will see the real error.
                _USE_CURL = False
    return _USE_CURL


def get_bytes(url: str, timeout: int = 60) -> bytes:
    """Fetch a URL, verifying TLS via whichever transport trusts this network."""
    if not _probe():
        import requests

        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content

    result = subprocess.run(
        ["curl", "-sSL", "--fail", "--max-time", str(timeout), url],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fetch failed for {url}: {result.stderr.decode()[:200]}")
    return result.stdout


def cache_dir(name: str) -> "object":
    """A writable cache directory, wherever this is running.

    The repo's data/ tree is right locally, but /workspace is read-only in Cloud
    Functions and these modules mkdir at import time — which turns a read-only
    filesystem into an import error rather than a degraded cache. Fall back to /tmp,
    which is writable there and perfectly good for a re-fetchable cache.
    """
    from pathlib import Path

    here = Path(__file__).resolve()
    root = here.parents[2] if len(here.parents) > 2 else here.parent
    for candidate in (root / "data" / name, Path("/tmp") / name):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.touch()
            probe.unlink()
            return candidate
        except OSError:
            continue
    return Path("/tmp")
