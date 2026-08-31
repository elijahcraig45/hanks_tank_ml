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

# Set once a TLS failure proves interception, so the cost is paid at most once rather
# than on every request.
_USE_CURL = False


def _is_tls_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    return "SSL" in name or "Certificate" in name or "CERTIFICATE_VERIFY_FAILED" in str(exc)


def _curl_bytes(url: str, timeout: int) -> bytes:
    result = subprocess.run(
        ["curl", "-sSL", "--fail", "--max-time", str(timeout), url],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fetch failed for {url}: {result.stderr.decode()[:200]}")
    return result.stdout


def get_bytes(url: str, timeout: int = 60) -> bytes:
    """Fetch a URL, verifying TLS via whichever transport trusts this network.

    Detection is per request rather than one upfront probe against a fixed host: the
    proxy's policy is per host, so probing sports.core.api.espn.com said "no
    interception" while site.api.espn.com failed. Try requests, and switch to curl only
    when a TLS trust error actually proves it — anything else (404, timeout, DNS) is a
    real error and propagates instead of being retried on a second transport.
    """
    global _USE_CURL

    if not _USE_CURL:
        try:
            import requests

            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            if not _is_tls_error(exc):
                raise
            logger.info("TLS interception detected — switching to curl transport")
            _USE_CURL = True

    return _curl_bytes(url, timeout)


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
