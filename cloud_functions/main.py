"""
DEPRECATED — this module is no longer the active Cloud Function entry point.

The active entry point is src/cloud_function_main.py (function: daily_pipeline).
This file is kept for audit purposes. See cloud_functions/DEPRECATED.md for details.

Original purpose: re-export top-level functions from daily_updates.py for
Cloud Functions framework discovery. Replaced by mode-based routing in
cloud_function_main.py as of 2026-04-08.
"""
