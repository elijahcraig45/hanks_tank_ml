"""
Cloud Functions entry point - contains all functions for daily data pipeline
"""

# Re-export all functions from daily_updates
from daily_updates import (
    update_statcast_2026,
    update_pitcher_stats_2026,
    rebuild_v7_features,
    predict_today_games
)

# Make functions available to functions-framework
__all__ = [
    'update_statcast_2026',
    'update_pitcher_stats_2026',
    'rebuild_v7_features',
    'predict_today_games'
]
