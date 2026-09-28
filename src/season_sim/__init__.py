"""Rest-of-season Monte Carlo for football (NFL and college). EXPERIMENT, shadow only.

    engine.py   rating state from the margin ridge, posterior draws, in-season refits,
                and the vectorised game simulator shared by both sports
    nfl.py      NFL divisions, standings with the official tiebreakers, 14-team playoff
    cfb.py      FBS conference standings, title games, committee proxy, CFP brackets
    output.py   per-team and per-bracket-slot tables, the most-likely bracket, writers
    run.py      the `season_sim` mode behind both Cloud Functions

Evidence and the backtest live in scripts/football/backtest_season_sim.py and
docs/EXPERIMENT_LOG.md.
"""
