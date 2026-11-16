# Reproduce the NFL result

In 2026 the football headline on hankstank.com moved from XGBoost to a margin ridge, on the strength of a
walk-forward backtest. This demo reruns that backtest on your machine in about five seconds. It reads one committed
file and runs it through the production code in [`src/nfl`](../src/nfl). It needs no network and no cloud account.

## Run it

```bash
make demo          # the first run creates .venv-demo and installs five pinned packages
make demo-weekly   # retrain XGBoost before every week instead of every season (about 40 seconds)
```

It needs Python 3.11 or newer. On macOS, XGBoost also needs OpenMP (`brew install libomp`). Without `make`:

```bash
python3 -m pip install -r requirements-demo.txt
python3 demo/nfl_backtest.py                       # add --weekly, or --out predictions.csv for every game
```

## What it prints

```
NFL model comparison: 7,261 decided games, 1999-2025 (nflverse schedules)

Margin ridge settings, chosen on 2010-2016 (never scored): alpha=3, tau=16, sigma=12.48, the settings production ships
XGBoost: 84 features, retrained before every season
Every prediction uses only games played before it.

2017-2024 walk-forward, 2,202 games
  model                  accuracy  log loss   Brier
  Elo                       63.7%    0.6462  0.2273
  XGBoost                   63.4%    0.6456  0.2268
  Margin ridge              63.8%    0.6342  0.2217
  Vegas closing spread      66.6%    0.6086  0.2103
  Ridge minus XGBoost, log loss: -0.0113 (95% CI -0.0188 to -0.0034)

2025 holdout, 284 games
  model                  accuracy  log loss   Brier
  Elo                       65.5%    0.6473  0.2278
  XGBoost                   62.0%    0.6426  0.2269
  Margin ridge              64.4%    0.6339  0.2221
  Vegas closing spread      65.8%    0.6093  0.2116
  Ridge minus XGBoost, log loss: -0.0087 (95% CI -0.0260 to +0.0071, not significant on its own)

Lower is better for log loss and Brier. Always guessing 50% scores 0.6931 and 0.2500.
The intervals are a paired bootstrap over games (2,000 resamples). No model beats the betting market.
```

How to read it: over eight seasons the ridge beats XGBoost by 0.011 in log loss, and the interval stays below zero.
On the 2025 holdout alone it's ahead by a similar amount, but 284 games are too few to call. Both models trail the
betting market by a wide margin. The ridge earned the headline by being better and simpler, not by beating Vegas.

## What it does

1. **Loads** 7,276 NFL games from 1999 to 2025 and drops the 15 ties, as production does
   ([`data.py`](../src/nfl/data.py)).
2. **Builds production's features**, 84 per game, from scores and the schedule only: Elo, Pythagorean record,
   recent point differential, rest, head-to-head and so on. Each row is computed from games before it
   ([`features.py`](../src/nfl/features.py)).
3. **Tunes the margin ridge on 2010–2016** over a 4 × 4 grid of `alpha` and `tau`. These seasons are never scored.
   The search lands on the settings production ships (`NFL_RIDGE` in [`margin_ridge.py`](../src/nfl/margin_ridge.py)).
4. **Walks forward through 2017–2025.** The ridge refits before every week. XGBoost, the previous headline, retrains
   before every season, or every week with `--weekly`.
5. **Scores** all four on the same games, with a paired bootstrap on the ridge-minus-XGBoost difference.

The four models:

| Model | What it is |
|---|---|
| Elo | The feature builder's Elo rating (K = 20, home bonus 32 points of Elo), as a win probability |
| XGBoost | Depth-3 gradient-boosted trees on the 84 features ([`models.py`](../src/nfl/models.py)) |
| Margin ridge | One rating per team plus home field, fit by ridge regression to the point margins of the last two seasons, with older games weighted down (`tau` = 16 weeks). The predicted margin becomes a win probability through a normal curve (`sigma` = 12.48 points) |
| Vegas closing spread | nflverse's closing spread, turned into a win probability the same way (`sigma` = 13.45) |

## How it lines up with the experiment log

The 2026-09-25 margin-ridge row of [`docs/EXPERIMENT_LOG.md`](../docs/EXPERIMENT_LOG.md) has ridge 0.6342, Elo
0.6462 and Vegas 0.6086 for 2017–24, and ridge 0.6339 for 2025. The demo prints the same numbers. XGBoost lands within
0.001 of the logged values (0.6456 vs 0.6455; 2025: 0.6426 vs 0.6416). The source of that gap hasn't been traced. It
doesn't change the result. [`tests/test_nfl_demo.py`](../tests/test_nfl_demo.py) runs the demo in CI and fails if
these numbers move.

## What it leaves out

- **MLB.** Its features are built from BigQuery tables, so it can't run from one file.
- **College football.** The data comes from CollegeFootballData, which needs an API key.
- **The EPA variants.** Those need nflverse play-by-play, a much larger download.
- **The live shadow.** That's the second half of how a model earns the headline: it is scored on games nobody had
  seen when it was built. Those rows live in BigQuery.

## The data

`data/nfl_games_1999_2025.csv.gz` (130 KB) is the [nflverse](https://github.com/nflverse/nflverse-data) schedules
table, licensed CC BY 4.0. It's cut to completed 1999–2025 games and the 16 columns the features use, and kept in
nflverse's row order: XGBoost subsamples rows, so reordering games moves its scores slightly.
[`make_snapshot.py`](make_snapshot.py) rebuilds it.
