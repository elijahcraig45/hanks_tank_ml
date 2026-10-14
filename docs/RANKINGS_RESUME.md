# College football resume board (`resume_rank`)

The headline rating answers "who would win next week" (margin-based, schedule-adjusted, last season carried in).
On a season scale that can rank a team above the team that beat it. `resume_rank` answers "who has earned it",
from this season's games alone, no scores.

## Method (`src/rankings/resume.py`)
1. Opponent quality: the headline fit with 25% weight on the win/loss likelihood (rest margins) and last season carried
   in at the headline's decaying weight. A win/loss-only fit of five or six games cannot tell a weak conference from a
   strong one (first version: USC 6-1 #2, James Madison 6-0 #4 on soft schedules).
2. Strength of record: for each game, its score minus the score a typical TOP-25 team (mean rating of the division's best
   25) would expect against that opponent at that venue. The score is `(1-LAM)*win + LAM*Phi(margin/SIGMA)` with LAM=0.75,
   SIGMA=14 points: a bounded margin credit, so a blowout counts more than a narrow win but never outweighs the result.
   A win never scores below zero and a loss never above it (`FLOOR`): without that, a narrow win over a decent team could fall
   just under what a top-25 team would have done and show as a small negative. Games without scores count as wins/losses.
   Variance comes from the same expectation.
3. Head-to-head: start from strength of record and let played games reorder two teams only when those two are TIED (gap
   under Z=0.5 standard deviations of the difference). A team is never moved past one whose record is clearly different.
   Every move strictly reduces the number of contradicted games.
4. Preseason / no games: no rank. One outcome class so far: every opponent counts the same, so the order is the record.

## Evidence (2026-10-14, walk-forward, FBS vs FBS, boards as of weeks 4,5,6,8,10,12)
Tune seasons 2017-19+2021, eval seasons 2022-25 frozen. Scripts: homelab `work/resumeexp`.

Margin weight (z=0.5):

| LAM | reliability (split-half) tune / eval | next-3-weeks accuracy tune / eval | top-25 pairs contradicted tune / eval |
|---|---|---|---|
| 0 (win only) | 0.60 / 0.60 | 66.2% / 65.2% | 5.5% / 3.6% |
| **0.5 (shipped)** | **0.68 / 0.70** | **67.9% / 65.9%** | 6.8% / 4.4% |
| 0.75 | 0.71 / 0.74 | 68.6% / 67.3% | 7.5% / 5.5% |

LAM=0.5 raises accuracy +1.7 pts on tune and +0.7 on eval (at z=1: tune CI +1.0..+2.6; eval CI -0.1..+1.4). LAM=0.75 is
better on both splits and significant on eval (+1.8, CI +0.7..+2.9) but leans toward a margin ranking; 0.5 keeps the win
and the loss clearly apart. Headline for comparison: 69.4% eval, 25.5% of top-25 pairs contradicted.

Tie width: z=2 gave the fewest contradictions (top-25 1.5-2% eval) but let a team climb about 10 places above its
strength-of-record order (a 3-2 team in the top 10); z=0.5 averages about 3.5 places and contradicts more (4.4% top-25,
9.0% of all games; results order: 5.6%). Chosen: z=0.5, because moving a team past teams it is only barely level with is
what reads as wrong.

## Known limits
- After 5-6 games the top is a pack: the top 9 teams' strengths of record span about 0.5 against a per-team noise of
  about 0.8. The order inside it is a best estimate, not a separation; the site should show the number and the tie band.
- The yardstick (top-25 mean) and the opponent ratings both come from the headline family, so a mistake there moves the
  resume too.
- 502s from ESPN dropped some postseason weeks from the backtest data (identical for every board).

## Other caveats
- ESPN returned 502 for some postseason weeks during the backtest (same data for every board).
- Experiments: `mllab` scratch scripts `cfb_resume_exp.py` (results kept on the homelab under `work/resumeexp`).

## Final choice of margin weight and floor (2026-10-14)

`LAM = 0.75` and `FLOOR = True`, chosen by a rule fixed before the run (docs/RESUME_V2_REPORT.md section 7.5) and measured on
the production code against the previously live board (0.5, no floor), paired block bootstrap, eval = 2022-25:

| | next-3-weeks accuracy | top-25 contradictions | reliability | persistence skill |
|---|---|---|---|---|
| tune | +0.35 pts [-0.15,+0.86] | +1.63 pts [-0.31,+3.62] | -0.00 | -0.008 |
| eval | **+1.03 pts [+0.50,+1.53]** | +1.11 pts [-0.50,+2.75] | -0.02 | -0.001 |

All four tolerances held. The floor alone (at 0.5) costs 0.02-0.03 of reliability and nothing else; at 0.75 the margin weight
recovers it. The unfloored 0.75 is a little more reliable (+0.04) but can score a win negative. More margin weight means
margin counts for more relative to the result; 0.5 remains available as a one-line change if that is not wanted.
