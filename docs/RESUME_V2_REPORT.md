# A defensible season-scale ranking for college football: design, experiments and results

Status: **protocol locked 2026-10-14, before any v2 result was computed.** Sections 1-5 were written first and are not
edited after the results arrive except to correct errors (listed in the changelog at the end). Sections 6 onward report
what happened, including what did not work.

## 1. The problem

hankstank.com ranks college football teams. Two different questions get asked of one number:

* **Forecast** ("who would win next week?"). Answered by the headline rating: a margin-based, schedule-adjusted ridge
  regression with last season carried in. It is the right tool for matchups and simulations, and it is measured
  against the only fair yardstick for a forecast: how well it predicts games not yet played.
* **Resume** ("who has earned it?"). A retrospective measure of what each team has actually done this season, against
  the schedule it actually played. A resume has no future in it. It is not validated by prediction (although prediction
  is a useful sanity check); it is validated by whether it behaves the way an accountable ranking must.

Mixing them produces visible failures. In week 7 of 2026 the forecast ranked Alabama (5-1) first, a team that had
just lost at home to #4 Georgia, because its margins were large against a strong schedule. That is correct for a
forecast and indefensible as a statement about who has earned what.

An earlier resume board (shipped 2026-10-14, "v1") fixed the worst cases but was built by patching one defect at a
time. This report starts from the requirements instead.

### What went wrong with v1 (all found by use, not by theory)

| symptom | cause |
|---|---|
| USC (6-1) ranked #2, James Madison (6-0) #4 | opponents rated from this season's win/loss only; five games cannot tell a weak conference from a strong one |
| a 3-2 team in the top 10, a 3-3 team at #17 | "tie groups" built from a leader's gap were wider than the whole top 25's spread |
| Georgia Tech's 28-25 win over Duke scored -0.01 | additive margin credit is measured against what a top-25 team would do, so a narrow win over a decent team can fall below the yardstick |
| "tied with #1-#15" | the tie rule compared noise from results (sum of p(1-p)) with a spread it could not resolve, and it ignored uncertainty in the yardstick itself |
| unbeaten teams indistinguishable after 5-6 games | binary results carry little information; margin carries more, but v1 mixed margin in additively |

## 2. Desiderata (what "defensible" means here)

A ranking is defensible if a skeptical reader can check each of the following without taking our word for it.

* **D1 Schedule-aware.** A win over a strong opponent counts more than a win over a weak one; a loss to a weak team
  costs more than a loss to a strong one. (Yardstick: a typical top-25 team.)
* **D2 Result-monotone.** Against the same opponent at the same venue, a win never scores below a loss, and a win never
  scores below zero while a loss never scores above zero. A reader who sees a negative number next to a W has found a bug.
* **D3 Margin-informed but bounded.** Margin is evidence, and the only extra evidence a few games provide. It may
  refine a result; it may never outweigh it, and a blowout of a hopeless team may not outscore a narrow win over a good one.
* **D4 Honest about uncertainty, consistently.** The ranges shown must come from the same machinery as the order shown
  (a team inside its own range, ranges that widen when evidence is thin, no ad hoc thresholds), and must be testable:
  a rank described as firm should persist.
* **D5 Game integrity.** No ranking should contradict a head-to-head result unless the evidence for the opposite order
  is strong. "Strong" must be a stated, tunable number, not a side effect of an algorithm.
* **D6 Order-independent and reproducible.** Shuffling the schedule changes nothing; the same data gives the same board
  (seeded Monte Carlo is allowed if the seed is fixed).
* **D7 Explainable.** A team's number is a sum of per-game contributions a reader can inspect.
* **D8 Fair to unequal schedules.** A bye week or a cancelled game should not decide the order by itself.
* **D9 Few knobs, each chosen on seasons it is not scored on.**
* **D10 Not worse as a predictor than the alternatives**, and more reliable. Prediction is not the objective but it
  catches a ranking that has drifted into nonsense.

## 3. Hypotheses

Informed by earlier experiments where stated; everything marked NEW had not been tested when this protocol was locked.

* **H1 (NEW)** A multiplicative game value, `surprise x decisiveness`, satisfies D2 and D3 by construction and costs no
  more than the pre-set non-inferiority margins against v1's additive credit.
* **H2 (NEW)** Replacing "sum of credits" with a shrunk mean `sum / (n + k)` removes the games-played bias (D8) without
  hurting the other metrics.
* **H3 (partly known)** Margin information raises reliability and forward accuracy (known: weight 0.5 gave +0.08 and
  +0.7..1.7 pts). The bounded multiplicative form keeps most of that.
* **H4 (NEW)** Propagating uncertainty in the yardstick (posterior draws of every team's strength, hence of the
  benchmark and of every opponent) gives ranges whose firmness predicts order persistence better than v1's
  p(1-p) tie rule.
* **H5 (NEW)** A consensus ranking, `argmin  sum over pairs [P(j above i) + eta * 1(j beat i)]`, is a principled
  replacement for "sort, then let results reorder tied pairs", and `eta` has a direct meaning: reverse a played result only if
  the other order has probability above `(1 + eta) / 2`.
* **H6 (NEW)** If H1-H5 hold, v2 is non-inferior to v1 on forward accuracy and better on reliability, coherence with
  played games and persistence calibration.

## 4. Experimental protocol

**Data.** ESPN game results 2016-2025 (FBS and FCS, regular season; some postseason weeks returned 502 and are absent
in every arm identically). Seasons are split once: **tune** 2017, 2018, 2019, 2021 and **eval** 2022-2025. 2020 is
dropped (COVID season) but serves as 2021's prior. Every knob is chosen on tune; eval is scored once per decision.

**Boards.** For every season and every week w in 4..12, a board is built from games strictly before week w (plus last
season at its decayed weight, as the headline does). FBS teams that have played are ranked.

**Metrics** (all on FBS-vs-FBS games):

| id | metric | direction | what it answers |
|---|---|---|---|
| M1 | next-3-weeks accuracy: the higher-ranked team wins | higher | sanity check on prediction (D10) |
| M2 | split-half reliability: Spearman-Brown correlation of team scores from alternate games | higher | how much of the order is signal (D3, D8) |
| M3 | contradicted played games, all and among the board's top 25 | lower | game integrity (D5) |
| M4 | order persistence calibration: Brier score of `P(A above B)` for pairs in the top 40, against whether A is still above B three weeks later | lower | are the ranges honest? (D4) |
| M5 | games-played bias: mean score difference between teams with n and n-1 games, minus what their records imply | near 0 | fairness to unequal schedules (D8) |
| M6 | axiom checks (property tests) | all pass | D2, D3, D6, D7 |

**Uncertainty.** Block bootstrap over (season, week) boards, 2000 resamples, 95% intervals.

**Decision rules (fixed now).**

1. *Components* are kept only if they pass their own hypothesis on **tune**: H1 and H2 by non-inferiority (point
   estimate no worse than -0.5 pts of M1 and -0.05 of M2, with the CI lower bound above -1.5 pts / -0.10), H4 and
   H5 by strict improvement in M4 / M3 with a non-negative point estimate on M1.
2. *Parameters* (`lambda` margin weight, `k` shrinkage, `eta`, interval width) are the values on a pre-listed grid
   that maximize M2 subject to M3 (top-25) not exceeding 3% on tune. Ties go to the simpler value.
3. *v2 vs v1* is judged once on eval with the same non-inferiority margins on M1 and strict improvement required
   on at least two of M2, M3, M4 with no regression on the others.
4. If v2 does not pass, v1 plus the "win never negative" floor remains the recommendation and this report says so.

## 5. Things this protocol deliberately does not do

* It does not tune against AP/Coaches polls or any outside ranking. They are human opinion, not truth.
* It does not use the 2025 touch-once holdout reserved for the football forecast models (this is a ranking, but the
  eval seasons 2022-2025 are already used by the headline tuning, so any improvement claimed here is *relative to v1
  under identical data*, not an out-of-sample claim about the world).
* It does not try to rank FCS teams differently from FBS; both boards use the same method.


## 6. Method: what was built and why each piece exists

All code is in `src/rankings/resume_v2.py` (tested by `tests/test_rankings_resume_v2.py`, 26 property tests) and the
experiment scripts are in `research/resume_v2/`.

**The yardstick (D1, D4).** A resume needs a yardstick: what would a typical top-25 team have done against this
schedule? The yardstick is a ridge regression on scoring margin (the headline's model), which is exactly a Gaussian
prior on every team's strength, so the posterior is closed-form: `beta ~ N(beta_hat, sigma^2 (X'WX + alpha I)^-1)`.
Sampling it gives alternative worlds in which every opponent, and the benchmark itself, is a little better or worse.
The posterior mean was verified equal to the headline engine's fit (a test), and the estimated game-margin sd recovers a
known 17 points on synthetic data (a test).

**Game credits (D1-D3).** Four credit functions, each a one-line change so they can be compared on identical data:

| kind | definition | satisfies D2 (win never negative)? |
|---|---|---|
| `win` | W - p, with p the benchmark's chance to win | yes, but ignores margin |
| `add` (v1) | (1-L)(W-p) + L(Phi(M/14) - E[Phi]) | **no**: a narrow win over a good team scores below zero |
| `addfloor` | `add`, wins floored at 0 and losses capped at 0 | yes |
| `mult` (new) | win: (1-p)f, loss: -p f, f = (1-L) + L(2 Phi(abs(M)/14) - 1) | yes, by construction, and bounded by the surprise of the result |

`mult` was designed as the principled answer to D2 and D3: the *size* of a result is how surprising it was for a
top-25 team, and margin only scales it between (1-L) and 1. Properties verified by test, over random opponents and
margins: a win is never negative, a loss never positive, every win outscores every loss against the same opponent, a
bigger win is worth more, a blowout cannot exceed the benchmark's surprise.

**Aggregation (D8).** Sum, or shrunk mean `sum / (n + k)`.

**Order and ranges (D4, D5).** Two designs. (a) v1's rule: sort by score, then let played games reorder two teams only if
they are statistically indistinguishable (gap under z standard deviations of the difference), each move strictly reducing
contradicted games. (b) New: a weighted *consensus* ranking, `argmin sum over pairs [P(j above i) + eta * 1(j beat i)]`,
solved by insertion moves, with `P` read off either yardstick draws, a game bootstrap, or the normal approximation
`Phi((s_i - s_j) / sqrt(v_i + v_j))`. Ranges are the places a team could be passed (`P >= 0.2`).

**Standardised score.** `z = sum(credit) / sqrt(sum(variance))`, how many standard deviations better than a top-25 team's
season, with per-game variance by exact split-point quadrature (verified against `p(1-p)` for the win credit; a first
Gauss-Hermite version overstated it several-fold at the win/loss step and was replaced).

## 7. Results

### 7.1 Round 1 (pre-registered): the new machinery

72 boards (8 seasons x weeks 4-12), 29 arms, 100 yardstick draws. Tune 2017-19+2021 / eval 2022-25.

| arm | acc tune / eval | top-25 contradicted | reliability | persistence skill (Brier skill score) |
|---|---|---|---|---|
| **v1 shipped** (exact production code) | 68.2 / 66.4 | **7.1 / 4.5** | 0.680 / 0.707 | **+0.095 / +0.090** |
| add 0.5, sorted | 68.5 / 66.7 | 25.2 / 17.6 | 0.680 / 0.707 | +0.056 / +0.033 |
| addfloor 0.5, sorted | 68.2 / 66.4 | 23.0 / 17.0 | 0.653 / 0.665 | +0.031 / -0.006 |
| mult 0.5, sorted | 67.9 / 66.6 | 20.1 / 17.6 | 0.652 / 0.652 | -0.016 / -0.045 |
| mult 0.5, consensus eta=0.3 (yardstick draws) | 67.9 / 66.7 | 16.6 / 14.1 | 0.652 / 0.652 | -0.016 / -0.049 |
| mult 0.5, consensus eta=0.8 | 67.9 / 66.8 | 14.3 / 11.8 | 0.652 / 0.652 | -0.017 / -0.050 |
| mult 0.5, k=3 sorted, +game bootstrap | 68.0 / 66.4 | 21.1 / 18.9 | 0.652 / 0.652 | +0.091 / +0.071 |

(Top-25 contradictions are percentages of played top-25 pairs. Sorted arms have no head-to-head step.)

**H1 (multiplicative credit satisfies D2/D3 within the non-inferiority margins): mixed.** It satisfies D2/D3 by
construction. Against v1 on eval its accuracy is level (+0.02 pts, CI -0.55..+0.60) but its reliability is 0.06 lower
(CI -0.06..-0.05), slightly past the -0.05 margin. `addfloor` is inside the margin (-0.04). The difference between the two
is small: mult minus addfloor is -0.014 reliability on eval (CI -0.022..-0.006), no difference on accuracy or coherence.

**H2 (shrunk mean fixes the games-played bias): not testable with the data we have.** Shrinking changes accuracy by at
most 0.1 pt and nothing else measurably. The bias metric M5 was mechanical (see 9, deviations), so the choice between sum and
shrunk mean rests on D8 alone.

**H3 (margin raises reliability and accuracy): confirmed.** Win-only credit has reliability 0.600/0.607; additive 0.25,
0.5, 0.75 give 0.644/0.661, 0.680/0.707, 0.719/0.749; forward accuracy rises 2.0 points (tune) and 2.9 points (eval) from win-only to 0.75.

**H4 (propagating yardstick uncertainty gives honest ranges): rejected, and that is the most informative result of the
round.** Pair probabilities from yardstick draws have *negative* skill at predicting whether an order persists three weeks
later (-0.016 / -0.045), worse than a constant. The same machinery with the games themselves resampled has the skill of
v1's simple `p(1-p)` rule (+0.091 / +0.071 vs +0.095 / +0.090). So the instability of a resume comes from **sampling noise
in a handful of results, not from not knowing how good the opponents are.** v1's variance rule was already calibrated.

**H5 (consensus ordering is a principled replacement for the tie rule): rejected as stated.** With yardstick-draw
probabilities, no eta on the pre-registered grid comes near v1's coherence (best: 11.8% of top-25 pairs at eta=0.8 vs 4.5%
for v1), because the consensus pays for every team it passes while v1's search only requires the two teams to be tied.

### 7.2 Round 2 (EXPLORATORY: designed after round 1 failed)

Round 2 was designed after seeing round-1 results, so it is exploratory: parameters were read from tune, and eval was
looked at once. Calibrated probabilities `Phi((s_i - s_j)/sqrt(v_i + v_j))` replace yardstick draws, and v1's own ordering
is applied to every credit so credit and ordering effects separate.

Paired block bootstrap against v1 shipped (percentage points for accuracy and contradictions, raw for reliability):

| arm | acc tune | acc eval | top-25 tune | top-25 eval | reliability tune / eval |
|---|---|---|---|---|---|
| add 0.5, v1 order (plumbing check) | +0.02 [-0.39,+0.45] | +0.42 [-0.02,+0.86] | +2.56 [+1.20,+3.91] | +1.29 [+0.16,+2.50] | 0.00 / 0.00 |
| **addfloor 0.5**, v1 order | -0.25 [-0.74,+0.23] | -0.39 [-1.00,+0.21] | +0.23 [-1.77,+2.19] | +1.15 [+0.28,+2.00] | -0.03 / -0.04 |
| **mult 0.5**, v1 order | -0.27 [-0.89,+0.37] | +0.02 [-0.55,+0.60] | -0.55 [-2.72,+1.76] | +0.56 [-0.41,+1.46] | -0.03 / -0.06 |
| mult 0.75, v1 order | -0.17 [-0.82,+0.53] | **+1.26 [+0.59,+1.92]** | +1.34 [+0.05,+2.78] | +0.91 [-0.31,+2.11] | -0.01 / -0.04 |
| mult 0.5, standardised score z | -0.15 [-0.85,+0.56] | -0.40 [-1.14,+0.25] | +0.12 [-1.79,+2.17] | +0.60 [-0.28,+1.44] | -0.03 / -0.06 |
| mult 0.5, consensus eta=2 | -0.34 [-0.93,+0.29] | +0.07 [-0.47,+0.68] | -0.96 [-3.19,+1.41] | -0.35 [-1.67,+0.86] | -0.03 / -0.06 |
| mult 0.5, consensus eta=1 | -0.27 [-0.83,+0.33] | +0.13 [-0.38,+0.72] | +2.01 [+0.13,+4.00] | +1.04 [+0.10,+1.89] | -0.03 / -0.06 |

* **The plumbing is sound.** Re-implementing v1 on the new code (`add 0.5`) gives accuracy within 0.4 points of the shipped
  board and identical reliability.
* **Neither the standardised score nor the consensus beats v1's ordering.** z is slightly worse on eval (accuracy -0.40,
  persistence skill 0.084 vs 0.097); consensus matches v1's coherence only at eta=2, where played results are nearly hard
  constraints, and then is statistically indistinguishable from it on every metric. A rule that needs a knob set to twice
  its maximum meaningful value to match a simpler rule is not an improvement.
* **The multiplicative credit and the floored credit are statistically equivalent** on accuracy and coherence; floored is
  0.014 more reliable on eval.
* **Margin weight is a real dial, not noise.** At 0.75 the credit is non-inferior on everything and 1.26 points more
  accurate on eval, at the price of margin counting for more relative to the result and slightly more contradictions (+1.3
  pts, tune).

### 7.3 Sensitivity

| dial | setting | tune acc / top-25 / reliability | eval acc / top-25 / reliability |
|---|---|---|---|
| benchmark size (mult .5) | top 10 | 67.5 / 6.7 / 0.623 | 66.3 / 4.8 / 0.613 |
| | **top 25** | 67.9 / 6.6 / 0.652 | 66.4 / 5.1 / 0.652 |
| | top 40 | 67.9 / 8.0 / 0.669 | 66.1 / 5.1 / 0.677 |
| | top 70 | 67.8 / 8.1 / 0.693 | 66.2 / 5.6 / 0.709 |
| tie width z (mult .5, standardised) | 0.35 / 0.5 / 0.7 / 1.0 | top-25 9.8 / 7.3 / 4.9 / 4.0 | top-25 6.4 / 5.1 / 3.2 / 1.9 |

Accuracy is flat in both dials. A broader yardstick is more reliable (the top-70 benchmark reaches 0.71 on eval) because
it grades more games against a bar they can actually discriminate on; "typical top-25 team" is kept because it has a plain
meaning, but "typical bowl team" would be defensible too. Tie width is a straight trade of coherence against responsiveness.

By stage of the season (all seasons): next-3-weeks accuracy climbs from about 64% (weeks 4-6) to 67% (7-9) to 71% (10-12)
for every arm; the arms are indistinguishable at each stage. Contradicted top-25 pairs are essentially zero before week 7
(few top teams have met) and rise to 7-10% by week 10.

### 7.4 What the live board looks like (2026, through week 7)

| # | team | rec | v1 credit | addfloor .5 | mult .5 | mult .75 |
|---|---|---|---|---|---|---|
| 1 | Georgia | 6-0 | 1 | 1 | 1 | 2 |
| 2 | Alabama | 5-1 | 2 | 2 | 2 | 3 |
| 3 | Notre Dame | 6-0 | 3 | 3 | 3 | 1 |
| 4 | Indiana | 6-0 | 4 | 4 | 4 | 6 |
| 5 | Texas Tech | 5-0 | 7 | 7 | 5 | 5 |
| 6 | Miami | 5-0 | 5 | 5 | 6 | 4 |
| 7 | UCLA | 5-0 | 6 | 6 | 7 | 7 |
| 8 | BYU | 5-0 | 10 | 10 | 8 | 9 |
| ... | USC (6-1) | | 31 | 35 | 33 | 31 |
| ... | James Madison (6-0) | | 22 | 19 | 16 | 16 |
| ... | Georgia Tech (2-3) | | 97 | 104 | 94 | 93 |

Georgia Tech's 28-25 win over Duke scores -0.06 under the v1 credit (the defect), 0.00 under `addfloor`, +0.06 under `mult`.
The defect is a bounded, local one: the top seven teams are the same set under every credit and no team there moves more than two places; places 8-10 reshuffle among BYU, Kentucky and Houston. (This uses the v2 yardstick. The published board, whose opponents are rated with the 25% win/loss blend, shows the Duke win at -0.01.)

### 7.5 Round 3: confirming the exact thing that would ship (protocol fixed before the run)

Rounds 1 and 2 tested the floor at lambda 0.5 and the multiplicative credit at 0.75, on the research code (margin-only
opponents). The candidate the owner approved, **the floor plus margin weight 0.75 on the production code** (opponents rated
with the 25% win/loss blend, logistic variance), had not been run. Round 3 runs it, and only it, against what is live.

Arms (production module `rankings.resume`, same boards, same metrics): `live` (lambda 0.5, no floor), `F.5` (lambda 0.5,
floor), `F.75` (lambda 0.75, floor), `noF.75` (lambda 0.75, no floor: a reference that violates D2).

Decision rule, fixed before running. Candidate X is *adoptable* if, against `live`, paired block bootstrap:

1. eval accuracy difference has a CI lower bound of at least -0.5 points, and the tune point estimate is at least -0.5;
2. top-25 contradictions are not worse by more than 2.0 points (point estimate, tune and eval);
3. reliability is not lower by more than 0.05 (point estimate, tune and eval);
4. persistence skill (M4) is not lower by more than 0.03 (point estimate, tune and eval).

Ship `F.75` if adoptable; otherwise `F.5` if adoptable; otherwise ship nothing and keep the live credit. If `F.75` is
adoptable it is shipped even though it was not the more conservative option, because the owner asked for it; the report
records its measured costs next to its measured gains.

**Results (production module, 72 boards; paired block bootstrap against `live`):**

| arm | acc tune | acc eval | top-25 tune | top-25 eval | reliability tune / eval | persistence skill diff tune / eval |
|---|---|---|---|---|---|---|
| `F.5` (floor, 0.5) | -0.20 [-0.61,+0.20] | -0.12 [-0.56,+0.32] | -0.55 [-1.41,+0.24] | +0.19 [-0.28,+0.74] | -0.02 / -0.03 | -0.003 / -0.004 |
| **`F.75` (floor, 0.75)** | +0.35 [-0.15,+0.86] | **+1.03 [+0.50,+1.53]** | +1.63 [-0.31,+3.62] | +1.11 [-0.50,+2.75] | -0.00 / -0.02 | -0.008 / -0.001 |
| `noF.75` (0.75, no floor; violates D2) | +0.59 [+0.17,+1.05] | +1.48 [+0.98,+1.97] | +1.46 [-0.61,+3.26] | +0.96 [-0.68,+2.65] | +0.04 / +0.04 | +0.018 / -0.001 |

Level of each arm (accuracy tune / eval, top-25 contradicted %, reliability): live 68.2 / 66.4, 7.1 / 4.5, 0.684 / 0.712;
F.5 68.0 / 66.3, 6.6 / 4.7, 0.663 / 0.681; **F.75 68.5 / 67.4, 8.8 / 5.6, 0.681 / 0.696**; noF.75 68.8 / 67.9, 8.6 / 5.5, 0.721 / 0.751.

**Rule applied.** `F.75` passes all four tolerances (accuracy lower bound +0.50 against the -0.5 floor; top-25 +1.63 and +1.11
against +2.0; reliability -0.00 and -0.02 against -0.05; persistence skill -0.008 and -0.001 against -0.03). It is shipped.

**Reading it.** Compared with the previous board, the shipped one is about one point more accurate on the frozen seasons, as
reliable (-0.00 / -0.02), about one point more contradicted in the top 25 (not significant), and it no longer scores a win
below zero. The floor itself costs about 0.04 of reliability at either weight; the extra margin weight buys it back and then
some in accuracy. This is a measured improvement in accuracy and in the one thing the owner reported as wrong (negative wins),
at a measured cost in coherence that the confidence intervals cannot distinguish from zero.

## 8. Decisions, and the evidence behind each

| decision | alternatives tried | choice | why |
|---|---|---|---|
| Yardstick | win/loss-only opponents; blend; margin ridge + Gaussian posterior | margin ridge with last season at a decayed weight | W/L-only could not rate a conference after five games (USC #2, JMU #4); margin-only matches the blend on every metric (R0 vs v1) and has a closed-form posterior |
| Benchmark | top 10 / 25 / 40 / 70 | mean of the top 25 | conventional ("a typical top-25 team"), plain to explain; reliability rises with N but accuracy does not, so this is a values choice (7.3) |
| Credit | win, add, addfloor, mult | **addfloor** (shipped) | D2 requires a sign-consistent credit; addfloor is the smallest change that gets it, and costs 0.03-0.04 reliability and nothing measurable elsewhere. mult is more elegant (continuous, bounded) but 0.014 less reliable; both are defensible, the evidence leans to addfloor |
| Margin weight L | 0 .. 0.9 | **0.75** (shipped; round 3 rule) | L=0 is least informative (rel 0.60). Round 3 measured 0.75 with the floor against the live 0.5: +1.0 point of accuracy on eval (CI +0.5..+1.5) with reliability and coherence inside the pre-set tolerances. The cost is that margin counts for more relative to the result (the narrowest win is worth a quarter of a blowout win, not half); 0.5 is a one-line change if that is not wanted |
| Aggregation | sum, shrunk mean k=0..6 | sum (shipped) | no measurable difference; shrunk mean is the fairer *in principle* to byes and could be adopted on D8 alone |
| Order | sort; v1 tie rule; consensus (draws, bootstrap, calibrated) | v1's tie rule (z=0.5) | matches or beats every alternative on coherence at equal accuracy, with no extra knob; consensus needs eta=2 to match it |
| Uncertainty | yardstick draws; game bootstrap; p(1-p) | v1's variance (sum of per-game variance) | the only family with positive order-persistence skill besides the bootstrap; cheaper than the bootstrap and no worse |
| Tie width | z 0.35-1.0 | 0.5 | the largest width that does not let a team climb far above its strength-of-record order (earlier face-validity checks) |
| Ranges shown | tie range | "could rank anywhere from #a to #b", or "firm" | same machinery as the order, always contains the shown rank |

## 9. Deviations from the protocol (and errors found)

* **Decision rule 2 was infeasible.** It required top-25 contradictions at or below 3% on tune; no arm, including v1 (7.1%),
  achieves that. It was written without a feasibility check. Parameters were instead chosen by the stated principle
  (D3 operationalised as "the narrowest win is worth at least half a blowout", lambda <= 0.5), which was fixed after seeing
  results and is flagged as such above.
* **M5 (games-played bias) was dropped as a metric.** It is mechanical: a sum grows with games, a mean does not, so the
  metric measures the definition, not the behaviour. The fairness property is covered by a property test (a mean is
  unchanged by the number of games).
* **v1's reliability is a proxy** (the reliability of the additive credit it is built on); its coherence, accuracy and
  persistence are measured on the shipped code.
* **Round 2 is exploratory** (section 7.2), not pre-registered. Round 3 (7.5) is pre-registered and was run once.
* **A bug found by tests:** the played-result term of the consensus cost was oriented backwards, so a 0.64 pair did not
  reverse as designed; fixed before any result was computed. The Gauss-Hermite variance error was found by a test and fixed.
* **ESPN returned 502 for some postseason weeks** in every arm identically.

## 10. Limitations

* The eval seasons (2022-25) were also used to tune the headline; differences claimed here are relative to v1 on identical
  data, not claims about the world.
* Forward accuracy is a sanity check for a resume, not its objective. A resume that predicted better would not thereby be
  a better resume; a worse one is a warning.
* All metrics are noisy: board-level resamples give intervals of about +/-0.6 points on accuracy and +/-2 on top-25
  contradictions. Differences smaller than that are not evidence.
* Persistence calibration compares each arm's order with its own order three weeks later, so it measures stability
  as much as truth.
* The yardstick treats FCS opponents as an offset below FBS and rates them from few games.

## 11. Recommendation

1. **Do not replace v1's architecture.** Its order rule and variance-based ranges are as good as or better than every
   alternative tested here. Posterior-draw uncertainty is the wrong uncertainty for a resume; consensus ordering and
   the standardised score add machinery without improving any measured quantity.
2. **Make the one real fix: a win never scores below zero, and raise the margin weight to 0.75.** Shipped 2026-10-14 after the round-3
   rule passed (section 7.5): about one point more accurate on the frozen seasons, reliability within 0.02, no negative wins.
   Effect on the 2026 board: top seven unchanged as a set, Georgia Tech's Duke win goes from slightly negative to 0.00.
3. **Know the three dials and who owns them:** margin weight (0.5 now, 0.75 is measurably better on accuracy), benchmark
   size (25 now, 40-70 is more reliable), tie width (0.5 now). They are values, not errors; each has a measured trade.
4. **Keep the evaluation harness.** Order-persistence calibration and the property tests are the reusable result: they
   caught the defect classes this report found, and they let the next proposed change be judged on evidence.

## 12. Reproduction

`research/resume_v2/README.md` lists the scripts and the exact commands (homelab, `pipeline` user, capped unit). Round 1 is
`exp_v2.py`, round 2 `exp_v2b.py`, analysis `analyze_v2.py` and `ci_v2.py`, robustness `extras_v2.py`, live board `live_v2.py`.
A full round takes about 30 seconds. Raw per-board results (`v2_rows.parquet`, `v2_pairs.parquet`, `v2b_*`) are kept on the
homelab under `work/resumeexp`.

## Changelog

* 2026-10-14: protocol locked (sections 1-5), then results (6-12) appended. Section 9 lists every departure from the protocol.
