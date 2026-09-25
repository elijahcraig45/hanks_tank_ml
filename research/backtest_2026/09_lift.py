"""Correct the window confound: score every rule as LIFT over the same-window baseline.

08 found rules clearing 60% on validation, but almost everything improved there --
the late-season window is simply easier (V10 itself jumps). A slice is only
interesting if it beats what the model already does *in that same period*, and if
it does so in BOTH windows. Anything good in one window only is noise.
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = pd.read_csv("data/backtest_2026/rule_search.csv")
exec(open("research/backtest_2026/06_headtohead.py").read().split("out, diag = run()")[0])
out, diag = run(min_train=400, cadence_days=7, C=0.1)
mask = ~np.isnan(out["logit3"]); D = d.loc[mask].reset_index(drop=True); yv = y[mask]
pv10 = v10[mask]
n = len(D); cut = int(n*0.60)
disc = np.zeros(n, bool); disc[:cut] = True; val = ~disc
corr = (pv10 >= .5).astype(int) == yv

bd, bv = corr[disc].mean()*100, corr[val].mean()*100
print(f"WINDOW BASELINES (V10 accuracy on all games in the window)")
print(f"  discovery  n={disc.sum():4d}  V10 acc={bd:.2f}%   home rate={yv[disc].mean()*100:.2f}%")
print(f"  validation n={val.sum():4d}  V10 acc={bv:.2f}%   home rate={yv[val].mean()*100:.2f}%")
print(f"  -> the validation window is {bv-bd:+.2f}pp easier for V10 before any slicing\n")

R["d_lift"] = R.d_acc - bd
R["v_lift"] = R.v_acc - bv
R["consistent"] = (R.d_lift > 0) & (R.v_lift > 0)

print("="*104)
print("RULES WITH POSITIVE LIFT IN *BOTH* WINDOWS (the only survivors), sorted by validation lift")
print("="*104)
K = R[R.consistent & (R.v_n >= 100)].sort_values("v_lift", ascending=False)
if len(K):
    print(K[["rule","d_n","d_acc","d_lift","v_n","v_acc","v_lift","v_lo","v_hi"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
else:
    print("  none")

print(f"\n{R.consistent.sum()} of {len(R)} rules show positive lift in both windows "
      f"(pure chance would give ~{len(R)*0.25:.0f} if lift were random)")

print("\n" + "="*104)
print("SANITY CHECK -- the 60%+ validation rules from 08, re-scored as lift")
print("="*104)
chk = ["|sp_quality_diff| >= q70","stack conf>=0.55","|sp_quality_diff| >= q60 & agree2",
       "series opener","v10 conf>=0.55 & home-fav","v10 conf>=0.6",
       "v10+logit3+elo agree & v10conf>=0.5","v10 conf>=0.64"]
sub = R[R.rule.isin(chk)][["rule","d_n","d_acc","d_lift","v_n","v_acc","v_lift"]]
print(sub.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("\n  Read the d_lift column: most of these were at or BELOW baseline during discovery.")
print("  A slice that only works in the second window is a coin that landed heads twice.")
