import sys
W = sys.argv[1]
sys.argv = ["analyze_v2.py", W, "v1x", "live"]
sys.path.insert(0, W)
import analyze_v2 as A
import pandas as pd
pd.set_option("display.width", 200)
for part in ("tune", "eval"):
    print(f"\n== {part} ==")
    print(A.summary(part).round(4).to_string())
ref = "live"
print(f"\n{'arm vs live':10s} {'part':5s} {'acc (pts)':>24s} {'top25 (pts)':>24s} {'reliability':>20s}  {'persistence skill diff':>22s}")
S = {p: A.summary(p) for p in ("tune", "eval")}
for arm in ("F.5", "F.75", "noF.75"):
    for part in ("tune", "eval"):
        a = A.boot_diff(part, arm, ref, "acc"); t = A.boot_diff(part, arm, ref, "top25"); r = A.boot_diff(part, arm, ref, "rel")
        f = lambda x, k=100: f"{x[0]*k:+.2f} [{x[1]*k:+.2f},{x[2]*k:+.2f}]"
        d_bss = S[part].loc[arm, "bss"] - S[part].loc[ref, "bss"]
        print(f"{arm:10s} {part:5s} {f(a):>24s} {f(t):>24s} {f(r, 1):>20s}  {d_bss:+.4f}")
