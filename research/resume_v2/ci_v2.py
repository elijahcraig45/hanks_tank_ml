import sys
sys.argv = ["analyze_v2.py", sys.argv[1], "v2b", "R0 add .5 sum v1order"]
sys.path.insert(0, sys.argv[1])
import analyze_v2 as A
import pandas as pd
ref = "v1 shipped"
arms = ["R0 add .5 sum v1order", "R1 addfloor .5 sum v1order", "R2 mult 0.5 sum v1order", "R2 mult 0.75 sum v1order",
        "R3 mult .5 z v1order z=0.5", "R4 mult .5 sum cons eta=2.0", "R4 mult .5 sum cons eta=1.0"]
print(f"{'arm vs v1 shipped':34s} {'part':5s} {'acc diff (pts)':>26s} {'top25 diff (pts)':>26s} {'reliability diff':>22s}")
for arm in arms:
    for part in ("tune", "eval"):
        a = A.boot_diff(part, arm, ref, "acc"); t = A.boot_diff(part, arm, ref, "top25"); r = A.boot_diff(part, arm, ref, "rel")
        f = lambda x, k=100: f"{x[0]*k:+.2f} [{x[1]*k:+.2f},{x[2]*k:+.2f}]"
        print(f"{arm:34s} {part:5s} {f(a):>26s} {f(t):>26s} {f(r, 1):>22s}")
# credit comparison with the same ordering: mult vs addfloor
print()
for part in ("tune", "eval"):
    for k in ("acc", "top25", "rel"):
        d = A.boot_diff(part, "R2 mult 0.5 sum v1order", "R1 addfloor .5 sum v1order", k)
        print(f"mult .5 minus addfloor .5, {part}, {k}: {d[0]*(1 if k=='rel' else 100):+.3f} [{d[1]*(1 if k=='rel' else 100):+.3f},{d[2]*(1 if k=='rel' else 100):+.3f}]")
