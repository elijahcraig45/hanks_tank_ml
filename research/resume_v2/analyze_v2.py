import sys, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

W = sys.argv[1]; PREFIX = sys.argv[2] if len(sys.argv) > 2 else "v2"; PROXY = sys.argv[3] if len(sys.argv) > 3 else "A add 0.5"
rows = pd.read_parquet(f"{W}/{PREFIX}_rows.parquet"); pairs = pd.read_parquet(f"{W}/{PREFIX}_pairs.parquet")
# v1's reliability is the reliability of its credit (additive, lam .5): use that arm's value as the proxy
proxy = rows[rows.arm == PROXY].set_index(["season", "week"]).rel
m = rows.arm == "v1 shipped"
rows.loc[m, "rel"] = [proxy.get((s, w), np.nan) for s, w in zip(rows[m].season, rows[m].week)]

def auc(p, y):
    if y.nunique() < 2: return np.nan
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def summary(part):
    out = []
    for arm, d in rows[rows.part == part].groupby("arm", sort=False):
        pr = pairs[(pairs.part == part) & (pairs.arm == arm)]
        base = pr.persist.mean() if len(pr) else np.nan
        brier = ((pr.p - pr.persist) ** 2).mean() if len(pr) else np.nan
        out.append(dict(arm=arm, acc=1 - d.fw.sum() / d.fn.sum(), contra=d.c.sum() / d.n.sum(), top25=d.tc.sum() / max(d.tn.sum(), 1),
                        rel=d.rel.mean(), width=d.width.mean(), brier=brier, bss=1 - brier / (base * (1 - base)) if base == base else np.nan,
                        auc=auc(pr.p, pr.persist) if len(pr) else np.nan))
    return pd.DataFrame(out).set_index("arm")

def boot_diff(part, arm, ref, metric, n=2000, seed=0):
    """Paired block bootstrap over (season, week) boards of metric(arm) - metric(ref)."""
    a = rows[(rows.part == part) & (rows.arm == arm)].set_index(["season", "week"])
    b = rows[(rows.part == part) & (rows.arm == ref)].set_index(["season", "week"])
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    rng = np.random.default_rng(seed)
    def stat(sel):
        if metric == "acc": return (1 - a.fw.values[sel].sum() / a.fn.values[sel].sum()) - (1 - b.fw.values[sel].sum() / b.fn.values[sel].sum())
        if metric == "top25": return a.tc.values[sel].sum() / max(a.tn.values[sel].sum(), 1) - b.tc.values[sel].sum() / max(b.tn.values[sel].sum(), 1)
        if metric == "rel": return np.nanmean(a.rel.values[sel]) - np.nanmean(b.rel.values[sel])
    full = np.arange(len(idx))
    boots = [stat(rng.integers(0, len(idx), len(idx))) for _ in range(n)]
    return stat(full), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

if __name__ == "__main__":
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100)
    for part in ("tune", "eval"):
        s = summary(part)
        print(f"\n=========== {part.upper()} ===========")
        print(s.round(4).to_string())
    s = {p: summary(p) for p in ("tune", "eval")}
    s["tune"].to_csv(f"{W}/{PREFIX}_summary_tune.csv"); s["eval"].to_csv(f"{W}/{PREFIX}_summary_eval.csv")
