#!/bin/bash
# Usage: bash trend.sh              -> all runs
#        bash trend.sh sig0.05 mappo -> only runs whose name contains one of the words
python - "$@" <<'PY'
import csv, glob, re, sys
import numpy as np
from pathlib import Path
filters = sys.argv[1:]
groups = {}
for path in sorted(glob.glob("outputs/twc/*seed*/training_metrics.csv")):
    name = Path(path).parent.name
    if filters and not any(f in name for f in filters):
        continue
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 20:
        continue
    m = re.match(r"(.*)_seed(\d+)$", name)
    group, seed = (m.group(1), int(m.group(2))) if m else (name, -1)
    last = rows[-50:]
    def mean(key, fallback=None):
        vals = [float(r[key]) for r in last if r.get(key) not in (None, "")]
        if vals: return float(np.mean(vals))
        return mean(fallback) if fallback else float("nan")
    def final(key):
        v = rows[-1].get(key)
        return float(v) if v not in (None, "") else float("nan")
    groups.setdefault(group, []).append((seed, len(rows), mean("common_reward"),
        mean("greedy_outage", "outage"), mean("greedy_signaling", "signaling"),
        final("lambda_sig"), final("lambda_out")))
fmt = lambda x, w=7, d=3: (" " * (w - 3) + "  -") if np.isnan(x) else f"{x:{w}.{d}f}"
print(f"{'group':30s} {'seed':>4s} {'iter':>5s}  {'common':>7s} {'outage':>7s} {'signal':>7s} {'lam_sig':>7s} {'lam_out':>7s}")
for group, items in groups.items():
    for seed, n, c, o, s, ls, lo in sorted(items):
        print(f"{group:30s} {seed:4d} {n:5d}  {fmt(c,7,4)} {fmt(o)} {fmt(s)} {fmt(ls)} {fmt(lo)}")
    if len(items) > 1:
        arr = np.array([[c, o, s] for _, _, c, o, s, _, _ in items])
        mu, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        print(f"{'  mean +- std':30s} {'':4s} {'':5s}  {mu[0]:7.4f} {mu[1]:7.3f} {mu[2]:7.3f}   (common std {sd[0]:.4f})")
    print()
print("Values: mean of the last 50 training iterations (5% exploration included).")
print("outage/signal = non-exploring agents when logged (EG-CMADQN family), behaviour policy otherwise.")
print("Final comparisons must use eval_twc.py, not these training numbers.")
PY
