#!/bin/bash
python - <<'PY'
import csv, glob, re
import numpy as np
from pathlib import Path
groups = {}
for path in sorted(glob.glob("outputs/twc/proposed_v2_sig*_seed*/training_metrics.csv")):
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 20: continue
    name = Path(path).parent.name
    group = re.sub(r"_seed\d+$", "", name)
    last = rows[-50:]
    f = lambda k: np.mean([float(r[k]) for r in last if r.get(k) not in (None, "")])
    groups.setdefault(group, []).append((name[-1], len(rows), f("common_reward"), f("greedy_outage"),
                                         f("greedy_signaling"), float(rows[-1].get("lambda_sig") or 0), float(rows[-1].get("lambda_out") or 0)))
print(f"{'group':30s} seed  iter   common  g-outage  g-signal  lam_sig  lam_out")
for g, items in groups.items():
    for seed, n, c, o, s, ls, lo in items:
        print(f"{g:30s}  {seed}   {n:4d}  {c:7.4f}  {o:7.3f}   {s:7.3f}   {ls:6.3f}  {lo:6.3f}")
    print()
PY
