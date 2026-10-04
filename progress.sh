#!/bin/bash
python - <<'PY'
import csv, glob
from pathlib import Path
for path in sorted(glob.glob("outputs/twc/*seed*/training_metrics.csv")):
    rows = list(csv.DictReader(open(path)))
    if not rows: continue
    done = len(rows)
    recent = [float(r["seconds"]) for r in rows[-20:]]
    sec = sum(recent) / len(recent)
    left_h = (2000 - done) * sec / 3600
    status = "완료" if done >= 2000 else f"남은 시간 약 {left_h:.1f} h"
    print(f"{Path(path).parent.name:24s} {done:4d}/2000  ({sec:4.1f} s/회)  {status}")
PY
