#!/bin/bash
# Usage: bash progress.sh        -> unfinished runs (running or stopped)
#        bash progress.sh --all  -> all runs
python - "$@" <<'PY'
import csv, glob, sys, time, os
from pathlib import Path
show_all = "--all" in sys.argv
done_count, running, stopped = 0, 0, 0
for path in sorted(glob.glob("outputs/twc/*seed*/training_metrics.csv")):
    rows = list(csv.DictReader(open(path)))
    if not rows: continue
    n = len(rows)
    sec = sum(float(r["seconds"]) for r in rows[-20:]) / len(rows[-20:])
    name = Path(path).parent.name
    age_min = (time.time() - os.path.getmtime(path)) / 60
    if n >= 2000:
        done_count += 1
        if show_all: print(f"{name:34s} {n:4d}/2000  done")
        continue
    if age_min > max(5, 3 * sec / 60):
        stopped += 1
        print(f"{name:34s} {n:4d}/2000  STOPPED (last write {age_min:.0f} min ago)")
    else:
        running += 1
        print(f"{name:34s} {n:4d}/2000  ({sec:4.1f} s/it)  ~{(2000 - n) * sec / 3600:.1f} h left")
print(f"\nrunning {running}, stopped {stopped}, done {done_count}")
PY
