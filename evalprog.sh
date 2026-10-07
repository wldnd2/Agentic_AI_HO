#!/bin/bash
# Usage: bash evalprog.sh [tag]        (default tag: scale_curve)
#   progress of outputs/twc/eval/<tag>.partial.jsonl; the ETA uses the rate
#   since the previous call, so run it twice a few minutes apart.
python - "${1:-scale_curve}" <<'PY'
import json, os, re, sys, time
tag = sys.argv[1]
part = f"outputs/twc/eval/{tag}.partial.jsonl"
final = f"outputs/twc/eval/{tag}.jsonl"
log = f"logs/{tag}.log"
total = None
if os.path.exists(log):
    for line in open(log, encoding="utf-8", errors="ignore"):
        m = re.search(r"\[(\d+)/(\d+)\]", line) or re.search(r"(\d+) of (\d+) jobs", line)
        if m: total = int(m.group(2))
done = 0
if os.path.exists(part):
    for line in open(part, encoding="utf-8", errors="ignore"):
        try: json.loads(line); done += 1
        except Exception: pass
if os.path.exists(final) and total and done >= total:
    print(f"{tag}: finished ({done}/{total})"); sys.exit()
cache = f".evalprog_{tag}.json"
now = time.time()
prev = json.load(open(cache)) if os.path.exists(cache) else None
json.dump({"t": now, "done": done}, open(cache, "w"))
head = f"{tag}: {done}/{total or '?'} jobs" + (f" ({100*done/total:.0f}%)" if total else "")
if prev and done > prev["done"] and total:
    rate = (done - prev["done"]) / (now - prev["t"])          # jobs per second
    print(f"{head}, ~{(total - done) / rate / 3600:.1f} h left "
          f"({(done - prev['done'])} jobs in the last {(now - prev['t'])/60:.0f} min)")
elif prev and done == prev["done"]:
    age = (now - os.path.getmtime(part)) / 60 if os.path.exists(part) else float("nan")
    print(f"{head}, no new job since the last check (last result {age:.0f} min ago)")
else:
    print(f"{head}, run again in a few minutes for an ETA")
PY
