#!/bin/bash
# Phase 1d (Windows): does a lower signalling budget steer every seed into the
# low-signalling / low-outage region that unconstrained MADQN reaches only in
# 2 of 5 seeds (seeds 7, 11: signalling ~0.06, best rewards)?
#   proposed v2, signalling budget 0.05 / 0.08 / 0.12          (seeds 7, 8, 9)
#   proposed v2, budget 0.08, outage constraint off (target 1.0) (seeds 7, 8, 9)
cd ~/Desktop/Agentic_AI_HO
source .venv/Scripts/activate
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p logs
log "git pull"; git pull

RUNS=()
start() { # outdir seed [extra args]
  local out=$1 s=$2; shift 2
  python train_twc.py --variant proposed --seed "$s" --kappa 0.5 --outdir "$out" "$@" \
    > "logs/$(basename "$out").log" 2>&1 &
  RUNS+=("$out")
}
for s in 7 8 9; do
  for b in 0.05 0.08 0.12; do
    start "outputs/twc/proposed_v2_sig${b}_seed$s" "$s" --signaling-target "$b"
  done
  start "outputs/twc/proposed_v2_sig0.08_noout_seed$s" "$s" --signaling-target 0.08 --outage-target 1.0
done

sleep 60
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | ForEach-Object { \$_.PriorityClass = 'AboveNormal'; \$_.ProcessorAffinity = 0xFFFF }"
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | Select-Object Id,PriorityClass,ProcessorAffinity | Format-Table"
log "training started: ${#RUNS[@]} runs"
wait
log "training finished"

python eval_twc.py --runs "${RUNS[@]}" --seeds 20 21 22 --episodes 4 --conditions nominal stress \
  --workers 10 --tag phase1d
git pull
git add "${RUNS[@]}" outputs/twc/eval/phase1d.jsonl run_phase1d.sh
git commit -m "Phase 1d: signalling-budget sweep (0.05/0.08/0.12) and signalling-only constraint, seeds 7-9" && git push
log "=== DONE ==="
