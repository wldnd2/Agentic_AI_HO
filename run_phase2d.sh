#!/bin/bash
# Phase 2d (Windows): fix the signalling-budget violation at 48 UEs.
#   sigonly : signalling constraint only (outage target 1.0 = off), dual lr 0.02   seeds 7, 8, 9
#   fastdual: both constraints (outage 0.20), signalling dual lr 0.02             seeds 7, 8, 9
cd ~/Desktop/Agentic_AI_HO
source .venv/Scripts/activate
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "git pull"; git pull
POP="--ues 48 --rate 0.6"
RUNS=()
start() {
  local v=$1 s=$2 out=$3; shift 3
  RUNS+=("$out")
  if [ -f "$out/egcmadqn_weights.npz" ] || [ -f "$out/mappo_weights.npz" ]; then log "skip (finished): $out"; return; fi
  python train_twc.py --variant "$v" --seed "$s" --outdir "$out" $POP "$@" >> "logs/$(basename "$out").log" 2>&1 &
}
for s in 7 8 9; do
  start proposed $s outputs/twc/u48_sigonly_seed$s  --kappa 0.5 --signaling-target 0.05 --outage-target 1.0 --dual-lr-sig 0.02
  start proposed $s outputs/twc/u48_fastdual_seed$s --kappa 0.5 --signaling-target 0.05 --dual-lr-sig 0.02
done
sleep 60
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | ForEach-Object { \$_.PriorityClass = 'AboveNormal'; \$_.ProcessorAffinity = 0xFFFF }"
log "training started: ${#RUNS[@]} runs"
wait
log "training finished"
python eval_twc.py --runs "${RUNS[@]}" --seeds 20 21 22 --episodes 4 --ues 24 48 96 --load total \
  --workers 10 --tag phase2d_u48
git pull
git add "${RUNS[@]}" outputs/twc/eval/phase2d_u48.jsonl run_phase2d.sh
git commit -m "Phase 2d: 48-UE proposed with signalling-only constraint / faster dual (seeds 7-9)" && git push
log "=== DONE ==="
