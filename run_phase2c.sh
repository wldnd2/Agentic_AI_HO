#!/bin/bash
# Phase 2c: seeds 10, 11 of the 48-UE baselines (MADQN, MAPPO, fixed penalty) -> 5 seeds each.
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
for s in 10 11; do
  start madqn   $s outputs/twc/u48_madqn_seed$s
  start mappo   $s outputs/twc/u48_mappo_seed$s
  start c3_only $s outputs/twc/u48_penalty_seed$s --signaling-target 0.05 --fixed-lambda-out 0.6 --fixed-lambda-sig 1.0
done
sleep 60
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | ForEach-Object { \$_.PriorityClass = 'AboveNormal'; \$_.ProcessorAffinity = 0xFFFF }"
log "training started: ${#RUNS[@]} runs"
wait
log "training finished"
git pull
git add "${RUNS[@]}" run_phase2c.sh
git commit -m "Phase 2c: 48-UE baselines seeds 10-11" && git push
log "=== DONE ==="
