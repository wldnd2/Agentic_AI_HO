#!/bin/bash
# Phase 1c (overnight, Windows): decide the final model composition on 3+ seeds.
#   madqn seeds 10, 11                     -> 5 MADQN seeds in total
#   c3_only seeds 7, 8, 9                  -> constrained MADQN (no planner)
#   c3_c4   seeds 7, 8, 9                  -> constrained MADQN + congestion prices
#   proposed (v2, kappa 0.5) seeds 8, 9    -> 3 seeds of the full model
# All runs are evaluated on the validation seeds afterwards and pushed.
cd ~/Desktop/Agentic_AI_HO
source .venv/Scripts/activate
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p logs
log "git pull"; git pull

RUNS=()
start() { # variant seed outdir [extra args]
  local v=$1 s=$2 out=$3; shift 3
  python train_twc.py --variant "$v" --seed "$s" --outdir "$out" "$@" > "logs/$(basename "$out").log" 2>&1 &
  RUNS+=("$out")
}
for s in 10 11; do start madqn $s outputs/twc/madqn_seed$s; done
for s in 7 8 9; do start c3_only $s outputs/twc/c3_only_seed$s; done
for s in 7 8 9; do start c3_c4 $s outputs/twc/c3_c4_seed$s; done
for s in 8 9; do start proposed $s outputs/twc/proposed_v2_k0.5_seed$s --kappa 0.5; done

sleep 60
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | ForEach-Object { \$_.PriorityClass = 'AboveNormal'; \$_.ProcessorAffinity = 0xFFFF }"
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | Select-Object Id,PriorityClass,ProcessorAffinity | Format-Table"
log "training started: ${#RUNS[@]} runs"
wait
log "training finished"

python eval_twc.py --runs "${RUNS[@]}" --seeds 20 21 22 --episodes 4 --conditions nominal stress \
  --workers 10 --tag phase1c
git pull
git add "${RUNS[@]}" outputs/twc/eval/phase1c.jsonl run_phase1c.sh
git commit -m "Phase 1c: MADQN seeds 10-11, C3-only and C3+C4 (seeds 7-9), proposed v2 seeds 8-9" && git push
log "=== DONE ==="
