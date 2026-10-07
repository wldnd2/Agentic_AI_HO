#!/bin/bash
# Phase 2b (Windows): train directly in the dense regime (48 UEs, same total load
# 48 x 0.6 = 28.8 Mb/s) and evaluate at 24 / 48 / 96 UEs.
#   proposed (budget 0.05), MADQN, MAPPO, MADQN + fixed penalty   seeds 7, 8, 9
# Re-running this script after an interruption resumes everything.
cd ~/Desktop/Agentic_AI_HO
source .venv/Scripts/activate
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p logs
log "git pull"; git pull
POP="--ues 48 --rate 0.6"
RUNS=()
start() { # variant seed outdir [extra args]
  local v=$1 s=$2 out=$3; shift 3
  RUNS+=("$out")
  if [ -f "$out/egcmadqn_weights.npz" ] || [ -f "$out/mappo_weights.npz" ]; then
    log "skip (finished): $out"; return
  fi
  python train_twc.py --variant "$v" --seed "$s" --outdir "$out" $POP "$@" >> "logs/$(basename "$out").log" 2>&1 &
}
for s in 7 8 9; do
  start proposed $s outputs/twc/u48_proposed_sig0.05_seed$s --kappa 0.5 --signaling-target 0.05
  start madqn    $s outputs/twc/u48_madqn_seed$s
  start mappo    $s outputs/twc/u48_mappo_seed$s
  start c3_only  $s outputs/twc/u48_penalty_seed$s --signaling-target 0.05 --fixed-lambda-out 0.6 --fixed-lambda-sig 1.0
done
sleep 60
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | ForEach-Object { \$_.PriorityClass = 'AboveNormal'; \$_.ProcessorAffinity = 0xFFFF }"
log "training started: ${#RUNS[@]} runs"
wait
log "training finished"

python eval_twc.py --runs "${RUNS[@]}" --methods a3 t1 --seeds 20 21 22 --episodes 4 \
  --ues 24 48 96 --load total --workers 10 --tag phase2b_u48
git pull
git add "${RUNS[@]}" outputs/twc/eval/phase2b_u48.jsonl run_phase2b.sh
git commit -m "Phase 2b: training at 48 UEs (proposed, MADQN, MAPPO, fixed penalty; seeds 7-9)" && git push
log "=== DONE ==="
