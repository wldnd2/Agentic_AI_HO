#!/bin/bash
# Phase 2e (Windows): final model = fastdual (outage target 0.20 + signalling budget, signalling dual lr 0.02), 48 UEs.
#   budget 0.05: seeds 10, 11 (-> 5 seeds with Phase 2d)
#   budgets 0.025 / 0.10 / 0.15: seeds 7, 8, 9 (Pareto frontier)
cd ~/Desktop/Agentic_AI_HO
source .venv/Scripts/activate
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "git pull"; git pull
POP="--ues 48 --rate 0.6"
FD="--kappa 0.5 --dual-lr-sig 0.02"
RUNS=()
start() {
  local v=$1 s=$2 out=$3; shift 3
  RUNS+=("$out")
  if [ -f "$out/egcmadqn_weights.npz" ]; then log "skip (finished): $out"; return; fi
  python train_twc.py --variant "$v" --seed "$s" --outdir "$out" $POP $FD "$@" >> "logs/$(basename "$out").log" 2>&1 &
}
for s in 10 11; do start proposed $s outputs/twc/u48_fastdual_seed$s --signaling-target 0.05; done
for b in 0.025 0.10 0.15; do
  for s in 7 8 9; do start proposed $s outputs/twc/u48_fastdual_sig${b}_seed$s --signaling-target $b; done
done
sleep 60
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | ForEach-Object { \$_.PriorityClass = 'AboveNormal'; \$_.ProcessorAffinity = 0xFFFF }"
log "training started: ${#RUNS[@]} runs"
wait
log "training finished"
python eval_twc.py --runs "${RUNS[@]}" --seeds 20 21 22 --episodes 4 --ues 24 48 96 --load total \
  --workers 10 --tag phase2e_u48
git pull
git add "${RUNS[@]}" outputs/twc/eval/phase2e_u48.jsonl run_phase2e.sh
git commit -m "Phase 2e: fastdual seeds 10-11 and budget sweep (0.025/0.10/0.15), 48 UEs" && git push
log "=== DONE ==="
