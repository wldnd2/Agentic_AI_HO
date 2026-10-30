#!/bin/bash
# Phase 3b (Windows): final model = C2 + C3 + C4 (no C1). Seeds 10, 11 at 48 UEs,
# then the same held-out evaluation as Phase 3 for the two new seeds.
cd ~/Desktop/Agentic_AI_HO
source .venv/Scripts/activate
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "git pull"; git pull
NEW=()
for s in 10 11; do
  out=outputs/twc/u48_fd_no_c1_seed$s; NEW+=("$out")
  if [ -f "$out/egcmadqn_weights.npz" ]; then log "skip (finished): $out"; continue; fi
  python train_twc.py --variant no_c1 --seed $s --outdir $out --ues 48 --rate 0.6 \
    --kappa 0.5 --dual-lr-sig 0.02 --signaling-target 0.05 >> logs/$(basename $out).log 2>&1 &
done
sleep 60
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | ForEach-Object { \$_.PriorityClass = 'AboveNormal'; \$_.ProcessorAffinity = 0xFFFF }"
log "training started"
wait
log "training finished"
COMMON="--runs ${NEW[*]} --seeds 101 102 103 104 105 --episodes 4 --load total --workers 10"
python eval_twc.py $COMMON --conditions nominal --ues 24 48 96 --tag final_noc1_nominal
python eval_twc.py $COMMON --conditions stress --ues 48 --tag final_noc1_stress
git pull
git add "${NEW[@]}" outputs/twc/eval/final_noc1_*.jsonl run_phase3b.sh
git commit -m "Phase 3b: final model (no C1) seeds 10-11, held-out evaluation" && git push
log "=== DONE ==="
