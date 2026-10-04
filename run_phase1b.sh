#!/bin/bash
# Phase 1b (overnight, Windows): C3 fix check + MADQN seed variance.
#   proposed_v2_k0.5_seed7 : C3 with reward relabelling + greedy constraint estimates, outage target 0.20
#   no_c3_k0.5_seed7       : C1 + C2 + C4 without the dual constraints
#   madqn_seed8, madqn_seed9
# After the runs finish, everything is evaluated on the validation seeds and pushed.
cd ~/Desktop/Agentic_AI_HO
source .venv/Scripts/activate
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p logs

log "git pull"; git pull
python train_twc.py --variant proposed --seed 7 --kappa 0.5 --outdir outputs/twc/proposed_v2_k0.5_seed7 > logs/proposed_v2.log 2>&1 &
python train_twc.py --variant no_c3 --seed 7 --kappa 0.5 --outdir outputs/twc/no_c3_k0.5_seed7 > logs/no_c3.log 2>&1 &
python train_twc.py --variant madqn --seed 8 > logs/madqn_s8.log 2>&1 &
python train_twc.py --variant madqn --seed 9 > logs/madqn_s9.log 2>&1 &

# Keep the runs on performance cores (logical CPUs 0-15 of the i7-12700) at above-normal priority.
sleep 60
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | ForEach-Object { \$_.PriorityClass = 'AboveNormal'; \$_.ProcessorAffinity = 0xFFFF }"
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | Select-Object Id,PriorityClass,ProcessorAffinity | Format-Table"
log "training started"
wait
log "training finished"

python eval_twc.py --runs outputs/twc/proposed_v2_k0.5_seed7 outputs/twc/no_c3_k0.5_seed7 \
  outputs/twc/madqn_seed8 outputs/twc/madqn_seed9 \
  --seeds 20 21 22 --episodes 4 --conditions nominal stress --workers 8 --tag phase1b
git pull
git add outputs/twc/proposed_v2_k0.5_seed7 outputs/twc/no_c3_k0.5_seed7 outputs/twc/madqn_seed8 \
  outputs/twc/madqn_seed9 outputs/twc/eval/phase1b.jsonl run_phase1b.sh
git commit -m "Phase 1b: C3 fix (relabelling, greedy estimates), no-C3 ablation, MADQN seeds 8-9" && git push
log "=== DONE ==="
