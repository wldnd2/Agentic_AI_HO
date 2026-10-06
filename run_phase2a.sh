#!/bin/bash
# Phase 2a (Windows): learning-based SOTA comparison under the signalling budget.
#   c3_c4, budget 0.05                 seeds 7,8,9   -> are C1/C2 needed at the final operating point?
#   proposed, budget 0.05              seeds 10,11   -> 5 seeds of the final model
#   MAPPO (Li 2026-style CTDE)         seeds 8-11    -> 5 MAPPO seeds
#   hysteretic MADQN (Liu 2024 update) seeds 7,8,9   -> second value-based SOTA
#   MADQN + fixed penalty              seeds 7,8,9   -> "why not a weighted penalty?" (lambdas = learned values)
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
for s in 7 8 9;    do start c3_c4    $s outputs/twc/c3_c4_sig0.05_seed$s --signaling-target 0.05; done
for s in 10 11;    do start proposed $s outputs/twc/proposed_v2_sig0.05_seed$s --kappa 0.5 --signaling-target 0.05; done
for s in 8 9 10 11; do start mappo   $s outputs/twc/mappo_seed$s; done
for s in 7 8 9;    do start madqn    $s outputs/twc/hmadqn_b0.5_seed$s --hysteretic-beta 0.5; done
for s in 7 8 9;    do start c3_only  $s outputs/twc/penalty_seed$s --signaling-target 0.05 \
                        --fixed-lambda-out 0.6 --fixed-lambda-sig 1.0; done

sleep 60
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | ForEach-Object { \$_.PriorityClass = 'AboveNormal'; \$_.ProcessorAffinity = 0xFFFF }"
log "training started: ${#RUNS[@]} runs"
wait
log "training finished"

python eval_twc.py --runs "${RUNS[@]}" --seeds 20 21 22 --episodes 4 --conditions nominal stress \
  --workers 10 --tag phase2a
git pull
git add "${RUNS[@]}" outputs/twc/eval/phase2a.jsonl run_phase2a.sh
git commit -m "Phase 2a: SOTA baselines under the signalling budget (MAPPO x5, hysteretic MADQN, fixed penalty), C3+C4@0.05, proposed@0.05 seeds 10-11" && git push
log "=== DONE ==="
