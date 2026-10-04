#!/bin/bash
cd ~/Desktop/Agentic_AI_HO
source .venv/Scripts/activate
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
COMMON="--seeds 20 21 22 --episodes 4 --conditions nominal stress --workers 8"

log "git pull"; git pull

log "=== 1) proposed x3, MADQN, MAPPO, T1, A3 ==="
python eval_twc.py --runs outputs/twc/proposed_k0.1_seed7 outputs/twc/proposed_k0.5_seed7 \
  outputs/twc/proposed_k1.0_seed7 outputs/twc/madqn_seed7 outputs/twc/mappo_seed7 \
  --methods t1 a3 $COMMON --tag phase1_seed7

log "=== 2) MAPPO argmax ==="
python eval_twc.py --runs outputs/twc/mappo_seed7 $COMMON --mappo-deterministic --tag phase1_mappo_argmax

log "=== 3) waiting for MAPPO+C2-C4 (max 3 h) ==="
for i in $(seq 1 36); do
  [ -f outputs/twc/mappo_c234_seed7/mappo_weights.npz ] && break
  sleep 300
done
if [ -f outputs/twc/mappo_c234_seed7/mappo_weights.npz ]; then
  sleep 30
  python eval_twc.py --runs outputs/twc/mappo_c234_seed7 $COMMON --tag phase1_mappo_c234
  python eval_twc.py --runs outputs/twc/mappo_c234_seed7 $COMMON --mappo-deterministic --tag phase1_mappo_c234_argmax
  git add outputs/twc/mappo_c234_seed7
else
  log "MAPPO+C2-C4 did not finish within 3 h - check logs/mappo_c234.log"
fi

log "=== 4) commit ==="
git pull
git add outputs/twc/eval/phase1_*.jsonl
git commit -m "Phase 1: validation evaluations (seed 7), MAPPO+C2-C4" && git push
log "=== DONE ==="
