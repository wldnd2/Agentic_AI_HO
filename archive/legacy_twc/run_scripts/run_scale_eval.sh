#!/bin/bash
cd ~/Desktop/Agentic_AI_HO
source .venv/Scripts/activate
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=outputs/twc
RUNS="$(echo $R/madqn_seed{7,8,9,10,11} $R/mappo_seed{7,8,9,10,11} $R/proposed_v2_sig0.05_seed{7,8,9,10,11} $R/penalty_seed{7,8,9} $R/c3_c4_sig0.05_seed{7,8,9})"
python eval_twc.py --runs $RUNS --methods a3 t1 --seeds 20 21 22 --episodes 4 \
  --ues 12 24 36 48 72 96 --load total --workers 10 --tag scale_curve
git pull
git add outputs/twc/eval/scale_total.jsonl outputs/twc/eval/scale_curve.jsonl run_scale_eval.sh
git commit -m "UE-density evaluation (constant total load, 12-96 UEs)" && git push
echo "=== DONE ==="
