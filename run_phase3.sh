#!/bin/bash
# Phase 3 (Windows): ablations of the final model at 48 UEs, then the final evaluation
# on held-out evaluation seeds 101-105 (never used for tuning or selection).
cd ~/Desktop/Agentic_AI_HO
source .venv/Scripts/activate
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "git pull"; git pull
POP="--ues 48 --rate 0.6"
FD="--kappa 0.5 --dual-lr-sig 0.02 --signaling-target 0.05"
NEW=()
start() {
  local v=$1 s=$2 out=$3; shift 3
  NEW+=("$out")
  if [ -f "$out/egcmadqn_weights.npz" ]; then log "skip (finished): $out"; return; fi
  python train_twc.py --variant "$v" --seed "$s" --outdir "$out" $POP $FD "$@" >> "logs/$(basename "$out").log" 2>&1 &
}
for s in 7 8 9; do
  for v in no_c1 no_c2 no_c3 no_c4; do start $v $s outputs/twc/u48_fd_${v}_seed$s; done
done
sleep 60
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | ForEach-Object { \$_.PriorityClass = 'AboveNormal'; \$_.ProcessorAffinity = 0xFFFF }"
log "ablation training started: ${#NEW[@]} runs"
wait
log "ablation training finished"
git pull; git add "${NEW[@]}" run_phase3.sh; git commit -m "Phase 3: 48-UE ablations (no C1/C2/C3/C4), seeds 7-9"; git push

R=outputs/twc
RUNS="$(echo $R/u48_fastdual_seed{7,8,9,10,11} $R/u48_madqn_seed{7,8,9,10,11} $R/u48_mappo_seed{7,8,9,10,11} \
             $R/u48_penalty_seed{7,8,9,10,11} $R/u48_sigonly_seed{7,8,9} $R/u48_proposed_sig0.05_seed{7,8,9} \
             $R/u48_fd_no_c{1,2,3,4}_seed{7,8,9})"
COMMON="--runs $RUNS --methods a3 t1 d2 maxsinr --seeds 101 102 103 104 105 --episodes 4 --load total --workers 10"
log "final evaluation (nominal)"
python eval_twc.py $COMMON --conditions nominal --ues 24 48 96 --tag final_nominal
log "final evaluation (stress)"
python eval_twc.py $COMMON --conditions stress --ues 48 --tag final_stress
git pull
git add outputs/twc/eval/final_nominal.jsonl outputs/twc/eval/final_stress.jsonl
git commit -m "Final evaluation on held-out seeds 101-105" && git push
log "=== DONE ==="
