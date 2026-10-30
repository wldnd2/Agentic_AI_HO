#!/bin/bash
# Phase 4 (Windows): mixed-criticality scenario (25% critical UEs), 48 UEs, seeds 7-9.
#   proposed       : C2 + C3 + C4, outage dual on the CRITICAL-UE outage (target 0.05), signalling budget 0.05
#   proposed_nodup : same, PDCP duplication removed from the action set (ablation)
#   madqn, mappo   : same scenario (class flag in the observation, class-weighted reward)
# Then held-out evaluation (seeds 101-105) at 48 and 96 UEs. Re-running resumes.
cd ~/Desktop/Agentic_AI_HO
source .venv/Scripts/activate
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "git pull"; git pull
POP="--ues 48 --rate 0.6"
PROP="--kappa 0.5 --dual-lr-sig 0.02 --signaling-target 0.05 --outage-metric critical_outage_ratio --outage-target 0.05"
MAIN=(); NODUP=()
start() { # list variant seed outdir [extra args]
  local list=$1 v=$2 s=$3 out=$4; shift 4
  eval "$list+=(\"$out\")"
  if [ -f "$out/egcmadqn_weights.npz" ] || [ -f "$out/mappo_weights.npz" ]; then log "skip (finished): $out"; return; fi
  python train_twc.py --variant "$v" --seed "$s" --outdir "$out" $POP "$@" >> "logs/$(basename "$out").log" 2>&1 &
}
for s in 7 8 9; do
  start MAIN  no_c1 $s outputs/twc/mix_proposed_seed$s --scenario mixed $PROP
  start NODUP no_c1 $s outputs/twc/mix_proposed_nodup_seed$s --scenario mixed_nodup $PROP
  start MAIN  madqn $s outputs/twc/mix_madqn_seed$s --scenario mixed
  start MAIN  mappo $s outputs/twc/mix_mappo_seed$s --scenario mixed
done
sleep 60
powershell -Command "Get-Process python | Where-Object {\$_.CPU -gt 5} | ForEach-Object { \$_.PriorityClass = 'AboveNormal'; \$_.ProcessorAffinity = 0xFFFF }"
log "training started: $(( ${#MAIN[@]} + ${#NODUP[@]} )) runs"
wait
log "training finished"
COMMON="--seeds 101 102 103 104 105 --episodes 4 --ues 48 96 --load total --conditions nominal --workers 10"
python eval_twc.py --scenario mixed --runs "${MAIN[@]}" --methods a3 a3critdup a3dup t1 maxsinr $COMMON --tag mixed_final
python eval_twc.py --scenario mixed_nodup --runs "${NODUP[@]}" $COMMON --tag mixed_nodup_final
git pull
git add "${MAIN[@]}" "${NODUP[@]}" outputs/twc/eval/mixed_final.jsonl outputs/twc/eval/mixed_nodup_final.jsonl run_phase4.sh
git commit -m "Phase 4: mixed-criticality scenario (proposed, no-dup ablation, MADQN, MAPPO; seeds 7-9)" && git push
log "=== DONE ==="
