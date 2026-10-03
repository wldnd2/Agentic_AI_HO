#!/bin/bash
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "=== A3 ===";  python scripts/twc_cho_sweep.py --episodes 2 --workers 6 --offsets 0.5 1.0 1.5 2.0 3.0 --ttts 1 2 --tag _v2 | tail -12
echo "=== T1 ===";  python scripts/twc_cho_sweep.py --policy t1 --episodes 2 --workers 6 --t1-elev 65 70 75 --t1-lead 15 --rules geometry --tag _v2 | tail -5
echo "=== D2 ===";  python scripts/twc_cho_sweep.py --policy d2 --episodes 2 --workers 6 --d2-th1 300 400 --d2-th2 1300 1500 --rules load --tag _v2 | tail -6
echo "=== EVAL ==="; python eval_twc.py --methods a3 t1 d2 maxsinr --seeds 20 21 22 --episodes 4 --conditions nominal stress --workers 6 --tag rules_validation
echo "=== DONE ==="
