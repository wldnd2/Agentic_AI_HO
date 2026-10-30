"""An interrupted run resumed from its saved state must reproduce an uninterrupted run."""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEGACY_TWC = ROOT / "archive" / "legacy_twc" / "experiments"
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def _rows(path):
    with open(path, newline="") as handle:
        return [{k: v for k, v in r.items() if k != "seconds"} for r in csv.DictReader(handle)]


@pytest.mark.parametrize("variant", ["proposed", "mappo_c234"])
def test_resume_reproduces_uninterrupted_run(tmp_path, variant):
    base = [sys.executable, str(LEGACY_TWC / "train_twc.py"), "--variant", variant, "--seed", "7",
            "--iterations", "6", "--smoke", "--state-every", "2"]
    a, b = tmp_path / "a", tmp_path / "b"
    subprocess.run(base + ["--outdir", str(a)], cwd=ROOT, env=ENV, check=True, capture_output=True)
    proc = subprocess.Popen(base + ["--outdir", str(b)], cwd=ROOT, env=ENV,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        time.sleep(0.1)
        metrics = b / "training_metrics.csv"
        if (b / "resume_state.pkl").exists() and metrics.exists() and len(_rows(metrics)) >= 3:
            proc.kill()
            proc.wait()
            break
    out = subprocess.run(base + ["--outdir", str(b)], cwd=ROOT, env=ENV, check=True,
                         capture_output=True, text=True).stdout
    assert "resumed at iteration" in out
    assert _rows(a / "training_metrics.csv") == _rows(b / "training_metrics.csv")
    assert not (b / "resume_state.pkl").exists()
