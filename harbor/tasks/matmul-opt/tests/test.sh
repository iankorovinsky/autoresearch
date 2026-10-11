#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier
LOG=/logs/verifier/kernelbench.log
cd /opt/autoresearch/KernelBench

if [[ ! -f /workspace/candidate.py ]]; then
  echo "Candidate file /workspace/candidate.py was not created." | tee /logs/verifier/kernelbench.log
  echo "0.0" > /logs/verifier/reward.txt
  exit 0
fi

set +e
uv run python scripts/check_candidate.py 1 1 /workspace/candidate.py \
  --eval-mode local \
  --setting num_correct_trials=2 \
  --setting num_perf_trials=10 \
  --setting 'gpu_arch=["Turing"]' 2>&1 | tee "$LOG"
STATUS=$?
set -e

if [[ "$STATUS" -ne 0 ]] || ! grep -q 'correctness=True' "$LOG"; then
  echo "0.0" > /logs/verifier/reward.txt
  exit 0
fi

SPEEDUP=$(sed -nE 's/^\[Speedup\] Speedup over eager: ([0-9]+([.][0-9]+)?)x$/\1/p' "$LOG" | tail -n 1)
if [[ -z "$SPEEDUP" ]]; then
  echo "0.0" > /logs/verifier/reward.txt
  exit 0
fi

python3 - "$SPEEDUP" > /logs/verifier/reward.txt <<'PY'
import sys
speedup = float(sys.argv[1])
# Reward is 0 for incorrect kernels, otherwise capped at 1.0x eager speed.
print(f"{min(max(speedup, 0.0), 1.0):.4f}")
PY
