#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier
LOG=/logs/verifier/pygpubench.log
RESULT=/logs/verifier/result.json

if [[ ! -f /workspace/candidate.py ]]; then
  echo "Candidate file /workspace/candidate.py was not created." | tee "$LOG"
  echo "0.0" > /logs/verifier/reward.txt
  exit 0
fi

cd /opt/autoresearch/pygpubench
python evaluate_pygpubench.py --task-id 1 --level 1 --kernel-path /workspace/candidate.py --output "$RESULT" 2>&1 | tee "$LOG"

if [[ ! -f "$RESULT" ]]; then
  echo "0.0" > /logs/verifier/reward.txt
  exit 0
fi

python - "$RESULT" > /logs/verifier/reward.txt <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))
speedup = result.get("speedup") or 0.0
reward = min(max(speedup, 0.0), 1.0) if result.get("correct") else 0.0
print(f"{reward:.4f}")
PY
