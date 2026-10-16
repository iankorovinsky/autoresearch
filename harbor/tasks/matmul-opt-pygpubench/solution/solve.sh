#!/usr/bin/env bash
set -euo pipefail
cp /opt/autoresearch/KernelBench/src/kernelbench/prompts/few_shot/model_new_ex_tiled_matmul.py /workspace/candidate.py
echo "oracle: installed reference tiled matmul candidate at /workspace/candidate.py"
