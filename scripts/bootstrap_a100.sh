#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e ".[dev,gpu,dense]"
python -m amazon_er.cli doctor --config configs/a100.yaml
python -m amazon_er.cli smoke --config configs/smoke.yaml

echo "Environment and offline smoke test passed."
echo "Run: bash scripts/run_cpu_pipeline.sh configs/a100.yaml /workspace/dataset /workspace/artifacts /workspace/output"
