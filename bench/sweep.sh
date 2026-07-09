#!/usr/bin/env bash
# Sweep load at rising rates and record p95 + throughput into sweep_points.csv
set -euo pipefail

TARGET="${1:-http://localhost:30080}"
MODEL="${2:?MODEL is required as second argument}"
TRACE="${TRACE:-traces/sample.jsonl}"
OUTPUT="${OUTPUT:-traces/sweep_points.csv}"

RATES=(0.5 1 2 4 8 12 16 24 32)

echo "rate,p95_ms,tokens_per_sec" > "$OUTPUT"

for rate in "${RATES[@]}"; do
  echo "Sweeping concurrency=${rate}..."
  result_file=$(mktemp)
  python3 replay.py \
    --target "$TARGET" \
    --model "$MODEL" \
    --trace "$TRACE" \
    --output "$result_file" \
    --speed 20 \
    --concurrency "$rate" \
    --numbers /dev/null 2>/dev/null || true

  if [[ -f "$result_file" ]]; then
    p95=$(python3 -c "import json; d=json.load(open('$result_file')); print(d['latency_ms']['p95'])")
    tps=$(python3 -c "import json; d=json.load(open('$result_file')); print(d['throughput_tok_s'])")
    echo "${rate},${p95},${tps}" >> "$OUTPUT"
    echo "  rate=${rate} p95=${p95}ms tps=${tps}"
  fi
  rm -f "$result_file"
done

echo "Wrote sweep points to $OUTPUT"
echo "Run: make knee CSV=$OUTPUT SLO_MS=500"
