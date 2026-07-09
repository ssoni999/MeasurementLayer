# Token Economics Demo Script

Five-minute walkthrough. Every number comes from a `make` command — never typed by hand.

## Before you start

1. Warm the model: send 5–10 requests or run `make replay` once and discard.
2. Confirm endpoint: `curl http://localhost:30080/v1/models`
3. Open Grafana: `kubectl port-forward svc/vllm-grafana 3000:80` (admin/admin)
4. Import `dashboards/token-economics.json`, set `gpu_cost` variable.

## Screen 1 · Bill today (~60s)

```bash
make sample-trace
make replay TARGET=http://localhost:30080 MODEL=<model>
make cost GPU_COST=1.50 RPD=500000 TPR=180
```

Say: *"This workload costs $X per year at today's throughput."*

Show: annual bill, cost/1M, throughput, p95/p99 from replay output.

## Screen 2 · Diagnose (~60s)

Terminal 1:
```bash
make replay TARGET=http://localhost:30080 MODEL=<model>
```

Terminal 2 (mid-replay):
```bash
make profile TARGET=http://localhost:30080
```

Say: *"Decode-bound, shared prefix, queue rising — here's the fix."*

## Screen 3 · Fix + quality held (~60s)

Platform applies FP8 + tuned flags (Step 9), then:

```bash
make replay TARGET=<tuned> MODEL=RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8
make quality TARGET=<baseline> TUNED=<tuned> MODEL=<model>
```

Say: *"Throughput up, eval score held, p99 held."*

## Screen 4 · The saving (~30s)

```bash
make cost TPS=<baseline> VS=<tuned> GPU_COST=1.50 RPD=500000 TPR=180
```

Say: *"$X saved per year, quality held."*

## Screen 5 · Forecast (~60s)

```bash
bash bench/sweep.sh http://localhost:30080 <model>
make knee CSV=traces/sweep_points.csv SLO_MS=500
make forecast AVG_TPS=<avg> KNEE_TPS=<knee> PEAK=2.5 GPU_COST=1.50
```

Say: *"At 3x traffic, cost lands in this range — assumptions on screen."*

## Close

Pre-run baseline and forecast before the live demo. Keep `demo/numbers.json` as the shared source of truth.
