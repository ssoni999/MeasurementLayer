# MeasurementLayer

Token Economics measurement tooling for the vLLM production-stack demo (Steps 6–13).

## Setup

```bash
pip install -r measurement-requirements.txt
```

Point at a running vLLM router (port-forward if on Kubernetes):

```bash
kubectl port-forward svc/vllm-router-service 30080:80 &
curl http://localhost:30080/v1/models
```

## Quick start

```bash
make sample-trace
make replay TARGET=http://localhost:30080 MODEL=<model-name>
make cost GPU_COST=1.50 RPD=500000 TPR=180
```

## Commands

| Command | Purpose |
|---------|---------|
| `make sample-trace` | Generate stand-in workload (`traces/sample.jsonl`) |
| `make replay` | Replay trace, report throughput + p50/p95/p99 |
| `make cost` | Price baseline bill (Screen 1); use `VS=` for savings (Screen 4) |
| `make profile` | Diagnose prefill/decode/cache/queueing (Screen 2) |
| `make quality` | Compare baseline vs tuned quality (Screen 3) |
| `make knee` | Per-copy capacity at SLO from sweep CSV |
| `make forecast` | Cost range at 1x/3x/5x traffic (Screen 5) |

See `demo/script.md` and `demo/checklist.md` for the full demo runbook.

## Layout

```
Makefile
replay.py              # Step 6 trace replay
bench/sample_trace.py  # Step 6 stand-in trace generator
bench/cost_model.py    # Steps 7, 11 cost model
discovery/profile.py   # Step 8 profiler
quality/eval.py        # Step 10 quality check
bench/sweep.sh         # Step 12 load sweep
knee.py                # Step 12 knee finder
forecast.py            # Step 13 forecast
dashboards/            # Grafana token-economics dashboard
demo/                  # Script, checklist, shared numbers.json
```
