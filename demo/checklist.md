# Token Economics Demo Checklist

## Platform (Steps 1–5)

- [ ] Cluster + GPU node pool ready (`nvidia-l4` label on nodes)
- [ ] KEDA operator Running
- [ ] vLLM stack serving model via router
- [ ] Shared storage (RWX) for 8B model weights
- [ ] Grafana imported with `dashboards/token-economics.json`
- [ ] `gpu_cost` variable set in Grafana

## Measurement (Steps 6–13)

- [ ] `pip install -r measurement-requirements.txt`
- [ ] `make sample-trace` produces `traces/sample.jsonl`
- [ ] `make replay` prints throughput + p50/p95/p99
- [ ] `make cost` prints annual bill tied to replay TPS
- [ ] `make profile` (during replay) prints diagnosis + fix
- [ ] Tuned config deployed (FP8 + prefix caching + batch tuning)
- [ ] `make quality` confirms eval score held
- [ ] `make cost TPS=... VS=...` prints saving
- [ ] `bench/sweep.sh` + `make knee` yields per-copy capacity
- [ ] `make forecast` prints ranged cost at 1x/3x/5x

## Presentation (Step 15)

- [ ] Screen 1–5 built with placeholders, then swapped from `demo/numbers.json`
- [ ] Baseline pre-run and saved (do not run live in demo)
- [ ] Forecast inputs pre-computed
- [ ] Full run rehearsed twice end-to-end in under 5 minutes

## Agreed parameters (`demo/numbers.json`)

- [ ] Model name confirmed from `/v1/models`
- [ ] GPU $/hour agreed
- [ ] SLO latency target (p95 ms) agreed
- [ ] Volume: requests/day and tokens/request agreed
