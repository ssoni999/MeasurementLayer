.PHONY: sample-trace replay cost profile quality knee forecast scale local-dash

TRACE ?= traces/sample.jsonl
OUTPUT ?= traces/replay_results.json

sample-trace:
	python3 bench/sample_trace.py -o $(TRACE)

replay:
	@test -n "$(TARGET)" || (echo "TARGET is required, e.g. TARGET=http://localhost:30080" && exit 1)
	@test -n "$(MODEL)" || (echo "MODEL is required, e.g. MODEL=meta-llama/Llama-3.1-8B-Instruct" && exit 1)
	python3 replay.py --target $(TARGET) --model $(MODEL) --trace $(TRACE) --output $(OUTPUT) \
		$(if $(SPEED),--speed $(SPEED),) \
		$(if $(CONCURRENCY),--concurrency $(CONCURRENCY),) \
		$(if $(API),--api $(API),)

cost:
	python3 bench/cost_model.py \
		$(if $(TPS),--tps $(TPS),) \
		$(if $(VS),--vs $(VS),) \
		$(if $(GPU_COST),--gpu-cost $(GPU_COST),) \
		$(if $(RPD),--rpd $(RPD),) \
		$(if $(TPR),--tpr $(TPR),) \
		$(if $(NUM_GPUS),--num-gpus $(NUM_GPUS),) \
		$(if $(REPLAY),--replay $(REPLAY),)

profile:
	@test -n "$(TARGET)" || (echo "TARGET is required" && exit 1)
	python3 discovery/profile.py --target $(TARGET) \
		$(if $(DURATION),--duration $(DURATION),) \
		$(if $(INTERVAL),--interval $(INTERVAL),)

quality:
	@test -n "$(TARGET)" || (echo "TARGET (baseline) is required" && exit 1)
	@test -n "$(TUNED)" || (echo "TUNED endpoint is required" && exit 1)
	@test -n "$(MODEL)" || (echo "MODEL is required" && exit 1)
	python3 quality/eval.py --target $(TARGET) --tuned $(TUNED) --model $(MODEL) \
		--trace $(TRACE) \
		$(if $(SAMPLE),--sample $(SAMPLE),)

knee:
	@test -n "$(CSV)" || (echo "CSV is required, e.g. CSV=traces/sweep_points.csv" && exit 1)
	python3 knee.py --csv $(CSV) \
		$(if $(SLO_MS),--slo-ms $(SLO_MS),)

forecast:
	python3 forecast.py \
		$(if $(AVG_TPS),--avg-tps $(AVG_TPS),) \
		$(if $(KNEE_TPS),--knee-tps $(KNEE_TPS),) \
		$(if $(PEAK),--peak $(PEAK),) \
		$(if $(GPU_COST),--gpu-cost $(GPU_COST),) \
		$(if $(NUM_GPUS),--num-gpus $(NUM_GPUS),)

scale:
	@test -n "$(TARGET)" || (echo "TARGET is required" && exit 1)
	@test -n "$(MODEL)" || (echo "MODEL is required" && exit 1)
	python3 replay.py --target $(TARGET) --model $(MODEL) --trace $(TRACE) --speed 1 --concurrency 64

local-dash:
	@echo "Start Grafana locally or port-forward: kubectl port-forward svc/vllm-grafana 3000:80"
