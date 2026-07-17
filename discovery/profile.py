#!/usr/bin/env python3
"""Profile live inference metrics and recommend a fix."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests


METRIC_PATTERNS = {
    "prefill": re.compile(r'^vllm:num_prefill_requests\{[^}]*\}\s+([0-9.eE+-]+)$', re.M),
    "decode": re.compile(r'^vllm:num_decoding_requests\{[^}]*\}\s+([0-9.eE+-]+)$', re.M),
    "waiting": re.compile(r'^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.eE+-]+)$', re.M),
    "prefix_hit_rate": re.compile(
        r'^vllm:gpu_prefix_cache_hit_rate\{[^}]*\}\s+([0-9.eE+-]+)$', re.M
    ),
    "input_tokens": re.compile(r'^vllm:input_tokens_total\{[^}]*\}\s+([0-9.eE+-]+)$', re.M),
    "output_tokens": re.compile(r'^vllm:output_tokens_total\{[^}]*\}\s+([0-9.eE+-]+)$', re.M),
    "qps": re.compile(r'^vllm:current_qps\{[^}]*\}\s+([0-9.eE+-]+)$', re.M),
}


def scrape_metrics(target: str) -> str:
    url = f"{target.rstrip('/')}/metrics"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def sum_matches(pattern: re.Pattern[str], text: str) -> float:
    return sum(float(m) for m in pattern.findall(text))


def _split_from_tokens(samples: list[dict]) -> tuple[float, float] | None:
    if len(samples) < 2:
        return None
    input_delta = samples[-1]["input_tokens"] - samples[0]["input_tokens"]
    output_delta = samples[-1]["output_tokens"] - samples[0]["output_tokens"]
    total = input_delta + output_delta
    if total <= 0:
        return None
    prefill_pct = input_delta / total * 100
    return prefill_pct, 100.0 - prefill_pct


def _split_from_replay(replay: dict) -> tuple[float, float] | None:
    ttft = replay.get("ttft_ms", {}).get("p50", 0)
    latency = replay.get("latency_ms", {}).get("p50", 0)
    if latency <= 0 or ttft <= 0 or ttft >= latency:
        return None
    prefill_pct = ttft / latency * 100
    return prefill_pct, 100.0 - prefill_pct


def _split_from_gauges(samples: list[dict], interval: float) -> tuple[float, float] | None:
    prefill_time = 0.0
    decode_time = 0.0
    for sample in samples:
        prefill_time += sample["prefill"] * interval
        decode_time += sample["decode"] * interval
    total = prefill_time + decode_time
    if total <= 0:
        return None
    prefill_pct = prefill_time / total * 100
    return prefill_pct, 100.0 - prefill_pct


def _gauge_metrics_unreliable(samples: list[dict]) -> bool:
    if not samples:
        return True
    decode_seen = any(sample["decode"] > 0 for sample in samples)
    prefill_seen = any(sample["prefill"] > 0 for sample in samples)
    return prefill_seen and not decode_seen


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _split_from_trace(trace_path: Path, *, max_output_tokens: int = 32) -> tuple[float, float] | None:
    if not trace_path.exists():
        return None
    input_tokens: list[float] = []
    output_tokens: list[float] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        prompt = record.get("prompt", "")
        output_tokens.append(min(int(record.get("max_tokens", max_output_tokens)), max_output_tokens))
        input_tokens.append(_approx_tokens(prompt))
    if not input_tokens:
        return None
    avg_input = sum(input_tokens) / len(input_tokens)
    avg_output = sum(output_tokens) / len(output_tokens)
    total = avg_input + avg_output
    if total <= 0:
        return None
    prefill_pct = avg_input / total * 100
    return prefill_pct, 100.0 - prefill_pct


def _replay_status(replay_path: Path | None) -> dict | None:
    if replay_path is None or not replay_path.exists():
        return None
    data = json.loads(replay_path.read_text(encoding="utf-8"))
    return {
        "requests_ok": int(data.get("requests_ok", 0)),
        "requests_error": int(data.get("requests_error", 0)),
        "has_timing": data.get("latency_ms", {}).get("p50", 0) > 0,
    }


def _load_replay_data(numbers_path: Path, replay_path: Path | None) -> dict | None:
    if replay_path and replay_path.exists():
        data = json.loads(replay_path.read_text(encoding="utf-8"))
        if data.get("latency_ms", {}).get("p50", 0) > 0:
            return data
    if numbers_path.exists():
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
        replay = numbers.get("replay")
        if isinstance(replay, dict) and replay.get("latency_ms", {}).get("p50", 0) > 0:
            return replay
    return None


def diagnose(
    samples: list[dict],
    *,
    interval: float,
    replay_data: dict | None = None,
    trace_split: tuple[float, float] | None = None,
    replay_status: dict | None = None,
) -> dict:
    waiting = max((s["waiting"] for s in samples), default=0.0)
    prefix_hit = sum(s["prefix_hit_rate"] for s in samples) / max(len(samples), 1)
    avg_qps = sum(s.get("qps", 0.0) for s in samples) / max(len(samples), 1)
    idle_samples = sum(1 for s in samples if s["prefill"] + s["decode"] == 0)

    token_split = _split_from_tokens(samples)
    replay_split = _split_from_replay(replay_data) if replay_data else None
    gauge_split = _split_from_gauges(samples, interval)
    gauge_unreliable = _gauge_metrics_unreliable(samples)

    warnings: list[str] = []
    method = "default"
    if token_split:
        prefill_pct, decode_pct = token_split
        method = "token_counters"
    elif replay_split:
        prefill_pct, decode_pct = replay_split
        method = "replay_timing"
        if gauge_unreliable:
            warnings.append(
                "Router prefill/decode gauges showed no decode activity; "
                "using replay TTFT/latency timing instead."
            )
        else:
            warnings.append("Using replay TTFT/latency timing for prefill/decode split.")
    elif gauge_split and not gauge_unreliable:
        prefill_pct, decode_pct = gauge_split
        method = "time_weighted_gauges"
    elif trace_split:
        prefill_pct, decode_pct = trace_split
        method = "trace_workload"
        warnings.append(
            "Router prefill/decode gauges are unreliable; "
            "using trace prompt/output token mix instead."
        )
    elif gauge_split:
        prefill_pct, decode_pct = gauge_split
        method = "time_weighted_gauges"
        warnings.append(
            "Decode gauge never moved during sampling; result may overstate prefill. "
            "Run `make replay` in another terminal while profiling."
        )
    else:
        prefill_pct, decode_pct = 50.0, 50.0
        method = "default"
        warnings.append(
            "No active load detected. Run `make replay` while profiling "
            "(Step 8 expects concurrent load)."
        )

    if replay_status and replay_status["requests_ok"] == 0 and replay_status["requests_error"] > 0:
        warnings.append(
            f"Latest replay had 0 successful requests ({replay_status['requests_error']} errors). "
            "Fix connectivity/port-forward before trusting live metrics."
        )

    if idle_samples == len(samples) and avg_qps <= 0:
        warnings.append("Endpoint looked idle during the entire sample window.")

    shared_prefix_pct = prefix_hit * 100 if prefix_hit <= 1 else prefix_hit
    queueing = "high" if waiting >= 2 else "low"

    fixes = []
    if decode_pct >= 55:
        fixes.append("FP8 weights (RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8)")
        fixes.append("--max-num-seqs 320 --gpu-memory-utilization 0.9")
    if shared_prefix_pct >= 30:
        fixes.append("--enable-prefix-caching")
    if prefill_pct >= 45:
        fixes.append("--enable-chunked-prefill")
    if queueing == "high":
        fixes.append("Increase max-num-seqs or add replicas via KEDA")
    if not fixes:
        fixes.append("Review batch size and memory utilization")

    diagnosis = "decode bound" if decode_pct >= 55 else "prefill bound"
    return {
        "prefill_pct": round(prefill_pct, 1),
        "decode_pct": round(decode_pct, 1),
        "shared_prefix_pct": round(shared_prefix_pct, 1),
        "queueing": queueing,
        "max_waiting": round(waiting, 1),
        "diagnosis": diagnosis,
        "recommended_fix": fixes,
        "method": method,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile inference and recommend fixes.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--duration", type=int, default=30, help="Seconds to sample")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--numbers", default="demo/numbers.json")
    parser.add_argument(
        "--replay",
        default="traces/replay_results.json",
        help="Replay results JSON for TTFT/latency fallback",
    )
    parser.add_argument(
        "--trace",
        default="traces/sample.jsonl",
        help="Trace JSONL for workload-based prefill/decode fallback",
    )
    args = parser.parse_args()

    numbers_path = Path(args.numbers)
    replay_path = Path(args.replay) if args.replay else None
    replay_data = _load_replay_data(numbers_path, replay_path)
    replay_status = _replay_status(replay_path)
    trace_split = _split_from_trace(Path(args.trace)) if args.trace else None

    samples = []
    end = time.time() + args.duration
    print(f"Sampling {args.target}/metrics for {args.duration}s...")
    while time.time() < end:
        text = scrape_metrics(args.target)
        prefill_matches = METRIC_PATTERNS["prefill"].findall(text)
        decode_matches = METRIC_PATTERNS["decode"].findall(text)
        sample = {
            "prefill": sum(float(m) for m in prefill_matches),
            "decode": sum(float(m) for m in decode_matches),
            "waiting": sum_matches(METRIC_PATTERNS["waiting"], text),
            "prefix_hit_rate": sum_matches(METRIC_PATTERNS["prefix_hit_rate"], text),
            "input_tokens": sum_matches(METRIC_PATTERNS["input_tokens"], text),
            "output_tokens": sum_matches(METRIC_PATTERNS["output_tokens"], text),
            "qps": sum_matches(METRIC_PATTERNS["qps"], text),
        }
        samples.append(sample)
        time.sleep(args.interval)

    result = diagnose(
        samples,
        interval=args.interval,
        replay_data=replay_data,
        trace_split=trace_split,
        replay_status=replay_status,
    )

    print()
    print("==================== Profile ==========================")
    print(f"  Prefill:           {result['prefill_pct']:.0f}%")
    print(f"  Decode:            {result['decode_pct']:.0f}%")
    print(f"  Shared prefix:     {result['shared_prefix_pct']:.0f}%")
    print(f"  Queueing:          {result['queueing']} (max waiting {result['max_waiting']:.0f})")
    print(f"  Diagnosis:         {result['diagnosis']}")
    if result.get("method"):
        print(f"  Split source:      {result['method']}")
    for warning in result.get("warnings", []):
        print(f"  Note:              {warning}")
    print("  Recommended fix:")
    for fix in result["recommended_fix"]:
        print(f"    - {fix}")
    print("=======================================================")

    numbers_path.parent.mkdir(parents=True, exist_ok=True)
    numbers = {}
    if numbers_path.exists():
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    numbers["profile"] = result
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    print(f"\nUpdated {numbers_path}")


if __name__ == "__main__":
    main()
