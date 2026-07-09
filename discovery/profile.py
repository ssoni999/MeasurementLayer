#!/usr/bin/env python3
"""Profile live inference metrics and recommend a fix."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

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
}


def scrape_metrics(target: str) -> str:
    url = urljoin(target.rstrip("/") + "/", "metrics")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def sum_matches(pattern: re.Pattern[str], text: str) -> float:
    return sum(float(m) for m in pattern.findall(text))


def diagnose(samples: list[dict]) -> dict:
    prefill = sum(s["prefill"] for s in samples) / max(len(samples), 1)
    decode = sum(s["decode"] for s in samples) / max(len(samples), 1)
    waiting = max(s["waiting"] for s in samples)
    prefix_hit = sum(s["prefix_hit_rate"] for s in samples) / max(len(samples), 1)
    total = prefill + decode
    prefill_pct = (prefill / total * 100) if total else 50.0
    decode_pct = 100.0 - prefill_pct

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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile inference and recommend fixes.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--duration", type=int, default=30, help="Seconds to sample")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--numbers", default="demo/numbers.json")
    args = parser.parse_args()

    samples = []
    end = time.time() + args.duration
    print(f"Sampling {args.target}/metrics for {args.duration}s...")
    while time.time() < end:
        text = scrape_metrics(args.target)
        sample = {
            "prefill": sum_matches(METRIC_PATTERNS["prefill"], text),
            "decode": sum_matches(METRIC_PATTERNS["decode"], text),
            "waiting": sum_matches(METRIC_PATTERNS["waiting"], text),
            "prefix_hit_rate": sum_matches(METRIC_PATTERNS["prefix_hit_rate"], text),
            "input_tokens": sum_matches(METRIC_PATTERNS["input_tokens"], text),
            "output_tokens": sum_matches(METRIC_PATTERNS["output_tokens"], text),
        }
        samples.append(sample)
        time.sleep(args.interval)

    result = diagnose(samples)

    print()
    print("==================== Profile ==========================")
    print(f"  Prefill:           {result['prefill_pct']:.0f}%")
    print(f"  Decode:            {result['decode_pct']:.0f}%")
    print(f"  Shared prefix:     {result['shared_prefix_pct']:.0f}%")
    print(f"  Queueing:          {result['queueing']} (max waiting {result['max_waiting']:.0f})")
    print(f"  Diagnosis:         {result['diagnosis']}")
    print("  Recommended fix:")
    for fix in result["recommended_fix"]:
        print(f"    - {fix}")
    print("=======================================================")

    numbers_path = Path(args.numbers)
    numbers_path.parent.mkdir(parents=True, exist_ok=True)
    numbers = {}
    if numbers_path.exists():
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    numbers["profile"] = result
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    print(f"\nUpdated {numbers_path}")


if __name__ == "__main__":
    main()
