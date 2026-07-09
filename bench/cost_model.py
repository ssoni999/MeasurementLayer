#!/usr/bin/env python3
"""Token economics cost model for baseline and before/after comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def annual_gpu_cost(gpu_cost: float, num_gpus: int) -> float:
    return gpu_cost * num_gpus * 24 * 365


def required_gpus(needed_tps: float, per_gpu_tps: float, num_gpus: int) -> float:
    if per_gpu_tps <= 0:
        return float(num_gpus)
    return max(num_gpus, needed_tps / per_gpu_tps)


def compute_bill(
    tps: float,
    gpu_cost: float,
    rpd: float,
    tpr: float,
    num_gpus: int,
) -> dict:
    tokens_per_year = rpd * 365 * tpr
    needed_tps = tokens_per_year / (365 * 24 * 3600)
    gpus_required = required_gpus(needed_tps, tps / max(num_gpus, 1), num_gpus)
    annual = annual_gpu_cost(gpu_cost, int(max(1, round(gpus_required))))
    util_factor = needed_tps / max(tps, 1e-9)
    annual_bill = annual * util_factor
    cost_1m = (annual_bill / tokens_per_year) * 1_000_000 if tokens_per_year else 0.0
    perf_per_dollar = tps / max(gpu_cost * max(num_gpus, 1), 1e-9)
    return {
        "throughput_tok_s": tps,
        "tokens_per_year": int(tokens_per_year),
        "needed_tps": round(needed_tps, 1),
        "gpus_required": round(gpus_required, 2),
        "annual_bill_usd": round(annual_bill, 0),
        "cost_per_million_usd": round(cost_1m, 4),
        "perf_per_dollar_tok_s": round(perf_per_dollar, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute inference cost from throughput.")
    parser.add_argument("--tps", type=float, help="Measured throughput (tok/s)")
    parser.add_argument("--vs", type=float, help="Tuned throughput for comparison")
    parser.add_argument("--gpu-cost", type=float, default=1.50, help="$/hour per GPU")
    parser.add_argument("--rpd", type=float, default=500_000, help="Requests per day")
    parser.add_argument("--tpr", type=float, default=180, help="Avg output tokens per request")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--replay", default="traces/replay_results.json")
    parser.add_argument("--numbers", default="demo/numbers.json")
    args = parser.parse_args()

    tps = args.tps
    if tps is None and Path(args.replay).exists():
        replay = json.loads(Path(args.replay).read_text(encoding="utf-8"))
        tps = replay.get("throughput_tok_s")
    if tps is None:
        raise SystemExit("TPS is required (or provide --replay with throughput_tok_s)")

    baseline = compute_bill(tps, args.gpu_cost, args.rpd, args.tpr, args.num_gpus)

    print()
    print("==================== Cost model ======================")
    print(f"  Throughput:        {baseline['throughput_tok_s']:,.1f} tok/s")
    print(f"  Annual bill:       ${baseline['annual_bill_usd']:,.0f}")
    print(f"  Cost / 1M tokens:  ${baseline['cost_per_million_usd']:.4f}")
    print(f"  Perf / dollar:     {baseline['perf_per_dollar_tok_s']:,.1f} tok/s/$")
    print(f"  GPUs required:     {baseline['gpus_required']:.1f}")
    print("======================================================")

    numbers_path = Path(args.numbers)
    numbers_path.parent.mkdir(parents=True, exist_ok=True)
    numbers = {}
    if numbers_path.exists():
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    numbers["cost_baseline"] = baseline

    if args.vs is not None:
        tuned = compute_bill(args.vs, args.gpu_cost, args.rpd, args.tpr, args.num_gpus)
        saving = baseline["annual_bill_usd"] - tuned["annual_bill_usd"]
        pct = (saving / baseline["annual_bill_usd"] * 100) if baseline["annual_bill_usd"] else 0
        print()
        print("==================== Saving ==========================")
        print(f"  Baseline annual:   ${baseline['annual_bill_usd']:,.0f}")
        print(f"  Tuned annual:      ${tuned['annual_bill_usd']:,.0f}")
        print(f"  Saving / year:     ${saving:,.0f} ({pct:.0f}% lower)")
        print("======================================================")
        numbers["cost_tuned"] = tuned
        numbers["saving"] = {
            "annual_usd": round(saving, 0),
            "percent_lower": round(pct, 1),
        }

    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    print(f"\nUpdated {numbers_path}")


if __name__ == "__main__":
    main()
