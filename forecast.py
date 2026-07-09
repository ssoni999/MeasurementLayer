#!/usr/bin/env python3
"""Forecast cost at higher traffic as a range with printed assumptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def forecast_range(
    avg_tps: float,
    knee_tps: float,
    peak_factor: float,
    gpu_cost: float,
    num_gpus: int,
    growth_levels: list[float],
) -> dict:
    peak_tps = avg_tps * peak_factor
    copies_at_peak = max(1.0, peak_tps / max(knee_tps, 1e-9))
    copies_low = max(1.0, copies_at_peak * 0.85)
    copies_high = copies_at_peak * 1.15

    forecasts = []
    for growth in growth_levels:
        annual_low = gpu_cost * num_gpus * copies_low * growth * 24 * 365
        annual_high = gpu_cost * num_gpus * copies_high * growth * 24 * 365
        forecasts.append(
            {
                "growth": f"{growth:.0f}x",
                "annual_low_usd": round(annual_low, 0),
                "annual_high_usd": round(annual_high, 0),
            }
        )

    return {
        "assumptions": {
            "avg_tps": avg_tps,
            "knee_tps": knee_tps,
            "peak_factor": peak_factor,
            "gpu_cost_per_hour": gpu_cost,
            "num_gpus_per_copy": num_gpus,
            "copies_at_peak_low": round(copies_low, 2),
            "copies_at_peak_high": round(copies_high, 2),
        },
        "forecasts": forecasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast cost at higher traffic.")
    parser.add_argument("--avg-tps", type=float, default=1200.0)
    parser.add_argument("--knee-tps", type=float, default=800.0)
    parser.add_argument("--peak", type=float, default=2.5, help="Peak/avg traffic factor")
    parser.add_argument("--gpu-cost", type=float, default=1.50)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--numbers", default="demo/numbers.json")
    args = parser.parse_args()

    result = forecast_range(
        args.avg_tps,
        args.knee_tps,
        args.peak,
        args.gpu_cost,
        args.num_gpus,
        [1.0, 3.0, 5.0],
    )

    print()
    print("==================== Forecast ========================")
    print("  Assumptions:")
    for key, value in result["assumptions"].items():
        print(f"    {key}: {value}")
    print()
    for item in result["forecasts"]:
        print(
            f"  {item['growth']:>3} traffic: "
            f"${item['annual_low_usd']:,.0f} - ${item['annual_high_usd']:,.0f} / year"
        )
    print("=======================================================")

    numbers_path = Path(args.numbers)
    numbers_path.parent.mkdir(parents=True, exist_ok=True)
    numbers = {}
    if numbers_path.exists():
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    numbers["forecast"] = result
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
