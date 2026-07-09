#!/usr/bin/env python3
"""Find per-copy capacity knee from sweep CSV at a latency SLO."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_sweep(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(
                {
                    "rate": float(row["rate"]),
                    "p95_ms": float(row["p95_ms"]),
                    "tokens_per_sec": float(row["tokens_per_sec"]),
                }
            )
        return rows


def find_knee(rows: list[dict], slo_ms: float) -> dict:
    eligible = [r for r in rows if r["p95_ms"] <= slo_ms]
    if not eligible:
        best = min(rows, key=lambda r: r["p95_ms"])
        return {
            "knee_tps": best["tokens_per_sec"],
            "rate": best["rate"],
            "p95_ms": best["p95_ms"],
            "slo_met": False,
        }
    knee = max(eligible, key=lambda r: r["tokens_per_sec"])
    return {
        "knee_tps": knee["tokens_per_sec"],
        "rate": knee["rate"],
        "p95_ms": knee["p95_ms"],
        "slo_met": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Find knee throughput at SLO.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--slo-ms", type=float, default=500.0)
    parser.add_argument("--numbers", default="demo/numbers.json")
    args = parser.parse_args()

    rows = load_sweep(Path(args.csv))
    result = find_knee(rows, args.slo_ms)
    result["slo_ms"] = args.slo_ms

    print()
    print("==================== Knee =============================")
    print(f"  SLO p95:           {args.slo_ms:.0f} ms")
    print(f"  Knee throughput:   {result['knee_tps']:,.1f} tok/s")
    print(f"  At rate:           {result['rate']}")
    print(f"  Observed p95:      {result['p95_ms']:.0f} ms")
    print(f"  SLO met:           {'yes' if result['slo_met'] else 'no (best effort)'}")
    print("=======================================================")

    numbers_path = Path(args.numbers)
    numbers_path.parent.mkdir(parents=True, exist_ok=True)
    numbers = {}
    if numbers_path.exists():
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    numbers["knee"] = result
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
