#!/usr/bin/env python3
"""Generate a realistic stand-in trace for the Token Economics demo."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SHARED_SYSTEM_PROMPT = """You are a helpful enterprise assistant for Acme Corp.
Follow company policy: be concise, cite sources when available, and never share
confidential data. Use markdown for structured answers. Default to bullet points
for lists. When unsure, ask a clarifying question before answering."""

USER_TOPICS = [
    "quarterly revenue breakdown",
    "customer churn analysis",
    "product roadmap priorities",
    "support ticket escalation policy",
    "security compliance checklist",
    "marketing campaign performance",
    "inventory forecast for Q4",
    "hiring plan for engineering",
    "API rate limit configuration",
    "data retention policy summary",
]


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _pad_to_tokens(text: str, target_tokens: int) -> str:
    words = text.split()
    if len(words) >= target_tokens:
        return " ".join(words[:target_tokens])
    filler = "Additional context for the request. " * ((target_tokens - len(words)) // 4 + 1)
    return f"{text} {filler}"


def _user_context(rng: random.Random, min_tokens: int, max_tokens: int) -> str:
    topic = rng.choice(USER_TOPICS)
    base = (
        f"User question about {topic}. "
        f"Please analyze the following notes and provide recommendations. "
    )
    target = rng.randint(min_tokens, max_tokens)
    detail = " ".join(
        f"Detail point {i}: metric value {rng.randint(10, 999)} with trend {rng.choice(['up', 'down', 'flat'])}."
        for i in range(1, 40)
    )
    return _pad_to_tokens(f"{base}{detail}", target)


def _max_tokens(rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.15:
        return rng.randint(64, 99)
    if roll < 0.55:
        return rng.randint(100, 199)
    if roll < 0.85:
        return rng.randint(200, 319)
    return rng.randint(320, 512)


def _generate_offsets(rng: random.Random, num_requests: int, span_seconds: float) -> list[float]:
    """Mix steady traffic with short bursts (peak ~2-3x average)."""
    offsets: list[float] = []
    t = 0.0
    burst_remaining = 0
    while len(offsets) < num_requests:
        if burst_remaining > 0:
            gap = rng.uniform(0.05, 0.25)
            burst_remaining -= 1
        elif rng.random() < 0.08:
            burst_remaining = rng.randint(8, 20)
            gap = rng.uniform(0.05, 0.2)
            burst_remaining -= 1
        else:
            avg_gap = span_seconds / num_requests
            gap = rng.expovariate(1.0 / avg_gap)
        t += gap
        offsets.append(round(t, 3))
    return offsets


def generate_trace(
    num_requests: int,
    span_seconds: float,
    seed: int,
    shared_prefix_tokens: int,
    user_min_tokens: int,
    user_max_tokens: int,
) -> list[dict]:
    rng = random.Random(seed)
    system_prompt = _pad_to_tokens(SHARED_SYSTEM_PROMPT, shared_prefix_tokens)
    offsets = _generate_offsets(rng, num_requests, span_seconds)
    records = []
    for offset in offsets:
        user_context = _user_context(rng, user_min_tokens, user_max_tokens)
        prompt = f"{system_prompt}\n\n---\n\n{user_context}"
        records.append(
            {
                "prompt": prompt,
                "max_tokens": _max_tokens(rng),
                "offset": offset,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a sample JSONL inference trace.")
    parser.add_argument("-o", "--output", default="traces/sample.jsonl")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--span-seconds", type=float, default=2400.0, help="Simulated trace span")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-prefix-tokens", type=int, default=1000)
    parser.add_argument("--user-min-tokens", type=int, default=500)
    parser.add_argument("--user-max-tokens", type=int, default=4000)
    args = parser.parse_args()

    records = generate_trace(
        num_requests=args.requests,
        span_seconds=args.span_seconds,
        seed=args.seed,
        shared_prefix_tokens=args.shared_prefix_tokens,
        user_min_tokens=args.user_min_tokens,
        user_max_tokens=args.user_max_tokens,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    offsets = [r["offset"] for r in records]
    gaps = [b - a for a, b in zip(offsets, offsets[1:])] if len(offsets) > 1 else [1.0]
    peak_gap = min(gaps)
    avg_gap = records[-1]["offset"] / max(len(records), 1)
    peak_factor = avg_gap / max(peak_gap, 0.001)

    print(f"Wrote {len(records)} requests to {output}")
    print(f"Simulated span: {records[-1]['offset']:.1f}s")
    print(f"Approx shared prefix tokens: {_approx_tokens(records[0]['prompt'].split('---')[0])}")
    print(f"Peak/avg gap factor (for forecast): ~{peak_factor:.1f}x")


if __name__ == "__main__":
    main()
