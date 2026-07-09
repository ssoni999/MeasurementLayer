#!/usr/bin/env python3
"""Compare baseline vs tuned model quality on the same prompts."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import openai


def load_prompts(trace_path: Path, sample: int, seed: int) -> list[tuple[str, int]]:
    records = []
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            records.append((data["prompt"], int(data["max_tokens"])))
    rng = random.Random(seed)
    if sample < len(records):
        records = rng.sample(records, sample)
    return records


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def score_pair(baseline: str, tuned: str) -> float:
    if not baseline and not tuned:
        return 100.0
    if not baseline or not tuned:
        return 0.0
    b = set(normalize(baseline).split())
    t = set(normalize(tuned).split())
    if not b:
        return 100.0
    overlap = len(b & t) / len(b)
    return overlap * 100


def complete(client: openai.OpenAI, model: str, prompt: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality eval: baseline vs tuned.")
    parser.add_argument("--target", required=True, help="Baseline endpoint")
    parser.add_argument("--tuned", required=True, help="Tuned endpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", default="traces/sample.jsonl")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--numbers", default="demo/numbers.json")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        print(f"Trace not found: {trace_path}", file=sys.stderr)
        return 1

    prompts = load_prompts(trace_path, args.sample, args.seed)
    baseline_url = args.target.rstrip("/")
    tuned_url = args.tuned.rstrip("/")
    if not baseline_url.endswith("/v1"):
        baseline_url += "/v1"
    if not tuned_url.endswith("/v1"):
        tuned_url += "/v1"

    baseline_client = openai.OpenAI(api_key="EMPTY", base_url=baseline_url)
    tuned_client = openai.OpenAI(api_key="EMPTY", base_url=tuned_url)

    scores = []
    for i, (prompt, max_tokens) in enumerate(prompts):
        eval_max = min(max_tokens, 128)
        baseline_out = complete(baseline_client, args.model, prompt, eval_max)
        tuned_out = complete(tuned_client, args.model, prompt, eval_max)
        score = score_pair(baseline_out, tuned_out)
        scores.append(score)
        print(f"  [{i + 1}/{len(prompts)}] score={score:.1f}%")

    avg_score = sum(scores) / max(len(scores), 1)
    held = avg_score >= 95.0

    print()
    print("==================== Quality ==========================")
    print(f"  Eval score:        {avg_score:.1f}%")
    print(f"  Quality held:      {'yes' if held else 'NO - do not claim saving'}")
    print("=======================================================")

    numbers_path = Path(args.numbers)
    numbers_path.parent.mkdir(parents=True, exist_ok=True)
    numbers = {}
    if numbers_path.exists():
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    numbers["quality"] = {
        "eval_score_pct": round(avg_score, 1),
        "held": held,
        "sample_size": len(prompts),
    }
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    return 0 if held else 2


if __name__ == "__main__":
    raise SystemExit(main())
