#!/usr/bin/env python3
"""Replay a JSONL trace against an OpenAI-compatible inference endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import openai


# #region agent log
def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    log_path = os.environ.get("DEBUG_LOG_PATH")
    if not log_path:
        return
    payload = {
        "sessionId": "fe7190",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass
# #endregion


@dataclass
class TraceRecord:
    prompt: str
    max_tokens: int
    offset: float


@dataclass
class RequestResult:
    index: int
    offset: float
    max_tokens: int
    status: str
    latency_ms: float
    ttft_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None


def load_trace(path: Path) -> list[TraceRecord]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(
                TraceRecord(
                    prompt=data["prompt"],
                    max_tokens=int(data["max_tokens"]),
                    offset=float(data.get("offset", 0.0)),
                )
            )
    return records


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def check_endpoint(base_url: str) -> None:
    """Fail fast if the router is unreachable."""
    models_url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=5) as resp:
            body = resp.read(500).decode("utf-8", errors="replace")
            # #region agent log
            _agent_log(
                "H1",
                "replay.py:check_endpoint",
                "preflight_ok",
                {"models_url": models_url, "status": resp.status, "body_preview": body[:200]},
            )
            # #endregion
    except Exception as exc:  # noqa: BLE001
        # #region agent log
        _agent_log(
            "H1",
            "replay.py:check_endpoint",
            "preflight_failed",
            {
                "models_url": models_url,
                "exc_type": type(exc).__name__,
                "exc_msg": str(exc),
            },
        )
        # #endregion
        print(f"ERROR: Cannot reach {models_url}: {exc}", file=sys.stderr)
        print(
            "\nNothing is listening on that address. Common fixes:\n"
            "\n"
            "  1. Connect kubectl to your GKE cluster (if not in Cloud Shell):\n"
            "     gcloud container clusters get-credentials te-poc --zone=us-central1-a\n"
            "     kubectl get pods\n"
            "\n"
            "  2. Start port-forward and keep it running in another terminal:\n"
            "     kubectl port-forward svc/vllm-router-service 30080:80\n"
            "\n"
            "  3. Verify the router responds:\n"
            "     curl http://localhost:30080/v1/models\n"
            "\n"
            "  Or run MeasurementLayer inside GCP Cloud Shell where kubectl\n"
            "  is already configured for the cluster you deployed in Steps 1-5.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def resolve_api_mode(model: str, api: str) -> str:
    """Pick chat vs completions API. opt-125m and similar base models need completions."""
    if api != "auto":
        return api
    model_lower = model.lower()
    if "opt-125" in model_lower or "/opt-" in model_lower:
        return "completions"
    return "chat"


async def send_request(
    client: openai.AsyncOpenAI,
    model: str,
    record: TraceRecord,
    index: int,
    api_mode: str,
) -> RequestResult:
    start = time.perf_counter()
    ttft_ms = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    try:
        if api_mode == "completions":
            stream = await client.completions.create(
                model=model,
                prompt=record.prompt,
                max_tokens=min(record.max_tokens, 32),
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
            )
        else:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": record.prompt}],
                max_tokens=min(record.max_tokens, 32),
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
            )
        first_token = None
        async for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
                completion_tokens = chunk.usage.completion_tokens or completion_tokens
            if not chunk.choices:
                continue
            if api_mode == "completions":
                content = chunk.choices[0].text
            else:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None) or getattr(
                    delta, "reasoning_content", None
                )
            if content and first_token is None:
                first_token = time.perf_counter()
        end = time.perf_counter()
        if first_token is None:
            first_token = end
        ttft_ms = (first_token - start) * 1000
        latency_ms = (end - start) * 1000
        return RequestResult(
            index=index,
            offset=record.offset,
            max_tokens=min(record.max_tokens, 32),
            status="ok",
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        end = time.perf_counter()
        return RequestResult(
            index=index,
            offset=record.offset,
            max_tokens=min(record.max_tokens, 32),
            status="error",
            latency_ms=(end - start) * 1000,
            ttft_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(exc),
        )


async def warmup(client: openai.AsyncOpenAI, model: str, count: int, api_mode: str) -> None:
    for i in range(count):
        await send_request(
            client,
            model,
            TraceRecord(prompt=f"WARMUP request {i}: say ok", max_tokens=8, offset=0.0),
            index=-1,
            api_mode=api_mode,
        )


async def replay(
    target: str,
    model: str,
    trace: list[TraceRecord],
    speed: float,
    concurrency: int,
    warmup_count: int,
    api_mode: str,
) -> tuple[list[RequestResult], float]:
    base_url = target.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    check_endpoint(base_url)
    client = openai.AsyncOpenAI(api_key="EMPTY", base_url=base_url)

    await warmup(client, model, warmup_count, api_mode)

    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    replay_start = time.perf_counter()

    async def run_one(index: int, record: TraceRecord) -> None:
        target_time = replay_start + (record.offset / speed)
        delay = target_time - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        async with semaphore:
            result = await send_request(client, model, record, index, api_mode)
            results.append(result)

    await asyncio.gather(*(run_one(i, record) for i, record in enumerate(trace)))
    wall_time = time.perf_counter() - replay_start
    results.sort(key=lambda r: r.index)
    return results, wall_time


def summarize(
    results: list[RequestResult],
    wall_time: float,
    trace_path: str,
    target: str,
    model: str,
) -> dict:
    ok = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status != "ok"]
    latencies = [r.latency_ms for r in ok]
    ttfts = [r.ttft_ms for r in ok]
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in ok)
    throughput = total_tokens / wall_time if wall_time > 0 else 0.0
    output_tokens = sum(r.completion_tokens for r in ok)
    gen_tps = output_tokens / wall_time if wall_time > 0 else 0.0

    summary = {
        "target": target,
        "model": model,
        "trace": trace_path,
        "wall_time_s": round(wall_time, 2),
        "requests_ok": len(ok),
        "requests_error": len(errors),
        "throughput_tok_s": round(throughput, 1),
        "generation_tok_s": round(gen_tps, 1),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "p99": round(percentile(latencies, 99), 1),
        },
        "ttft_ms": {
            "p50": round(percentile(ttfts, 50), 1),
            "p95": round(percentile(ttfts, 95), 1),
            "p99": round(percentile(ttfts, 99), 1),
        },
        "avg_output_tokens_per_request": round(
            output_tokens / max(len(ok), 1), 1
        ),
        "results": [asdict(r) for r in results],
    }
    return summary


def print_summary(summary: dict) -> None:
    print()
    print("==================== Replay summary ======================")
    print(f"  Throughput:     {summary['throughput_tok_s']:,.1f} tok/s")
    print(f"  Generation:     {summary['generation_tok_s']:,.1f} tok/s")
    print(f"  Latency p50:    {summary['latency_ms']['p50']:.0f} ms")
    print(f"  Latency p95:    {summary['latency_ms']['p95']:.0f} ms")
    print(f"  Latency p99:    {summary['latency_ms']['p99']:.0f} ms")
    print(f"  TTFT p95:       {summary['ttft_ms']['p95']:.0f} ms")
    print(
        f"  Requests:       {summary['requests_ok']} ok / "
        f"{summary['requests_error']} err"
    )
    print(f"  Trace:          {summary['trace']}")
    print(f"  Wall time:      {summary['wall_time_s']:.1f}s")
    print("==========================================================")
    print()


def write_numbers_json(summary: dict, numbers_path: Path) -> None:
    numbers_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if numbers_path.exists():
        existing = json.loads(numbers_path.read_text(encoding="utf-8"))
    existing["replay"] = {
        "throughput_tok_s": summary["throughput_tok_s"],
        "generation_tok_s": summary["generation_tok_s"],
        "latency_ms": summary["latency_ms"],
        "ttft_ms": summary["ttft_ms"],
        "requests_ok": summary["requests_ok"],
        "wall_time_s": summary["wall_time_s"],
        "target": summary["target"],
        "model": summary["model"],
    }
    numbers_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a JSONL trace.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", default="traces/sample.jsonl")
    parser.add_argument("--output", default="traces/replay_results.json")
    parser.add_argument("--numbers", default="demo/numbers.json")
    parser.add_argument("--speed", type=float, default=10.0, help="Time compression factor")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--api",
        choices=["auto", "chat", "completions"],
        default="auto",
        help="API to use: chat (Instruct models), completions (opt-125m etc.), auto (detect)",
    )
    args = parser.parse_args()

    api_mode = resolve_api_mode(args.model, args.api)

    trace_path = Path(args.trace)
    if not trace_path.exists():
        print(f"Trace not found: {trace_path}. Run: make sample-trace", file=sys.stderr)
        return 1

    trace = load_trace(trace_path)
    if not trace:
        print(f"No records in trace: {trace_path}", file=sys.stderr)
        return 1

    print(
        f"Replaying {len(trace)} requests to {args.target} "
        f"(speed={args.speed}x, api={api_mode})"
    )
    results, wall_time = asyncio.run(
        replay(
            args.target,
            args.model,
            trace,
            args.speed,
            args.concurrency,
            args.warmup,
            api_mode,
        )
    )
    summary = summarize(results, wall_time, str(trace_path), args.target, args.model)
    err_counts = Counter(
        r.get("error", "unknown") for r in summary["results"] if r["status"] != "ok"
    )
    # #region agent log
    _agent_log(
        "H5",
        "replay.py:main",
        "replay_done",
        {
            "ok": summary["requests_ok"],
            "err": summary["requests_error"],
            "error_histogram": dict(err_counts),
        },
    )
    # #endregion
    print_summary(summary)
    if summary["requests_error"] > 0:
        print("Errors (sample):", list(err_counts.items())[:5])
        if any("chat template" in str(e).lower() for e in err_counts):
            print(
                "\nHint: this model needs the completions API:\n"
                "  make replay TARGET=... MODEL=... API=completions",
                file=sys.stderr,
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote detailed results to {output_path}")

    write_numbers_json(summary, Path(args.numbers))
    print(f"Updated {args.numbers}")

    return 0 if summary["requests_error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
