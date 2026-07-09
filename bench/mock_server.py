#!/usr/bin/env python3
"""Minimal OpenAI-compatible mock server for measurement smoke tests."""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            body = {"data": [{"id": "mock-model"}]}
            self._json(200, body)
        elif self.path == "/metrics":
            body = (
                'vllm:num_prefill_requests{server="mock"} 20\n'
                'vllm:num_decoding_requests{server="mock"} 80\n'
                'vllm:num_requests_waiting{server="mock"} 3\n'
                'vllm:gpu_prefix_cache_hit_rate{server="mock"} 0.55\n'
            )
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        max_tokens = int(payload.get("max_tokens", 16))
        stream = payload.get("stream", False)
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            time.sleep(0.02)
            chunk = {
                "choices": [{"delta": {"content": "ok"}, "index": 0}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            usage = {
                "choices": [],
                "usage": {"prompt_tokens": 120, "completion_tokens": max_tokens // 2},
            }
            self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            body = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": max_tokens // 2},
            }
            self._json(200, body)

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Mock server on http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
