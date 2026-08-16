#!/usr/bin/env python3
"""Prefill throughput across prompt lengths.

Decode is memory-bandwidth-bound; prefill is compute-bound. That matters when comparing
quantizations on hardware without native low-precision compute: 4-bit weights must be
dequantized before the matrix multiply, which is work that prefill pays for and decode
largely hides. So a quantization that speeds decode up can slow prefill down.

Each prompt is made unique by a seed so prefix caching cannot serve it. `max_tokens=1`
isolates prefill from decode.

Usage:  python prefill_bench.py [base_url] [model]
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8002/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"

# (repetitions, label) - repetitions chosen to land near 8K / 32K / 100K tokens
LENGTHS = [(230, "~8K"), (950, "~32K"), (3000, "~100K")]


def filler(seed: int, reps: int) -> str:
    """Unique per seed, so no run can hit another run's cached prefix."""
    out = [f"Reference document {seed}. Engineering conventions, revision {seed}."]
    for i in range(reps):
        out.append(
            f"Rule {seed}_{i}: wrap errors with context, never panic in request "
            f"handlers, document exported symbols with complete sentences, and keep "
            f"handler functions under sixty lines. Owner is team {(i * 7 + seed) % 97}."
        )
    return "\n".join(out)


def prefill(seed: int, reps: int) -> tuple[float, int]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": filler(seed, reps)},
            {"role": "user", "content": "Reply with the single word OK."},
        ],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return time.time() - start, resp["usage"]["prompt_tokens"]


if __name__ == "__main__":
    print(f"endpoint {BASE}  model {MODEL}")
    prefill(0, 20)  # warmup

    seed = 1000
    for reps, label in LENGTHS:
        # Two distinct seeds: both are cold, so this measures run-to-run spread rather
        # than a cache effect.
        results = []
        for _ in range(2):
            seed += 1
            elapsed, tokens = prefill(seed, reps)
            results.append((elapsed, tokens))
        best = min(results, key=lambda r: r[0])
        worst = max(results, key=lambda r: r[0])
        toks = best[1]
        print(f"{label:>6} prompt {toks:7d} tok | "
              f"{best[0]:6.2f}s = {toks / best[0]:8.1f} tok/s prefill "
              f"(slower run {worst[0]:6.2f}s = {toks / worst[0]:8.1f} tok/s)")
