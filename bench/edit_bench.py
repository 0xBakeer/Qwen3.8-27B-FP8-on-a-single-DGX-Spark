#!/usr/bin/env python3
"""Single-stream throughput on two workloads with opposite output predictability.

  fresh generation - nothing in the prompt to copy from; drafter acceptance is low
  edit-heavy       - most of the output already exists in the prompt; acceptance ~99%

Speculative decoding behaves completely differently on the two, which is the whole
point of measuring both. Also reports draft acceptance and mean tokens per forward
pass, read from the engine's own counters.

Usage:  python edit_bench.py [base_url] [model]
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8002/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"
METRICS = BASE.rsplit("/v1", 1)[0] + "/metrics"

FRESH = (
    "Write a complete Python module implementing an LRU cache with TTL expiry, "
    "thread safety, and a decorator API. Include docstrings and type hints. Code only."
)


def _source(n_classes: int = 45) -> str:
    src = '"""Inventory service helpers."""\nfrom dataclasses import dataclass, field\n'
    for i in range(1, n_classes + 1):
        src += f'''

@dataclass
class Item{i}:
    """Represents inventory item {i}."""
    sku: str
    quantity: int = 0
    price_cents: int = 0
    tags: list[str] = field(default_factory=list)

    def restock(self, n: int) -> None:
        """Add n units to the on-hand quantity."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self.quantity += n

    def total_value(self) -> int:
        """Return the total value of this line in cents."""
        return self.quantity * self.price_cents
'''
    return src


EDIT = (
    "Here is a Python module. Add a `discount(self, pct: int) -> int` method to EVERY "
    "Item class, returning the discounted total value. Output the COMPLETE modified "
    "file, nothing else.\n\n```python\n" + _source() + "\n```"
)


def spec_counters() -> dict:
    """Draft/accept totals from the engine. Empty when speculation is disabled."""
    try:
        body = urllib.request.urlopen(METRICS, timeout=30).read().decode()
    except Exception:
        return {}
    keys = {
        "vllm:spec_decode_num_drafts_total": "drafts",
        "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
        "vllm:spec_decode_num_accepted_tokens_total": "accepted",
    }
    out = {}
    for line in body.splitlines():
        for prefix, name in keys.items():
            if line.startswith(prefix):
                out[name] = float(line.split()[-1])
    return out


def run(prompt: str, max_tokens: int, label: str) -> float:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        # Thinking is on by default at high effort on this model. Without this the
        # benchmark measures reasoning-token generation, not decode throughput.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    before = spec_counters()
    start = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    elapsed = time.time() - start
    after = spec_counters()

    tokens = resp["usage"]["completion_tokens"]
    drafts = after.get("drafts", 0) - before.get("drafts", 0)
    draft_tokens = after.get("draft_tokens", 0) - before.get("draft_tokens", 0)
    accepted = after.get("accepted", 0) - before.get("accepted", 0)

    rate = tokens / elapsed
    if drafts:
        accept_pct = accepted / draft_tokens * 100 if draft_tokens else 0.0
        mean = 1 + accepted / drafts
        print(f"{label:16s} {tokens:5d} tok /{elapsed:7.1f}s = {rate:6.2f} tok/s "
              f"| accept {accept_pct:5.1f}% | mean {mean:4.2f} tok/pass")
    else:
        print(f"{label:16s} {tokens:5d} tok /{elapsed:7.1f}s = {rate:6.2f} tok/s "
              f"| no speculation")
    return rate


if __name__ == "__main__":
    print(f"endpoint {BASE}  model {MODEL}")
    run(FRESH, 32, "warmup")          # first request pays kernel/graph warmup
    run(FRESH, 400, "fresh-code")
    run(EDIT, 3000, "EDIT-heavy")
    run(EDIT, 3000, "EDIT-heavy #2")  # second run benefits from a warm prefix cache
