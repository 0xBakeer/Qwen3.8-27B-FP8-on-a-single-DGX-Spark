#!/usr/bin/env python3
"""Concurrency sweep: aggregate throughput at c1 / c4 / c8.

Aggregate throughput and single-stream latency are different measurements and differ
by several times on this hardware. Published figures frequently quote one without
saying which, so measure both before comparing against anything.

By default each stream gets a DISTINCT prompt. Sending the same prompt to every
stream lets prefix caching serve most of them and inflates the result - measured at
about 11% here. Pass --identical to see that effect for yourself.

Usage:  python conc_bench.py [base_url] [model] [--identical] [--levels=1,4,8,16]
"""
import json
import sys
import threading
import time
import urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--")]
IDENTICAL = "--identical" in sys.argv
BASE = args[0] if len(args) > 0 else "http://127.0.0.1:8002/v1"
MODEL = args[1] if len(args) > 1 else "qwen3.8-27b"

CONCURRENCIES = [int(x) for x in
                 (next((a.split("=",1)[1] for a in sys.argv if a.startswith("--levels=")),
                       "1,4,8")).split(",")]
MAX_TOKENS = 1500


def _source(seed: int, n_classes: int = 45) -> str:
    """Distinct class names and values per seed, so prefixes genuinely differ."""
    src = f'"""Inventory module variant {seed}."""\nfrom dataclasses import dataclass\n'
    for i in range(1, n_classes + 1):
        src += f'''

@dataclass
class Item{seed}_{i}:
    sku_{seed}: str
    quantity: int = {i}
    price_cents: int = {i * 7 + seed}

    def restock(self, n: int) -> None:
        self.quantity += n

    def total_value(self) -> int:
        return self.quantity * self.price_cents
'''
    return src


def prompt_for(stream: int) -> str:
    seed = 0 if IDENTICAL else stream
    return (
        "Here is a Python module. Add a `discount(self, pct: int) -> int` method to "
        "EVERY Item class, returning the discounted total value. Output the COMPLETE "
        "modified file, nothing else.\n\n```python\n" + _source(seed) + "\n```"
    )


def one(results: dict, stream: int, max_tokens: int) -> None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_for(stream)}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    results[stream] = resp["usage"]["completion_tokens"]


if __name__ == "__main__":
    mode = "IDENTICAL" if IDENTICAL else "DISTINCT"
    print(f"endpoint {BASE}  model {MODEL}  prompts: {mode}")
    one({}, 0, 32)  # warmup

    for c in CONCURRENCIES:
        results: dict = {}
        threads = [threading.Thread(target=one, args=(results, i, MAX_TOKENS))
                   for i in range(c)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start
        total = sum(results.values())
        print(f"{mode} c{c}: {total:6d} tok / {elapsed:6.1f}s = "
              f"{total / elapsed:7.2f} tok/s aggregate | "
              f"{total / elapsed / c:6.2f} per stream")
