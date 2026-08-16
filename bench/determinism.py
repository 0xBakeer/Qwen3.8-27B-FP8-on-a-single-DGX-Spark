#!/usr/bin/env python3
"""Output stability across serving configurations.

Runs a fixed prompt set at temperature 0 and hashes each completion, so two
configurations can be compared byte-for-byte. Used to answer "does turning on
speculative decoding and prefix caching change what the model writes?"

Run it three times to get a meaningful answer:

    python determinism.py out_tuned.json    # tuned engine
    # restart with --no-enable-prefix-caching and no --speculative-config
    python determinism.py out_stock.json    # stock engine
    python determinism.py out_stock2.json   # CONTROL: same engine, again

The control matters more than the comparison. Without it you cannot tell a real
configuration effect from ordinary run-to-run variation. On the hardware in
RESULTS.md the control came back 20/20 identical, which is what made the
8/20 tuned-vs-stock result interpretable. See EVAL.md.

Usage:  python determinism.py <output.json> [base_url] [model]
"""
import hashlib
import json
import sys
import urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else "out.json"
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8002/v1"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen3.8-27b"

# Deliberately mixed: some have one defensible answer (arithmetic, regex, SQL),
# some are open enough that near-tied tokens are likely (prose, verse). The
# second group is where configuration differences show up first.
PROMPTS = [
    "Write a Python function that merges two sorted lists. Code only.",
    "Explain why B-trees are used for database indexes instead of binary search trees.",
    "What is 17 * 243? Show the arithmetic.",
    "Write a SQL query returning the top 3 customers by total order value.",
    "Summarize the tradeoff between optimistic and pessimistic locking in two sentences.",
    "Translate to German: 'The deployment failed because the certificate expired.'",
    "Write a bash one-liner that finds files larger than 100MB modified in the last week.",
    "List the steps of the TCP three-way handshake.",
    "Refactor this to remove duplication: def a(x): return x*2+1\ndef b(x): return x*2+5",
    "What does the CAP theorem actually say? Be precise.",
    "Write a regex matching an ISO-8601 date with optional time.",
    "Explain memory bandwidth vs compute bound in one paragraph.",
    "Write a Go struct and method for an LRU cache node.",
    "Why does floating point 0.1 + 0.2 != 0.3?",
    "Give me a Dockerfile for a static Rust binary.",
    "Describe what a Bloom filter cannot do.",
    "Write a haiku about slow compile times.",
    "What is the difference between a mutex and a semaphore?",
    "Produce a JSON schema for a user with id, email and optional age.",
    "Explain tail call optimization and why Python lacks it.",
]


def complete(prompt: str) -> tuple[str, int]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0,
        "seed": 42,
        # Thinking is on by default at high effort on this model; without this
        # the run measures reasoning tokens and takes very much longer.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return resp["choices"][0]["message"]["content"], resp["usage"]["completion_tokens"]


def compare(path_a: str, path_b: str) -> None:
    """Report byte-identical count and where each divergence begins."""
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    same = sum(1 for k in a if a[k]["sha"] == b[k]["sha"])
    print(f"IDENTICAL: {same}/{len(a)}")
    for k in sorted(a, key=int):
        if a[k]["sha"] == b[k]["sha"]:
            continue
        ta, tb = a[k]["text"], b[k]["text"]
        i = 0
        while i < min(len(ta), len(tb)) and ta[i] == tb[i]:
            i += 1
        print(f"\n  {a[k]['prompt'][:70]}")
        print(f"    common : ...{ta[max(0, i - 60):i]!r}")
        print(f"    file A : {ta[i:i + 60]!r}")
        print(f"    file B : {tb[i:i + 60]!r}")
        print(f"    diverges at char {i}")


if __name__ == "__main__":
    if OUT == "--compare":
        compare(sys.argv[2], sys.argv[3])
        sys.exit(0)

    print(f"endpoint {BASE}  model {MODEL}  -> {OUT}")
    results = {}
    for i, prompt in enumerate(PROMPTS):
        text, tokens = complete(prompt)
        sha = hashlib.sha256(text.encode()).hexdigest()[:16]
        results[str(i)] = {"prompt": prompt, "text": text, "sha": sha, "tokens": tokens}
        print(f"  {i + 1:2d}/{len(PROMPTS)}  {sha}  {tokens:3d} tok", flush=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"saved {OUT}")
    print(f"compare with: python {sys.argv[0]} --compare {OUT} <other.json>")
