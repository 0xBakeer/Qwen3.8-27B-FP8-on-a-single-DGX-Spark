# Does the tuned config change the model's output?

Short answer: **yes, the text changes — but not because the drafter overrules the
model.** Within a fixed configuration the engine is bit-deterministic. Between two
configurations the text diverges, because changing the batch geometry changes
floating-point reduction order, which flips the argmax on near-tied tokens.

This document reports that test in full, including every divergence. It measures
**output stability, not quality** — see [What this does not establish](#what-this-does-not-establish).

## Method

20 prompts spanning code, SQL, arithmetic, regex, translation, explanation and
verse. Each run at `temperature=0`, `seed=42`, `max_tokens=300`, thinking
disabled. Completions hashed with SHA-256 and compared byte-for-byte.

| Config | Flags |
|---|---|
| **A — tuned** | `--enable-prefix-caching` + DSpark `k=7` |
| **B — stock** | `--no-enable-prefix-caching`, no speculative decoding |

Everything else identical: `Qwen/Qwen3.8-27B-FP8`, `max-model-len 262144`,
`gpu-memory-utilization 0.85`, sole occupant, vLLM `v0.27.1-aarch64`.

## Results

| Comparison | Byte-identical |
|---|---|
| **Control** — stock vs stock, same engine, run twice | **20 / 20** |
| **Test** — tuned vs stock | **8 / 20** |

The control is the important half. It establishes that the engine is
deterministic within a configuration, which means the 12 divergences below are
**caused by the config change** and are not run-to-run noise.

## Why the text diverges

Speculative decoding is verified, not trusted: the drafter proposes, and the full
model keeps a proposal only if it matches the token it would have produced itself.
That guarantee is intact here — nothing in this test shows the drafter forcing a
token through.

What changes is the *shape of the arithmetic*. Verifying eight tokens in one
forward pass instead of decoding one changes the batch dimensions, which changes
the order of floating-point accumulation inside the GEMMs. Logits shift by
fractions of a percent. Where two candidate tokens are near-tied, that is enough
to flip the argmax, and greedy decoding then follows a different branch for the
rest of the completion.

Prefix caching contributes the same way: KV entries computed in a different batch
context are equal in exact arithmetic and marginally different in floating point.

**Consequence:** the outputs are different samples from the same distribution, not
degraded ones. That is a claim about mechanism, not a measurement of quality.

## Every divergence

Common prefix shown up to the first differing character.

### 1. Haiku about slow compile times — diverges at char 37 of 18 tokens
```
common  Terminal blinks,\nCoffee cools in the
tuned   waiting,\nCode finally runs.
stock   silence,\nCode finally runs.
```

### 2. CAP theorem — diverges at char 23
```
common  To be precise, the CAP
tuned   theorem (also known as Brewer's Theorem) is a statement abou
stock   Theorem (also known as Brewer's Theorem) is a statement abou
```

### 3. 17 × 243 — diverges at char 54
```
common  To calculate $17 \times 243$, we can use the standard
tuned   long multiplication method or break the numbers down using t
stock   multiplication algorithm (long multiplication) or break the
```

### 4. Optimistic vs pessimistic locking — diverges at char 70 (57 vs 62 tokens)
```
common  Optimistic locking assumes conflicts are rare and only validates data
tuned   integrity at commit time, offering higher concurrency but ri
stock   at commit time, offering higher concurrency and throughput b
```

### 5. Floating point 0.1 + 0.2 — diverges at char 86
```
common  ## The Core Issue: Binary Representation\n\n
tuned   Floating-point numbers (IEEE 754 double precision) store val
stock   Computers store floating-point numbers in **binary** (base-2
```

### 6. Dockerfile for a static Rust binary — diverges at char 138
```
common  FROM rust:1.78-slim AS builder\n\n# Install build dependencies
tuned   for static linking\nRUN apt-get update && apt-get install -y
stock   (needed for linking, even for static builds)\nRUN apt-get upd
```

### 7. Memory bandwidth vs compute bound — diverges at char 166 (204 vs 193 tokens)
```
common  is defined by which resource acts as the bottleneck limiting overall
tuned   throughput. A workload is **memory-bound** when the time req
stock   performance. A workload is **memory-bound** when the time re
```

### 8. B-trees vs binary search trees — diverges at char 232
```
common  on disk (or SSDs), not in memory**. This changes the performance model
tuned   dramatically.\n\n### Binary Search Tree (BST) Limitations
stock   entirely.\n\n### Binary Search Tree (BST) Limitations
```

### 9. Tail call optimization — diverges at char 247
```
common  when a function's **last operation** is a call to another function (a
tuned   *tail call*).\n\n### Key Concept: Tail Position
stock   "tail call").\n\n### Key Concept: Tail Position
```

### 10. What a Bloom filter cannot do — diverges at char 417
```
common  **The Limitation:** A Bloom filter can only
tuned   provide two types of answers:\n    1.  **"Definitely Not Pres
stock   return two types of answers:\n    *   **"Not in the set"**: T
```

### 11. TCP three-way handshake — diverges at char 772
```
common  - An acknowledgment number (ACK)
tuned   equal to the client's ISN + 1.\n  - A random initial sequence
stock   = client's ISN + 1.\n  - A random initial sequence number (IS
```

### 12. Mutex vs semaphore — diverges at char 926
```
common  Any thread can release the semaphore (increment the count). |\n| **
tuned   Deadlock Risk** | High if not used carefully (e.g., nested l
stock   Priority Inheritance** | Often supported (to prevent priorit
```

## Reading the divergences

Every one is a **synonym, a formatting choice, or a different-but-valid ordering**:
`theorem`/`Theorem`, `method`/`algorithm`, `dramatically`/`entirely`,
`waiting`/`silence`, `equal to`/`=`, a table row appearing in a different order.
None of the twelve introduces a factual error, and no arithmetic, code or
structured output changed in a way that alters correctness.

Note also that eight prompts produced **identical** text despite the config
change, and several of the divergences appear only after 700–900 characters of
byte-identical output. That is the signature of an occasional near-tie, not of a
systematically different model.

**This is suggestive, not conclusive.** Twelve informal readings are not an
evaluation.

## What this does not establish

- **It does not measure quality.** A byte-diff can tell you the text changed; it
  cannot tell you whether it got better, worse, or neither. Answering that needs a
  scored benchmark run under both configurations and compared on accuracy.
- **20 prompts is a small sample**, chosen by hand and not stratified.
- **It says nothing about sampled generation.** Everything here is `temperature=0`.
- **It says nothing about the 4-bit checkpoints.** Those change the weights, and
  are a separate question addressed nowhere in this repository.
- The mechanism explanation (batch geometry → reduction order → argmax flips) is
  the standard account of this behaviour and is consistent with every observation
  here, but it was not proven by instrumenting the kernels.

## Update: the quality question, answered

The section above says a byte-diff cannot tell you whether output got better or worse, and
that answering it "needs a scored benchmark run under both configurations and compared on
accuracy". That has since been run.

| configuration | GSM8K (n=200) | MMLU (n=400) |
|---|---:|---:|
| FP8, no speculation | 93.5% | 84.0% |
| FP8 + DFlash 2 `k=7` | 92.5% | 84.0% |
| int4 AutoRound, no speculation | 94.0% | 82.0% |

Greedy, thinking disabled. Every difference sits inside its confidence interval
(+/-3.3 pp on GSM8K, +/-3.7 pp on MMLU at 95 %), so **none of them is established as real**.

Aggregate scores are weak evidence — two runs can score identically while disagreeing on many
individual items — so per-item agreement is the sharper test:

| comparison | GSM8K | MMLU |
|---|---:|---:|
| DFlash 2 vs no speculation | 94.0 % | **99.8 %** |
| FP8 vs int4 (both unspeculated) | 93.5 % | 96.2 % |

**Speculation perturbs the output less than changing precision does.** DFlash 2 agrees with
unspeculated decoding on 399 of 400 MMLU items; FP8 and int4 agree on 385. That is consistent
with the mechanism this document already describes: the drafter never overrules the model, and
the residual divergence is floating-point reduction order flipping the argmax on near-ties.

Still not established: this used a 200/400-item sample, enough to exclude a large regression
but not to resolve the 2 pp MMLU gap between FP8 and int4 (that needs ~2,000 items). And it
covers two tasks, not the space of things people use the model for.

## Reproducing

```bash
python bench/determinism.py out_tuned.json      # against the tuned engine
# restart with --no-enable-prefix-caching and no --speculative-config
python bench/determinism.py out_stock.json      # against the stock engine
python bench/determinism.py out_stock2.json     # control: same engine again
```

Then compare the `sha` field per prompt across the JSON files.
