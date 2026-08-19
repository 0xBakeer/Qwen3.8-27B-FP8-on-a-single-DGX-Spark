# DFlash 2 on Qwen3.8-27B-FP8

A third drafter, measured against the MTP and DSpark configurations already in this repo.
**DFlash 2 is the fastest single-stream option on FP8 — and FP8 is the only checkpoint that
can run it in vLLM at all.**

DFlash 2 is a *speculative-decoding drafter*, not an attention mechanism. It predicts an
entire block in one pass (where MTP builds its block with one sequential pass per token),
then a selector traces a coherent path through each position's top-16 candidates and a
two-tap convolution limits decay toward the end of the block.

## Method for this section

These numbers come from a **different harness** than `bench/` — a separate suite written for
the DFlash2 evaluation. They are internally consistent but are **not** directly comparable,
cell for cell, with the DSpark/MTP tables elsewhere in this repo, which use different prompts
and output lengths. Where the two overlap they agree to within a few percent (this harness
measures stock FP8 at 7.94 tok/s against this repo's 7.88).

- 4 prompts — code generation, prose explanation, arithmetic word problem, refactor —
  3 repetitions each, `temperature=0`, thinking disabled, 512 output tokens.
- Reported single-stream figure is the **mean of per-prompt medians**, and excludes TTFT
  (`completion_tokens / (wall - ttft)`).
- Concurrency sweep uses 384 output tokens and reports **aggregate** throughput including
  TTFT, so its c=1 point reads slightly below the single-stream decode figure. Prompts are
  assigned round-robin, so **c=1 and c=2 use an unbalanced prompt mix**; c≥4 is balanced and
  is the comparable part of the curve.
- Acceptance length is `1 + accepted/drafts` from the engine's own `spec_decode_*` counters,
  deltaed across the run — the same definition Inco AI use.
- Engine flags held constant across every configuration: `--max-model-len 131072
  --gpu-memory-utilization 0.80 --max-num-batched-tokens 16384 --enable-prefix-caching`,
  image/video input disabled. Sole occupant of the device.


## 1. Single-stream, generative

| drafter | chat | code | math | refactor | **mean** | acceptance |
|---|---:|---:|---:|---:|---:|---:|
| none | 7.94 | 7.96 | 7.93 | 7.93 | **7.94** | — |
| MTP `k=3` | 14.50 | 20.51 | 19.90 | 17.06 | **17.99** | 3.165 |
| DSpark `k=7` | 12.56 | 21.63 | 22.43 | 19.85 | **19.12** | 2.875 |
| **DFlash2 `k=7`** | **17.74** | **38.34** | **39.59** | **31.21** | **31.72** | **4.607** |

3.99x over stock, and 1.66x over DSpark `k=7`.
Inco AI publish 3.1-3.4x on an H200; the larger multiplier here is expected — this device is
far more bandwidth-starved, so unspeculated decode leaves more idle bandwidth for speculation
to spend.

**Chat is the worst case and it is not close.** 17.74 tok/s on prose
against 39.59 on arithmetic. Unpredictable prose gives the drafter
nothing to latch onto. Note also that on chat alone the ordering flips: MTP `k=3`
(14.50) beats DSpark `k=7` (12.56), because a
shorter draft wastes less when acceptance is low.

## 1a. DFlash2 measured on this repo's own harness

The section above uses a separate harness. To place DFlash2 on the *same* scale as the
numbers in [RESULTS.md](RESULTS.md), it was also run through `bench/edit_bench.py`,
`bench/conc_bench.py` and `bench/prefill_bench.py` unchanged, at this repo's own serving
settings (`gmu 0.85`, `max-model-len 262144`, sole occupant). Raw output is in
`bench/dflash2/results-orig/`.

**DSpark `k=14` was re-run at the same time as a control.** It reproduces:

| | published (2026-08-16) | re-measured | delta |
|---|---:|---:|---:|
| fresh-code | 18.77 | 18.87 | +0.5% |
| EDIT-heavy | 58.5 | 58.19 | −0.5% |
| mean tokens/pass | 10.62 | 10.42 | −1.9% |
| c8 aggregate | 119.7 | 116.09 | −3.0% |

So the device has not drifted, and the DFlash2 figures below sit on the same footing as
everything else in this repo.

### Single stream

| config | fresh-code | EDIT-heavy | tokens/pass | accept |
|---|---:|---:|---:|---:|
| stock (published) | 7.88 | 7.88 | — | — |
| DSpark `k=7` (published) | 20.05 | 46.8 | 7.91 | 98.7% |
| DSpark `k=14` (re-measured) | 18.87 | **58.19** | 10.42 / 15 | 67.3% |
| **DFlash2 `k=7`** | **29.86** | 46.51 | **7.97 / 8** | **99.6%** |

**Fresh generation: DFlash2 is 49% faster than the best previous FP8 configuration**
(29.86 vs 20.05). It also edges past this device's *4-bit* fresh numbers — NVFP4 + DSpark
`k=7` at 29.23 and `k=14` at 29.55 — while running the unmodified FP8 checkpoint.

**Edit-heavy: DFlash2 ties DSpark `k=7` and loses to `k=14` by 21%.** Not because it drafts
worse — it drafts far better, 99.6% acceptance against 67.3% — but because it converts
**7.97 of a possible 8.00** tokens per pass while `k=14` converts 10.42 of a possible 15.
DFlash2 is at **99.6% of its structural ceiling** here. No tuning remains in it; only a
larger block would help, and that needs a retrained drafter.

### Concurrency (aggregate tok/s, distinct prompts, 1,500 output tokens)

| c | DSpark `k=7` (published) | **DFlash2 `k=7`** | DSpark `k=14` (re-measured) |
|---|---:|---:|---:|
| 1 | 46.91 | 47.37 | **55.85** |
| 4 | **134.21** | 101.49 | 85.52 |
| 8 | **208.71** | 119.95 | 116.09 |
| 16 | **256.08** | 158.23 | 119.16 |

Throughput at concurrency is **monotonic in draft depth, inverted**: `k=7` dominates from
c4, DFlash2 (block 8) sits in the middle, `k=14` is worst. Deep drafts win alone and lose
under load, because rejected draft tokens consume compute that the batch would otherwise
spend on real sequences. DSpark `k=14` is the fastest single-stream config here (55.85 at
c1) and the slowest at c16 (119.16).

### Prefill — DFlash2 costs nothing

| prompt | DFlash2 `k=7` | DSpark `k=14` |
|---|---:|---:|
| ~8K | 936.7 | 968.5 |
| ~32K | 844.8 | 870.0 |
| ~100K | 690.4 | 688.5 |

Within 3% of each other, so **DFlash2 carries no prefill penalty**. Both sit ~35% below the
published FP8 prefill figures (1,506 / 1,312 / 939), which were taken at `k=7`; deeper
speculation reduces vLLM's `max_num_scheduled_tokens` and that is the likely cause. A
same-day `k=7` prefill run would confirm it and was not made — so treat the published
prefill numbers as valid for `k=7` only, and do not attribute the difference to DFlash2.

### What this means

Three regimes, three different winners, all on this repo's harness:

| regime | best config | number |
|---|---|---:|
| fresh generation, 1 stream | **DFlash2 `k=7`** | 29.86 tok/s |
| edit-heavy, 1 stream | **DSpark `k=14`** | 58.19 tok/s |
| concurrency >= 4 | **DSpark `k=7`** | 256.08 at c16 |

DFlash2 is a real win, and a narrow one: it owns fresh single-stream generation outright and
nothing else.

## 2. Edit-heavy

| config | tok/s | acceptance | accept rate |
|---|---:|---:|---:|
| DFlash2 `k=7` (FP8) | 49.20 | 7.808 / 8.00 | 97.26% |

DFlash2 reaches **97.6% of its structural ceiling** on edit workloads — it is saturated, and
no better selector can improve it. The only remaining lever is a larger block, which needs a
retrained drafter. See §4.

## 3. Concurrency, c=1 to c=16

Aggregate tok/s:

| config | c1 | c2 | c4 | c8 | c12 | c16 |
|---|---:|---:|---:|---:|---:|---:|
| no speculation | 7.86 | 15.75 | 30.69 | 59.00 | 83.68 | 107.07 |
| **DFlash2 `k=7`** | 34.68 | 35.80 | 67.87 | 115.50 | 142.76 | 178.56 |
| NVFP4 + MTP `k=3` * | 28.85 | 37.79 | 77.60 | 126.72 | 166.72 | 219.82 |

\* different weights, included because it is the configuration that wins above c=1.

Speedup over the FP8 no-speculation baseline at the same concurrency:

| config | c1 | c2 | c4 | c8 | c12 | c16 |
|---|---:|---:|---:|---:|---:|---:|
| DFlash2 `k=7` | 4.41x | 2.27x | 2.21x | 1.96x | 1.71x | 1.67x |

TTFT p50 (seconds):

| config | c1 | c2 | c4 | c8 | c12 | c16 |
|---|---:|---:|---:|---:|---:|---:|
| no speculation | 0.270 | 0.250 | 0.366 | 0.500 | 0.587 | 0.818 |
| DFlash2 `k=7` | 0.322 | 0.328 | 0.573 | 0.619 | 0.867 | 1.181 |

Acceptance length across the sweep:

| config | c1 | c2 | c4 | c8 | c12 | c16 |
|---|---:|---:|---:|---:|---:|---:|
| DFlash2 `k=7` | 5.691 | 3.914 | 4.589 | 4.32 | 4.348 | 4.272 |

**Acceptance is essentially concurrency-invariant** — 4.589
at c4 to 4.272 at c16, a 7% decay across a 4x batch
increase. The drafter does not degrade under load. The *speedup ratio* falls anyway, because
the unspeculated baseline improves as batching amortises the weight read: it scales
13.6x across the range at 85% efficiency.

**The crossover is at c=2.** FP8 + DFlash2 leads at c1 (34.68 vs
28.85) and loses at every point above it, by a widening margin —
+23% at c16. If you serve more than one request at
a time, 4-bit weights with free in-checkpoint MTP beat the best drafter available on FP8.

## 4. The constraint: DFlash2 needs an unquantized LM head

Not documented in the announcement or the model card.

```
ValueError: DFlash2 requires an unquantized target LM head for candidate TopK.
```

The selector reads the target's top-16 candidates directly off the LM head, so vLLM refuses
any checkpoint whose head is quantized:

```python
if not isinstance(self.lm_head.quant_method, UnquantizedEmbeddingMethod):
    raise ValueError(...)
```

| checkpoint | `lm_head` tensors | DFlash2 in vLLM |
|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | `lm_head.weight` (head in `ignore` list) | **works** |
| `unsloth/Qwen3.8-27B-NVFP4` | `lm_head.weight` + `lm_head.weight_scale` | blocked |
| `Pilcothink/…-MixedInt4-AutoRound` | `lm_head.weight` (bf16) but vLLM still assigns a quantized method | blocked |

Pre-flight check:

```bash
python3 -c "import json;w=json.load(open('model.safetensors.index.json'))['weight_map'];
print([k for k in w if k.startswith('lm_head')])"
# ['lm_head.weight']                        -> fine
# ['lm_head.weight','lm_head.weight_scale'] -> DFlash2 will not run
```

**This is the reason to keep an FP8 checkpoint around.** It is the only one of the three that
serves DFlash2 under vLLM. (llama.cpp has no such restriction — see the 4-bit repo.)

## 5. Cost

| | no spec | DFlash2 `k=7` |
|---|---:|---:|
| model load | 27.64 GiB | 31.41 GiB (+3.77 drafter) |
| available KV | 62.28 GiB | 57.16 GiB |
| **KV cache tokens** | 980,026 | **617,152** (-37%) |
| load time | 238 s | 268 s |

KV capacity falls 37% while memory falls only 8%, because DFlash2 forces vLLM's V2 model
runner with a second KV group for the drafter's own attention. At 617K tokens against a 128K
context that is still ~4.7 concurrent full-length requests.

## 6. Setup

vLLM PR #52816 is **pure Python** — 11 files, Triton kernels only, no C++/CUDA — and vLLM
0.27.1 already ships DFlash **1**. So it can be overlaid onto the stock image with bind
mounts rather than rebuilt:

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
```

`k` is capped at **7** by the drafter's `dflash_config.block_size: 8`. Measured `k=5` is slower
(33.29 vs 37.58 tok/s on Q4_K_M), so 7 is the right value, not merely the maximum.

**Verify it is actually active.** A DFlash2 checkpoint whose speculator falls back drafts as
DFlash1 *silently* — the PR forces the V2 model runner specifically to prevent this. Confirm
the V2 speculator was selected before trusting any measurement.

## 7. Caveats

- The DSpark drafter compared against here is `Doopeworld/Qwen3.8-27B-DSpark-vLLM`. Inco AI
  benchmarked `RadixArk/Qwen3.8-27B-DSpark`. Ours appears to be the weaker of the two, so the
  DFlash2-over-DSpark margin here (+60% acceptance) is
  wider than their published +33% and should not be read as a like-for-like comparison.
- MTP runs `k=3` in vLLM here (its acceptance is capped at 4.00 by construction) against
  DFlash2's `k=7`. On llama.cpp, where both were run at `k=7`, the acceptance gap narrows to
  +8.3% — much closer to Inco AI's published +12.1%.
- The c=1 and c=2 sweep points use an unbalanced prompt mix; c>=4 is balanced.
- No NVFP4 no-speculation curve was swept, so the NVFP4 rows carry absolute numbers only.

See [SOURCES.md](SOURCES.md).
