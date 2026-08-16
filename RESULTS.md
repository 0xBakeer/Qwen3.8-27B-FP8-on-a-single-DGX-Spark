# Measurements — Qwen3.8-27B-FP8 on one DGX Spark

Every figure below was measured on the hardware described in the README, against a
live vLLM server, with `temperature=0` and thinking disabled. Where two numbers appear
for the same cell they are two consecutive runs of the same benchmark.

## Method

Two workloads, chosen because they sit at opposite ends of output predictability:

- **Fresh generation** — "write a Python module implementing an LRU cache with TTL
  expiry, thread safety, and a decorator API", 400 output tokens. Nothing useful in the
  prompt to copy from.
- **Edit-heavy** — a ~2,000 token Python source file plus "add a method to every class,
  output the complete modified file", 3,000 output tokens. Most of the output already
  exists in the prompt.

Reported metrics: throughput (completion tokens ÷ wall clock), draft **acceptance rate**
(accepted ÷ proposed draft tokens) and **mean tokens per forward pass** (`1 + accepted ÷
drafts`), both read from the engine's own `spec_decode_*` counters.

## 1. Baseline — the cost of the defaults

Shared GPU (three other engines resident but idle), `gpu-memory-utilization 0.44`.

| | Fresh generation | Prose |
|---|---:|---:|
| Stock: no speculation, no prefix caching | **7.88** | **7.88** |
| MTP `k=3` | 19.00 | 13.35 |
| MTP `k=3` + prefix caching | 18.91 | 14.06 |

**2.4× on generation from speculative decoding alone**, on unchanged weights.

MTP `k=3` acceptance by draft position on fresh generation: **82.1 % / 65.9 % / 47.9 %**,
mean **2.96** tokens per forward pass. Note the official FP8 checkpoint ships its 22
`mtp.*` tensors quantized to FP8 as well, and still drafts well.

Enabling prefix caching cost nothing on decode (18.91 vs 19.00 is within run-to-run
noise) and is a large win on prefill — see §2.

## 2. Prefix caching — prefill only, but decisive

Same engine, shared prefix sent cold, then re-sent with a different suffix.
Time to first token, `max_tokens=1` to isolate prefill.

| Shared prefix | Cold | Warm | Speedup | Cache hits |
|---|---:|---:|---:|---:|
| 3.7 K tokens | 2.31 s | 2.49 s | — | 2,400 blocks |
| 19 K tokens | 12.64 s | 0.89 s | **14.2×** | 18,400 |
| 53 K tokens | 26.62 s | 1.21 s | **22.0×** | 52,000 |

At 3.7 K there is no visible gain because total latency is dominated by decoding the
output, not by prefill. The benefit scales with prefix length, which is exactly the
shape of an agent fleet sharing one long system prompt.

**Cost:** GPU KV cache falls from 348,497 to 271,315 tokens, i.e. concurrency at a *full*
262,144-token request drops from 1.33× to 1.03×. Worth it for shared-prefix workloads;
reconsider if you need several concurrent maximum-length requests.

**Correctness:** identical prompts returned byte-identical completions across cold and
warm runs, with correct answers.

## 3. Speculative decoding sweep — dedicated GPU

Sole occupant, `gpu-memory-utilization 0.85`, `max-model-len 262144`.

| Config | Fresh gen | Edit-heavy | Accept (edit) | Mean tok/pass | KV tokens |
|---|---:|---:|---:|---:|---:|
| MTP `k=3` | 17.70 | 20.52 / 21.30 | 99.9 % | 4.00 | 959,920 |
| MTP `k=8` | 18.64 | 31.34 / 32.23 | 99.3 % | 8.95 | 958,416 |
| MTP `k=15` | 13.39 | 37.50 / 39.00 | 98.8 % | 15.82 | 885,382 |
| **DSpark `k=7`** | **20.05** | **44.77 / 46.84** | 98.6 % | 7.91 | 635,895 |
| DSpark `k=14` | 18.77 | **52.86 / 58.49** | 67.6 % | 10.46 | 579,525 |

Fresh-generation acceptance for the same rows: 79.0 %, 50.3 %, 28.9 %, 31.7 %, 15.8 %.

The DSpark `k=7` row was measured twice, in separate containers on different days:
20.00 / 45.02 / 47.10 and 20.05 / 44.77 / 46.84 — agreement within 0.6 %.

`k=14` is the fastest single-stream configuration on FP8 (**58.49**, +25 % over `k=7`) but
see §3b before adopting it: it behaves very badly under concurrency on these weights.

Three things worth reading off this table:

1. **`k=15` is a trap for mixed workloads.** It is the best MTP row on edit work and the
   *worst* row overall on fresh generation — worse than `k=3`. Acceptance collapses to
   28.9 %, so most drafting is discarded.
2. **`k=8` is the best MTP compromise** — it beats `k=3` on both workloads.
3. **DSpark `k=7` beats every MTP row on both workloads simultaneously.**

### Why DSpark wins

Working the drafting cost backwards from measured pass rates (base decode = 7.88
forward passes/sec):

| | Throughput | Mean tok/pass | Passes/sec | Cost per draft token |
|---|---:|---:|---:|---:|
| MTP `k=8` | 32.2 | 8.95 | 3.60 | **0.153** |
| DSpark `k=7` | 47.1 | 7.91 | 5.95 | **0.046** |

DSpark accepts *fewer* tokens per pass and is still 46 % faster, because each pass costs
far less. MTP re-runs its single-layer head sequentially and pays a full `lm_head`
projection over the ~150K vocabulary per draft token; the DSpark drafter emits its whole
7-token block in one shot.

That 0.046 coefficient implies the asymptotic ceiling is roughly 21× base rather than the
~6.5× implied by MTP's — the limit is drafting cost and acceptance, not memory bandwidth.

## 3b. Concurrency — and why `k=14` is dangerous on FP8

Aggregate throughput, N simultaneous edit-heavy requests, distinct prompts, 1,500 output
tokens each.

| Concurrency | DSpark `k=7` | DSpark `k=14` |
|---|---:|---:|
| c1 | 46.91 | (58.49 single-stream) |
| c4 | **134.21** | 126.65 |
| c8 | **208.71** | **119.74** |
| c16 | **256.08** | not measured |

**`k=14` loses 43 % of aggregate throughput at c8** (208.71 → 119.74). Note it also
*inverts*: `k=14` is slower at c8 than at c4, i.e. adding concurrency made it worse.

This is a much harsher penalty than the same change causes on 4-bit weights, where `k=14`
cost only ~5 % at c8. Two contributing factors: FP8's larger weights leave less KV cache
(579,525 at `k=14` vs 635,895 at `k=7`, versus 1.3 M on NVFP4), and FP8 decode is more
bandwidth-bound, so speculative work competes harder with batch capacity.

**The interaction between draft depth and quantization is real, and it does not transfer.**
A `k` tuned on one set of weights should be re-measured on another.

| Goal on FP8 | Configuration | Result |
|---|---|---:|
| Single-stream latency | DSpark `k=14` | **58.49 tok/s** |
| Fleet throughput | DSpark `k=7` | **256.08 tok/s aggregate at c16** |

## 3c. Prefill throughput

Unique content per measurement so prefix caching cannot serve it; `max_tokens=1` isolates
prefill. Two independent cold runs per cell.

| Prompt | Tokens | FP8 | NVFP4 (4-bit) | FP8 advantage |
|---|---:|---:|---:|---:|
| ~8K | 10,260 | 1,506.3 | **1,527.3** | −1.4 % |
| ~32K | 42,584 | **1,312.0** | 1,225.5 | +7.1 % |
| ~100K | 136,624 | **939.4** | 861.1 | +9.1 % |

**4-bit costs prefill, and the cost grows with prompt length** — because Marlin
dequantization is compute-bound work, and prefill is the compute-bound phase. But the
effect is modest (7–9 % at long context) and absent at 8K.

Run-to-run agreement was within 0.05 % on every cell, making this the most reproducible
measurement in the project.

## 3d. The quantization advantage disappears under concurrency

Same benchmark, FP8 versus 4-bit NVFP4, both DSpark `k=7`:

| Concurrency | FP8 | NVFP4 | 4-bit advantage |
|---|---:|---:|---:|
| c1 | 46.91 | 59.79 | **+27 %** |
| c4 | 134.21 | 161.36 | +20 % |
| c8 | 208.71 | 229.31 | +10 % |
| c16 | 256.08 | 256.47 | **+0.2 %** |

A clean monotonic decay to zero, and the mechanism is the same one that makes quantization
work at c1 running in reverse. Single-stream decode is memory-bandwidth-bound, so reading
fewer weight bytes is decisive. As the batch grows, one weight read serves many sequences,
the workload becomes compute-bound, and byte count stops mattering.

**Practical consequence: at high concurrency, 4-bit buys nothing on this hardware.** If you
serve a fleet at c16, FP8 matches it — and FP8 carries no quantization quality question at
all. Quantization here is a single-stream and low-concurrency optimisation.

## 4. Dedicating the GPU did **not** make it faster

| | Fresh gen |
|---|---:|
| Shared with 3 idle engines, `gmu 0.44` | 19.00 |
| Sole occupant, `gmu 0.85` | 17.70 |

Idle engines hold memory but consume no bandwidth, so freeing them buys no throughput —
the larger KV cache (959,920 vs 263,672 tokens) slightly *increases* bookkeeping. The
entire value of dedicating the device is that it **allows a higher k**: at `gmu 0.44`,
`k=8` refused to start, needing 18.67 GiB of KV against 17.77 GiB available.

## 5. Failure modes encountered

| Attempt | Outcome |
|---|---|
| MTP `k=8` at `gmu 0.44` | `ValueError`: needs 18.67 GiB KV, has 17.77. Raise `gpu-memory-utilization` or lower `max-model-len`. |
| DSpark `k=14`, default batch budget | `max_num_scheduled_tokens` computed to **−1280**. Fix: raise `--max-num-batched-tokens`. |
| Suffix decoding | `ImportError`: needs `arctic-inference==0.1.1`, which pins `vllm==0.10.1`. Unusable on 0.27.x. |

## 6. Interpreting these numbers against published figures

Comparisons only hold when the measurement conditions match. Two live examples:

- A widely-cited **38.28 tok/s** figure for this model on this device is **single-stream
  with 4-bit weights**. The 47.1 above is single-stream with **8-bit** weights.
- An **84.3 tok/s** figure is **8-way concurrent aggregate throughput**, not single-stream
  latency. Aggregate and per-stream numbers differ by several times on this hardware and
  are not interchangeable.

Always check concurrency, output length, cold vs warm cache, and whether speculative
decoding was enabled before comparing.
