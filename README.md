# Qwen3.8-27B-FP8 on a single DGX Spark (GB10 / SM121)

A measured serving recipe for `Qwen/Qwen3.8-27B-FP8` on one DGX Spark, plus the
benchmark harness used to produce every number in [RESULTS.md](RESULTS.md).

**Headline: 7.88 → 58.5 tok/s single-stream, without touching the weights.**

All of the speedup in this repo comes from decode strategy — speculative decoding
and prefix caching — both of which are *output-preserving*. The model's answers do
not change. Only the number of tokens produced per forward pass does.

| Configuration | Fresh generation | Edit-heavy | c8 aggregate |
|---|---:|---:|---:|
| Stock (no speculation, no prefix caching) | 7.88 | 7.88 | — |
| MTP `k=3` | 17.70 | 21.3 | — |
| MTP `k=8` | 18.64 | 32.2 | — |
| MTP `k=15` | 13.39 | 39.0 | — |
| **DSpark `k=7`** | **20.05** | 46.8 | **208.7** |
| **DSpark `k=14`** | 18.77 | **58.5** | 119.7 |

**No single `k` wins both.** `k=14` is the fastest for one request and loses 43 % of
aggregate throughput at c8. Pick by whether you optimise latency or fleet throughput.

Tokens/sec, single stream, temperature 0, thinking disabled. Full method in
[RESULTS.md](RESULTS.md).

**This repository measures speed only** — no quality evaluation was performed. The stock
baseline was measured in a different device configuration from the tuned rows, each cell
is one or two runs, and the edit-heavy workload is synthetic and likely optimistic.
[LIMITATIONS.md](LIMITATIONS.md) states exactly what these numbers do and do not
establish; read it before quoting them.

**A finding worth knowing before you reach for 4-bit:** the advantage of quantization on
this hardware collapses as concurrency rises — +27 % at c1, +10 % at c8, and **+0.2 % at
c16**, where FP8 matches 4-bit exactly. If you serve a fleet, FP8 costs you nothing and
carries no quantization quality question. See [RESULTS §3d](RESULTS.md#3d-the-quantization-advantage-disappears-under-concurrency).

## DFlash 2 — a third drafter, and the fastest single-stream option

Since this repo was written, Inco AI released [DFlash 2](https://inco.ai/blog/dflash2/),
a drafter that predicts a whole block in one pass instead of one token at a time. Measured
on the same device with a separate harness:

| drafter | generative | edit-heavy | acceptance |
|---|---:|---:|---:|
| none | 7.94 | — | — |
| MTP `k=3` | 17.99 | — | 3.165 |
| DSpark `k=7` | 19.12 | — | 2.875 |
| **DFlash2 `k=7`** | **31.72** | **49.20** | **4.607** |

**3.99x over stock on generation, 1.66x over DSpark `k=7`.**

Two things make this repo's checkpoint newly important: **FP8 is the only build of
Qwen3.8-27B that can serve DFlash 2 under vLLM at all** — the drafter requires an
unquantized LM head, which both 4-bit checkpoints fail — and the advantage is
**single-stream only**. At c2 and above, 4-bit weights with free in-checkpoint MTP
overtake it, by +23 % at c16.

Full method, the c1-c16 curve, the LM-head constraint and its pre-flight check:
**[DFLASH2.md](DFLASH2.md)**. References: [SOURCES.md](SOURCES.md).

## Hardware and software

| | |
|---|---|
| Device | DGX Spark, GB10 Grace Blackwell, compute capability **SM121** |
| Memory | 128 GB unified (≈121 GiB usable), **273 GB/s** |
| Engine | vLLM `v0.27.1-aarch64` (official image) |
| Target | `Qwen/Qwen3.8-27B-FP8` — 28.5 GiB weights, dense, hybrid attention |
| Drafter | `Doopeworld/Qwen3.8-27B-DSpark-vLLM` — 5 layers, 2.6 GB, `block_size 7` |

Qwen3.8-27B is a **dense** 27B model with **hybrid attention**: 48 GatedDeltaNet
(linear-attention) layers and 16 full-attention layers. That hybrid property drives
two of the most important findings below.

## Quickstart

```bash
./serve.sh            # starts the container, waits for /health
python bench/edit_bench.py     # single-stream: fresh generation + edit-heavy
python bench/conc_bench.py     # concurrency sweep, distinct prompts (--levels=1,4,8,16)
python bench/prefill_bench.py  # prefill throughput at ~8K / ~32K / ~100K
```

## The recipe, and why each flag is there

See [serve.sh](serve.sh) for the runnable version.

```bash
docker run -d --name qwen38 --gpus all --ipc host -p 127.0.0.1:8002:8002 \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$VLLM_CACHE":/root/.cache/vllm \
  --entrypoint vllm vllm/vllm-openai:v0.27.1-aarch64 \
  serve Qwen/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8002 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --speculative-config '{"method":"dspark","model":"Doopeworld/Qwen3.8-27B-DSpark-vLLM","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

**`--enable-prefix-caching` is not optional, and not the default.** vLLM computes
`default_prefix_caching = is_prefix_caching_supported and not is_hybrid`. Qwen3.8-27B
reports `is_hybrid=True`, so prefix caching is **off unless you ask for it** — the
comment in `engine/arg_utils.py` calls it "opt-in for now while the feature matures".
It is supported, it is correct in our testing, and on shared prefixes it is worth
**14–22×** on prefill. See [NOTES.md](NOTES.md#1-prefix-caching-is-off-by-default-on-hybrid-models).

**DSpark over MTP.** Both are speculative decoders. MTP re-runs a single in-checkpoint
head *sequentially* k times and pays a full `lm_head` projection over the ~150K vocab
each time. DSpark is a separate 5-layer drafter that emits a **block of 7 at once**.
Measured cost per draft token: **MTP 0.153, DSpark 0.046** — DSpark drafts 3.3× cheaper,
and wins on both workloads.

**`--reasoning-parser qwen3` / `--tool-call-parser qwen3_xml`.** These names are not
interchangeable: `qwen3_xml` does *not* resolve as a reasoning parser and `qwen3` does
*not* resolve as a tool-call parser. Registries populate lazily, so listing keys reads
empty — call the getter and see whether it raises. A typo costs a full model load.

**No `--enforce-eager`.** It is reported to cost ~55 % throughput on SM121.

## Choosing k

`k` (`num_speculative_tokens`) is how many tokens the drafter proposes before the
target model verifies them in one pass. It trades drafting work against how often that
work is thrown away, so **the right k depends on how predictable your output is**:

- **Editing / refactoring** — the answer is largely already in the prompt. Acceptance
  measured at 98.6–99.9 %. High k pays enormously.
- **Fresh generation** — the drafter cannot know what is coming. MTP acceptance falls
  to 28.9 % at `k=15`, and throughput drops *below* `k=3`.

There is a second axis that matters just as much, and it is easy to miss because it only
appears once you test under load: **concurrency**. Deeper drafting consumes scheduling
capacity, so a `k` that is optimal for one request can be badly wrong for a fleet.

| | DSpark `k=7` | DSpark `k=14` |
|---|---:|---:|
| Single stream, edit-heavy | 46.8 | **58.5** |
| c8 aggregate | **208.7** | 119.7 |

**Choose `k=14` for interactive latency, `k=7` for throughput.** On FP8 the penalty for
getting this wrong is severe — `k=14` gives up 43 % of aggregate throughput at c8, and is
slower at c8 than it is at c4.

That penalty is specific to these weights: the same `k=14` on 4-bit weights costs only
~5 % at c8, because those have twice the KV cache and less bandwidth pressure. **A `k`
tuned on one quantization does not transfer to another** — re-measure it.

## Reproducing

Every number in [RESULTS.md](RESULTS.md) comes from the three scripts in `bench/`,
run against the live server. All use `temperature=0` and
`chat_template_kwargs={"enable_thinking": false}`.

Thinking is on by default at high reasoning effort on this model, and left unbounded it
will happily produce tens of thousands of reasoning tokens. Always cap `max_tokens` and
set the thinking flag explicitly when benchmarking, or you are measuring something else.

## Related

The 4-bit companion recipe — same harness, same hardware, quantized weights — lives in
the sibling directory. The two sets of measurements are directly comparable, and the
cross-cutting result is that **quantization's benefit is concentrated at low concurrency**:
+27 % at c1, +0.2 % at c16.

## License

MIT for the scripts and documentation in this repository. Model weights remain under
their upstream licenses.
