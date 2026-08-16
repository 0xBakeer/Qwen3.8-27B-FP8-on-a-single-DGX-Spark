# Findings and gotchas

Things that cost time to discover, in rough order of how much they matter.

## 1. Prefix caching is off by default on hybrid models

vLLM decides the default in `engine/arg_utils.py`:

```python
# Hybrid models support prefix caching but keep it opt-in for now
# while the feature matures.
default_prefix_caching = (
    model_config.is_prefix_caching_supported and not model_config.is_hybrid
)
```

Qwen3.8-27B reports `is_hybrid=True` (48 GatedDeltaNet + 16 full-attention layers) and
`is_prefix_caching_supported=True`. So prefix caching is **supported but disabled unless
you pass `--enable-prefix-caching` explicitly**, and nothing in the startup output
announces it. Every request re-prefills its prompt from cold.

Worth **14–22×** on shared-prefix prefill. Costs ~22 % of KV cache capacity.

**This is not caused by speculative decoding.** There is no MTP/prefix-caching conflict
in vLLM 0.27.1 — the only two places that force `enable_prefix_caching = False` are
RISC-V CPUs and models with non-causal attention. `config/speculative.py` even scales
speculation length *from* prefix-match length, i.e. the features are designed to
cooperate. If you carry a note saying otherwise from an older version, re-check it.

## 2. Speculative decoding cannot force a token — but it does change the text

The drafter only *proposes*. The target model verifies every token and keeps a proposal
only if it matches what it would have generated itself; the first mismatch is discarded
and replaced. That guarantee holds, and it is why `k` cannot make the model dumber.

**But "cannot force a token" is not the same as "identical output", and an earlier
version of this file wrongly claimed the latter.** Measured on 20 prompts at
`temperature=0`:

| Comparison | Byte-identical |
|---|---|
| Same config, run twice | **20 / 20** |
| Tuned vs stock | **8 / 20** |

The engine is deterministic *within* a configuration. Across configurations the text
diverges, because verifying eight tokens per pass instead of one changes the batch
dimensions, which changes floating-point accumulation order, which flips the argmax
wherever two candidate tokens are near-tied. Greedy decoding then follows a different
branch for the rest of the completion.

The same applies to any change of batch geometry — concurrency, `max_num_batched_tokens`,
prefix caching, tensor-parallel degree. **Byte-reproducible output is not something a
serving configuration can promise**, with or without speculation. `VLLM_BATCH_INVARIANT`
exists if you need it, at a throughput cost.

Full data, including all twelve divergences, in [EVAL.md](EVAL.md). The divergences are
synonyms and formatting, not errors — but a diff cannot establish quality either way.
That needs a scored eval, which is not in this repository yet.

**Two real behavioural changes**, neither about quality:

- `min_p` and `logit_bias` are **silently ignored** under speculative decoding. The engine
  logs a warning at startup and then never mentions it again. Check your client presets.
- KV cache shrinks, so maximum concurrency at full context drops.

## 3. `k` must be matched to workload predictability

| Workload | Acceptance at k=15 | Result |
|---|---:|---|
| Edit / refactor existing code | 98.8 % | 39.0 tok/s — best MTP result |
| Fresh generation | 28.9 % | 13.4 tok/s — worse than k=3 |

High `k` on unpredictable output is actively harmful: you pay for drafting and throw most
of it away. If you serve both workloads from one endpoint, either pick DSpark `k=7`
(best on both) or route by request type.

## 4. Drafter architecture matters more than draft depth

MTP re-runs one in-checkpoint layer sequentially, paying a full `lm_head` projection over
the ~150K vocabulary for every draft token. DSpark is a separate 5-layer, 1B-parameter
drafter that emits a whole block at once.

Measured cost per draft token: **MTP 0.153**, **DSpark 0.046**.

DSpark takes fewer tokens per pass (7.91 vs 8.95) and is still 46 % faster.

## 5. Scheduling constraints bound `k`, twice

Both are configuration limits, not hardware ones, and both surface as startup failures
rather than degraded performance — which is the good outcome.

- **KV capacity.** `k=8` at `gpu-memory-utilization 0.44` needed 18.67 GiB of KV to hold
  one 262,144-token sequence and had 17.77 GiB. Raise `gpu-memory-utilization`, or lower
  `max-model-len`.
- **Batch budget.** DSpark `k=14` with the default token budget computed
  `max_num_scheduled_tokens = -1280`, because draft slots for `k × max_num_seqs` exceed
  the default 2,048-token batch. Raise `--max-num-batched-tokens`.

## 6. Parser names do not follow a pattern

`--reasoning-parser qwen3` resolves; `qwen3_xml` does not.
`--tool-call-parser qwen3_xml` resolves; `qwen3` does not.

Registries populate lazily, so enumerating `.keys()` returns empty and tells you nothing.
Call `get_*_parser(name)` in a throwaway container and see whether it raises. A wrong name
costs a full model load before it fails.

Also: `qwen3_5_mtp` still resolves but logs *"deprecated and replaced with mtp"*. Use `mtp`.

## 7. Thinking is on by default, at high effort

Left uncapped this model will generate tens of thousands of reasoning tokens for a single
request. `--reasoning-parser` does not control effort. Send
`chat_template_kwargs: {"reasoning_effort": "low"}` or `{"enable_thinking": false}`, and
always set `max_tokens`. `preserve_thinking` also defaults on, so prior turns' reasoning
accumulates in multi-turn conversations.

Benchmarks that omit this are not measuring decode throughput.

## 8. Adaptive verification does not work with this model

vLLM shipped adaptive verification in [PR #47808](https://github.com/vllm-project/vllm/pull/47808)
(on `main`, tested here at `0.27.2rc1.dev122`). It varies the number of verified tokens per step from a learned
confidence head, behaving like a long draft block at low concurrency and a short one at
high concurrency — which is exactly the tradeoff §3 says you otherwise have to choose
manually.

**It cannot be used here:**

```
ValueError: Adaptive verification trims verification requests on device, which the
GDNAttentionBackend attention backend does not support. Pass
enable_adaptive_verification=false in the speculative config, or use a backend that does.
```

Qwen3.8-27B is hybrid — 48 GatedDeltaNet layers plus 16 full-attention — and those GDN
layers require the GDN backend, so there is no backend to switch to. **The restriction is
architectural, not hardware-specific:** this fails identically on any GPU. The published
benchmarks for the feature use a model with MLA attention, which does support it.

Two things worth knowing about the failure mode: it raises at KV-cache init, roughly six
minutes into startup, so it is not free to test; and it fails **loudly** rather than
silently falling back, which is the good outcome — a quiet fallback would look like
"adaptive didn't help" rather than "adaptive never ran".

Consequence: **`k` still has to be chosen by hand on this model**, per §3.

## 9. Suffix decoding is unavailable on modern vLLM

`suffix` appears in vLLM's speculative-method registry, but the implementation is gated
behind [`arctic-inference==0.1.1`](https://pypi.org/project/arctic-inference/0.1.1/), which
ships source-only and pins `vllm==0.10.1`. On
0.27.x it fails at config validation. It monkey-patches engine internals, so forcing the
install is not advisable.

## 10. SM121 quantization caveats

Relevant when moving to 4-bit weights; documented in full in the companion 4-bit recipe.

- GB10 exposes **no native FP4 compute path**, so `W4A4` gains nothing over `W4A16` and
  only costs accuracy. Prefer weight-only quantization.
- CUTLASS FP4 kernels are reported to produce silent garbage on this architecture.
- `VLLM_MARLIN_USE_ATOMIC_ADD=1` fixes a Marlin race condition that yields incorrect output.

## 11. Idle engines cost memory, not bandwidth

Co-resident but idle vLLM engines did not measurably slow decoding. Freeing them raised no
throughput; it only enabled a larger `gpu-memory-utilization` and therefore a higher `k`.
Plan capacity around memory, and expect contention only when other engines are actively
decoding.
