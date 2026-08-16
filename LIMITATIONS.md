# Limitations and methodology caveats

Everything in [RESULTS.md](RESULTS.md) is a real measurement taken on real hardware. This
document records what those measurements do **not** establish.

## 1. The stock baseline is not a controlled comparison

The `7.88 tok/s` figure that every multiplier is computed against was measured with the
model **sharing the device with three other (idle) engines at
`gpu-memory-utilization 0.44`**. The 58.5 tok/s headline was measured with the model as
**sole occupant at `gpu-memory-utilization 0.85`**.

A stock, no-speculation baseline was never re-measured in the dedicated configuration.

Evidence suggests this does not inflate the result: the same MTP `k=3` configuration
measured 19.00 shared vs 17.70 dedicated — dedicating the device was slightly *slower*.
So a dedicated stock baseline would likely be ≤ 7.88 and the true multiplier equal or
larger. But it was not measured, and "7.4×" therefore spans a configuration change it
does not advertise on its face.

## 2. Sample sizes are small and there are no error bars

Each cell is **one or two runs**. No repetitions beyond that, no variance analysis, no
confidence intervals. Observed run-to-run spread was ~1–5 % on stable configurations.
Treat differences below ~5 % as noise.

## 3. Derived quantities are derived, not measured

The cost-per-draft-token figures (MTP **0.153**, DSpark **0.046**) are arithmetic on
measured throughput and mean tokens-per-pass, assuming a base rate of 7.88 forward
passes/sec. The drafter was not instrumented directly.

The "asymptotic ceiling ≈ 21× base" in RESULTS §3 is therefore **a model extrapolated
from one data point, not data.** An earlier extrapolation of the same kind — built on the
MTP coefficient — underestimated what deeper drafting could achieve. Treat all such
projections as hypotheses that happen to be cheap to test.

## 4. The edit-heavy workload is synthetic and probably optimistic

The benchmark asks the model to reproduce a file of 45 near-identical dataclasses with one
method added. That is unusually repetitive, so drafter acceptance (98–99 %) sits at the
favourable end of what real work produces.

Expect real refactors — heterogeneous files, genuine logic changes, mixed prose and
code — to land **below** the edit-heavy column. Fresh generation is the pessimistic bound;
real workloads fall between the two columns rather than at the optimistic edge.

## 5. Speculative decoding preserves output — verified narrowly

The output-preservation property is guaranteed by the algorithm: the target model verifies
every drafted token and discards mismatches. This repo's empirical check was **identical
prompts returning byte-identical completions** across configurations at `temperature=0`.
That is consistent with the guarantee but is not an exhaustive proof, and it says nothing
about sampled (non-zero temperature) generation beyond the theory.

Two genuine behavioural changes, neither about quality: `min_p` and `logit_bias` are
silently ignored under speculative decoding, and KV cache capacity shrinks.

## 6. Prefix caching on hybrid models is upstream-flagged as maturing

vLLM's own comment describes prefix caching for hybrid architectures as "opt-in for now
while the feature matures". Our correctness check was one deterministic prompt comparison
plus a cache-hit count — not a proof of correctness across context lengths, eviction
pressure, or concurrency.

The 14–22× prefill speedups are solid; the correctness claim is narrower than the
performance claim.

## 7. Coverage gaps

Closed since the first revision: DSpark `k=14` on FP8, the concurrency curve c1–c16, the
prefill curve, and the KV cache figures.

Still open:

- **`k=14` concurrency was measured only at c4 and c8**, not c1 or c16. Its c8 collapse
  (119.74) was severe enough that extending the curve seemed less valuable than recording
  the failure clearly.
- **Only `k` values 3, 7, 8, 14, 15 were tested**, and only 7 and 14 for DSpark — chosen
  as the drafter's native `block_size` and 2× it, not found by search. The optimum for
  either objective may lie between them.
- **Concurrency beyond c16 was not tested.** Aggregate was still rising at c16, though
  per-stream had halved from c8.
- **Multimodal paths were not benchmarked**, despite the checkpoint shipping a vision
  tower.
- **Quality was not evaluated.** This repository measures throughput. The FP8 checkpoint
  is the vendor's own release and is generally regarded as near-lossless, but that claim
  is inherited, not tested here.
- **Fresh generation is the noisiest metric in this project.** Low draft acceptance means
  high variance; treat that column as approximate. Edit-heavy and prefill are far more
  reproducible — prefill reproduced to 0.05 %, and an independent re-run of DSpark `k=7`
  on a different day matched the original to 0.6 %.

## 8. Single hardware sample, single software version

One DGX Spark, vLLM `v0.27.1-aarch64`, one driver branch. Nothing has been reproduced on a
second device or engine version. Several gotchas in [NOTES.md](NOTES.md) are
version-specific — the prefix-caching default, the parser-name registry behaviour and the
deprecated `qwen3_5_mtp` alias could all change in either direction between releases.

Some SM121 recommendations reported by others — `--attention-backend TRITON_ATTN`,
avoiding `--enforce-eager`, staying on the 580.x driver — were **not** exercised here and
are repeated as reports, not as verified findings.
