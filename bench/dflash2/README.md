# DFlash2 benchmark harness

The suite that produced every number in [../../DFLASH2.md](../../DFLASH2.md). It is kept
separate from the sibling `bench/` scripts because it uses **different prompts and output
lengths** — the two are internally consistent but not cell-for-cell comparable. See the method
note in DFLASH2.md.

Raw results for every configuration are committed in `results/`, so the tables can be
regenerated and re-checked without re-running anything:

```bash
python3 analyze.py results/     # comparison tables + losslessness diff
python3 matrix.py               # precision x method matrix
```

## Re-running a single configuration

```bash
./serve_one.sh <tag> <model-repo> '<spec-json>|none'          # serve + health-gate on :8002
python3 bench_dflash.py --tag <tag> --reps 3 --max-tokens 512  --temp 0 --batch 8
python3 run_edit.py     --tag <tag> --reps 3 --max-tokens 1200 --temp 0 --batch 8
python3 sweep.py        --tag <tag> --levels 1,2,4,8,12,16 --max-tokens 384
python3 quality.py      --tag <tag> --workers 8
```

`serve_one.sh` holds every engine flag constant except the weights and
`--speculative-config`, which is what makes the configurations comparable:

```
--max-model-len 131072 --gpu-memory-utilization 0.80 --max-num-batched-tokens 16384
--enable-prefix-caching   (images/video disabled)
```

Engines must be started **sequentially** — a vLLM engine profiles free memory at startup, and
one profiling while another still allocates counts the other's memory as its own and dies with
"No available memory for the cache blocks".

## Re-running everything

```bash
python3 prep_data.py     # extract GSM8K/MMLU subsets from the HF cache (needs pyarrow)
./master.sh              # full vLLM matrix, sequential, ~2.5 h
./sweep_leg.sh           # concurrency 1..16 (in the 4-bit repo)
```

`master.sh` tears the serving stack down and runs every configuration in sequence. Read it
before running it on a machine you care about.

The vLLM DFlash2 overlay lives in `vllm-patch/` — PR #52816 is pure Python, so
`apply.py` patches the 0.27.1 sources with asserted anchors rather than rebuilding the image.

## Metric definitions

- **decode tok/s** excludes TTFT: `completion_tokens / (wall - ttft)`. Reported single-stream
  figure is the mean of per-prompt medians over 3 repetitions of 4 prompts.
- **aggregate tok/s** (sweep) *includes* TTFT, so its c=1 point reads slightly below the
  decode figure. Prompts are assigned round-robin, so **c=1 and c=2 use an unbalanced prompt
  mix**; c>=4 is balanced and is the comparable part of the curve.
- **acceptance length** = `1 + accepted/drafts` from the engine's `spec_decode_*` counters,
  deltaed across the run — the same definition Inco AI use. The `+1` is the verifier's
  always-free token.
- **quality**: GSM8K n=200 exact-match on the final number, MMLU n=400 single letter across
  56 subjects, greedy, thinking disabled. `analyze.py` also reports per-item agreement between
  runs, which discriminates better than the aggregate score.

## Gotchas that cost time

- **Qwen3.8 defaults to thinking at `xhigh`.** Every request here sends
  `chat_template_kwargs: {"enable_thinking": false}`; llama-server needs `--reasoning-budget 0`.
  Without it token counts explode and the numbers mean nothing.
- **llama-server serialises requests unless you pass `-np N`.** A concurrency sweep without it
  measures queueing, not throughput.
- **DFlash2 can silently degrade to DFlash1** if the V2 speculator is not selected. Confirm at
  startup rather than trusting the config.
- **DFlash2 needs an unquantized LM head** — see DFLASH2.md. Check before planning a run.
