# Sources

## DFlash 2

- **Announcement** — *DFlash 2: Keep Drafting Parallel*, Inco AI, 18 August 2026.
  <https://inco.ai/blog/dflash2/>
- **Drafter (safetensors)** — <https://huggingface.co/incoai/Qwen3.8-27B-DFlash2>
- **Drafter (GGUF)** — <https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF>
- **Mirror** — <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2>
- **Reference implementation** — <https://github.com/z-lab/dflash>
- **vLLM support** — PR #52816, *[Spec Decode] DFlash2: local convolution + candidate
  selector*. <https://github.com/vllm-project/vllm/pull/52816> (open at time of writing)
- **llama.cpp support** — PR #27342, *spec : add DFlash2 support (local convolution +
  candidate selector)*. <https://github.com/ggml-org/llama.cpp/pull/27342> (open)

Published figures cited for comparison (Inco AI, H200, SGLang, block size 8, temperature 1.0
with `xhigh` reasoning): acceptance length 4.80 mean for Qwen3.8-27B against MTP 4.28 and a
community DSpark drafter 3.62; throughput 2.7-3.4x autoregressive at concurrency 1.

## Models

- **Target** — <https://huggingface.co/Qwen/Qwen3.8-27B> and the `-FP8` variant
- **NVFP4** — <https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4>
- **int4 AutoRound** — <https://huggingface.co/Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound>
- **GGUF** — <https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF>
- **DSpark drafter used here** — <https://huggingface.co/Doopeworld/Qwen3.8-27B-DSpark-vLLM>
  (note: Inco AI benchmarked <https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark>, a
  different community checkpoint — see the caveat in DFLASH2.md)

## Engines

- vLLM 0.27.1 — <https://github.com/vllm-project/vllm>
- llama.cpp — <https://github.com/ggml-org/llama.cpp>

## Evaluation data

- GSM8K — <https://huggingface.co/datasets/openai/gsm8k> (test split, n=200, seeded sample)
- MMLU — <https://huggingface.co/datasets/cais/mmlu> (test, n=400 across 56 subjects)

## Prior art referenced by DFlash 2

- Canon Layers; Dynamic Short Convolutions; Convolution for Large Language Models —
  cited in the announcement as the basis for the two-tap dynamic depthwise convolution.
- Modal, *Speculation Is All You Need* — cited on speculative decoding for low-latency serving.
