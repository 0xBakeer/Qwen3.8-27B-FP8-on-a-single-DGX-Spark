#!/usr/bin/env bash
# Run the ORIGINAL repo harness (edit_bench / conc_bench / prefill_bench) against
# DFlash2 and, as controls, the two DSpark k=14 configs whose published numbers
# were 58.5 (FP8) and 75.0 (NVFP4) single-stream, 256 aggregate at c16.
set -uo pipefail
cd /home/bakeer/orig-harness
export HF_CACHE=/home/bakeer/models/hf VLLM_CACHE=/home/bakeer/models/vllm-cache
export D2_PATCH=/home/bakeer/dflash2-patch/patched/vllm
mkdir -p out

echo "=== tearing down stack ==="; date -Is
for c in nemotron35-nvfp4 vllm-embed qwen38fp8 vllm-rerank qwen38 qwen38-4bit; do docker rm -f $c >/dev/null 2>&1; done
sleep 10; free -g | sed -n 2p

one() { # tag MODEL SPEC K
  local tag="$1"
  echo; echo "################ $tag ################"; date -Is
  docker rm -f qwen38 >/dev/null 2>&1; sleep 5
  if MODEL="$2" SPEC="$3" K="$4" NAME=qwen38 ./serve.sh; then
    echo "--- edit_bench ($tag) ---"
    python3 edit_bench.py 2>&1 | tee out/$tag.edit.txt
    echo "--- conc_bench ($tag) ---"
    python3 conc_bench.py http://127.0.0.1:8002/v1 qwen3.8-27b --levels=1,4,8,16 2>&1 | tee out/$tag.conc.txt
    echo "--- prefill_bench ($tag) ---"
    python3 prefill_bench.py 2>&1 | tee out/$tag.prefill.txt
  else
    echo "[$tag] SERVE FAILED"
  fi
}

one fp8-dflash2-k7   Qwen/Qwen3.8-27B-FP8        dflash 7
one fp8-dspark-k14   Qwen/Qwen3.8-27B-FP8        dspark 14
one nvfp4-dspark-k14 unsloth/Qwen3.8-27B-NVFP4   dspark 14
docker rm -f qwen38 >/dev/null 2>&1; sleep 8

echo; echo "=== restoring production stack ==="; date -Is
PROFILE=b4 /home/bakeer/serve-stack2.sh
echo; echo "######## ORIG HARNESS DONE ########"; date -Is
