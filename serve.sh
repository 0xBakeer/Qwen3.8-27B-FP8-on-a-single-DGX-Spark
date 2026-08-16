#!/usr/bin/env bash
# Serve Qwen3.8-27B-FP8 on a single DGX Spark (GB10 / SM121) with vLLM.
#
# Defaults are the best measured configuration: DSpark k=7 speculative decoding
# plus prefix caching. See RESULTS.md for the numbers behind these choices.
#
#   SPEC=dspark  (default)  external 5-layer drafter, block_size 7 - best on both
#                           fresh generation and edit-heavy work
#   SPEC=mtp                in-checkpoint head; set K=8 for the best mixed-workload
#                           MTP setting, K=15 only if you serve edit work exclusively
#   SPEC=off                no speculative decoding
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
DRAFTER="${DRAFTER:-Doopeworld/Qwen3.8-27B-DSpark-vLLM}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.27.1-aarch64}"
NAME="${NAME:-qwen38}"
PORT="${PORT:-8002}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b}"

HF_CACHE="${HF_CACHE:-$HOME/models/hf}"
VLLM_CACHE="${VLLM_CACHE:-$HOME/models/vllm-cache}"

# 0.85 assumes this model is the sole occupant of the device. Lower it if you are
# co-hosting other engines - but note that k is bounded by KV capacity: k=8 at 0.44
# could not hold a single full-length sequence.
GMU="${GMU:-0.85}"
MAX_LEN="${MAX_LEN:-262144}"

# Raise this when increasing k. Draft slots are taken from the batch token budget, and
# k * max_num_seqs exceeding it makes max_num_scheduled_tokens go negative at startup.
MAX_BATCHED="${MAX_BATCHED:-16384}"

SPEC="${SPEC:-dspark}"
K="${K:-7}"

case "$SPEC" in
  dspark) SPEC_CFG="{\"method\":\"dspark\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"probabilistic\"}" ;;
  mtp)    SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$K}" ;;
  off)    SPEC_CFG="" ;;
  *)      echo "SPEC must be one of: dspark, mtp, off" >&2; exit 2 ;;
esac

mkdir -p "$HF_CACHE" "$VLLM_CACHE"
docker rm -f "$NAME" >/dev/null 2>&1 || true

ARGS=(
  serve "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0 --port "$PORT"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GMU"
  --max-num-batched-tokens "$MAX_BATCHED"
  # Not the default on this model: it reports is_hybrid=True, and vLLM keeps prefix
  # caching opt-in for hybrid architectures. Worth 14-22x on shared-prefix prefill.
  --enable-prefix-caching
  # These two parser names are not interchangeable - see NOTES.md section 6.
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --limit-mm-per-prompt.image 2
  --limit-mm-per-prompt.video 0
)
[ -n "$SPEC_CFG" ] && ARGS+=( --speculative-config "$SPEC_CFG" )

echo "starting $NAME :: $MODEL :: spec=$SPEC k=$K gmu=$GMU"

docker run -d --name "$NAME" --gpus all --ipc host \
  -p "127.0.0.1:$PORT:$PORT" \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$VLLM_CACHE":/root/.cache/vllm \
  --entrypoint vllm "$IMAGE" "${ARGS[@]}" >/dev/null

# Cold start is several minutes: weights, optional drafter, torch.compile and kernel
# autotune. Failures are usually config errors that surface early - watch for them.
echo -n "waiting for readiness"
for _ in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo; echo "ready on http://127.0.0.1:$PORT/v1  (model: $SERVED_NAME)"
    docker logs "$NAME" 2>&1 | grep -o "GPU KV cache size: [0-9,]*" | tail -1
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo; echo "container exited. last errors:" >&2
    docker logs "$NAME" 2>&1 | grep -iE "ValueError|Value error|ImportError|Error:" | tail -5 >&2
    exit 1
  fi
  echo -n "."
  sleep 5
done

echo; echo "timed out waiting for readiness" >&2
exit 1
