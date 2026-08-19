#!/usr/bin/env bash
# Serve ONE Qwen3.8-27B config on :8002, dedicated, and health-gate it.
#   serve_one.sh <tag> <model-repo> <spec-json|none>
# Every flag except the weights and the speculative config is held constant.
set -uo pipefail
TAG="$1"; MODEL="$2"; SPEC="${3:-none}"
VLLM=vllm/vllm-openai:v0.27.1-aarch64
HF=/home/bakeer/models/hf
VC=/home/bakeer/models/vllm-cache
source /home/bakeer/dflash2-patch/mounts.sh   # -> $D2MOUNTS

docker rm -f qwenbench >/dev/null 2>&1 || true
sleep 8

COMMON=(--served-model-name qwen3.8-27b --host 0.0.0.0 --port 8002
        --max-model-len 131072 --gpu-memory-utilization 0.80
        --max-num-batched-tokens 16384
        --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0
        --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
        --enable-prefix-caching)

if [ "$SPEC" = "none" ]; then
  docker run -d --name qwenbench --gpus all --ipc host -p 127.0.0.1:8002:8002 \
    -v $HF:/root/.cache/huggingface -v $VC:/root/.cache/vllm $D2MOUNTS \
    --entrypoint vllm "$VLLM" serve "$MODEL" "${COMMON[@]}" >/dev/null
else
  docker run -d --name qwenbench --gpus all --ipc host -p 127.0.0.1:8002:8002 \
    -v $HF:/root/.cache/huggingface -v $VC:/root/.cache/vllm $D2MOUNTS \
    --entrypoint vllm "$VLLM" serve "$MODEL" "${COMMON[@]}" \
    --speculative-config "$SPEC" >/dev/null
fi

echo "[$TAG] starting ($MODEL / spec=$SPEC)"
for i in $(seq 1 80); do
  if curl -sf --max-time 3 http://127.0.0.1:8002/health >/dev/null 2>&1; then
    echo "[$TAG] healthy after ~$((i*15))s"
    docker logs qwenbench 2>&1 | grep -E "DFLASH2 SPECULATOR|GPU KV cache size|Model loading took|Available KV cache memory|graph capturing finished" | sed "s/^/    /"
    free -g | sed -n 2p | awk '{print "    mem used/avail GiB: "$3" / "$7}'
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^qwenbench$'; then
    echo "[$TAG] CONTAINER DIED"; docker logs qwenbench 2>&1 | tail -40; exit 1
  fi
  sleep 15
done
echo "[$TAG] TIMEOUT"; docker logs qwenbench 2>&1 | tail -30; exit 1
