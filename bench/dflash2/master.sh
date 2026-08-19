#!/usr/bin/env bash
# ONE sequential driver for all remaining work. No inter-process gating.
cd /home/bakeer/dflash2-bench
mkdir -p logs results
INT4=Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound
NVFP4=unsloth/Qwen3.8-27B-NVFP4
FP8=Qwen/Qwen3.8-27B-FP8
DSP=Doopeworld/Qwen3.8-27B-DSpark-vLLM
DF2='{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
dspark() { echo "{\"method\":\"dspark\",\"model\":\"$DSP\",\"num_speculative_tokens\":$1,\"draft_sample_method\":\"probabilistic\"}"; }
mtp()    { echo "{\"method\":\"mtp\",\"num_speculative_tokens\":$1}"; }

# task <tag> <model> <spec> <suite: gen|edit|none> <quality: yes|no>
task() {
  local tag="$1" model="$2" spec="$3" suite="$4" qual="$5"
  echo; echo "################ $tag ################"; date -Is
  if ! ./serve_one.sh "$tag" "$model" "$spec"; then
    echo "[$tag] SERVE FAILED -- skipping"
    docker logs qwenbench 2>&1 | tail -200 > logs/$tag.serverlog 2>/dev/null
    return
  fi
  case "$suite" in
    gen)  python3 bench_dflash.py --tag "$tag" --reps 3 --max-tokens 512  --temp 0 --batch 8 || echo "[$tag] BENCH FAILED";;
    edit) python3 run_edit.py     --tag "$tag" --reps 3 --max-tokens 1200 --temp 0 --batch 8 || echo "[$tag] BENCH FAILED";;
  esac
  [ "$qual" = "yes" ] && { python3 quality.py --tag "$tag" --workers 8 || echo "[$tag] QUALITY FAILED"; }
  docker logs qwenbench 2>&1 | tail -400 > logs/$tag.serverlog
}

echo "======== PHASE 1: 4-bit AutoRound int4 (DFlash2-capable) ========"
task int4-nospec  "$INT4" none          gen yes
task int4-dflash2 "$INT4" "$DF2"        gen yes
task int4-dspark7 "$INT4" "$(dspark 7)" gen no

echo "======== PHASE 2: edit regime (high copy fraction) ========"
task edit-nvfp4-nospec   "$NVFP4" none            edit no
task edit-nvfp4-dspark14 "$NVFP4" "$(dspark 14)"  edit no
task edit-nvfp4-dspark7  "$NVFP4" "$(dspark 7)"   edit no
task edit-fp8-dflash2    "$FP8"   "$DF2"          edit no

echo "======== PHASE 3: complete matrices + quality reference ========"
task nvfp4-mtp3   "$NVFP4" "$(mtp 3)"  gen no
task int4-mtp3    "$INT4"  "$(mtp 3)"  gen no
task fp8-nospec-q "$FP8"   none        none yes
task fp8-dflash2-q "$FP8"  "$DF2"      none yes

echo; echo "######## ALL PHASES DONE ########"; date -Is
