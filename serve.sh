#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
  printf 'Usage: %s {512k|512k-throughput|1m|1m-throughput} /path/to/weights\n' "${0##*/}"
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  usage
  exit 0
fi
if [[ $# != 2 ]]; then
  usage >&2
  exit 2
fi

preset=$1
runtime_args=()
case "$preset" in
  512k)
    context=524288 batch_tokens=1024 sequences=2 cache_bytes=8200000000
    ;;
  512k-throughput)
    context=524288 batch_tokens=1024 sequences=32 cache_bytes=7400000000
    ;;
  1m)
    context=1048576 batch_tokens=512 sequences=1 cache_bytes=9800000000
    runtime_args+=(--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}')
    ;;
  1m-throughput)
    context=1048576 batch_tokens=512 sequences=32 cache_bytes=9600000000
    runtime_args+=(--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}')
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

model_files=(config.json tokenizer.json model.safetensors.index.json model-00001-of-00001.safetensors)
if [[ $context == 524288 ]]; then
  model_files+=(mtp.safetensors)
  runtime_args+=(--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}')
fi

if [[ ! -d $2 ]]; then
  printf 'Model directory not found: %s\n' "$2" >&2
  exit 2
fi
weights_dir=$(cd -- "$2" && pwd -P)
for file in "${model_files[@]}"; do
  if [[ ! -s $weights_dir/$file ]]; then
    printf 'Missing model file: %s/%s\n' "$weights_dir" "$file" >&2
    exit 2
  fi
done

gpu_args=(--gpus all)
if [[ ${INKLING_USE_CDI:-0} == 1 ]]; then
  gpu_args=(--device nvidia.com/gpu=all)
fi

exec docker run --rm "${gpu_args[@]}" --ipc=host \
  --publish 127.0.0.1:18753:8000 \
  --mount "type=bind,source=$weights_dir,target=/model,readonly" \
  inkling-2-mini-ciru /model \
  --host 0.0.0.0 --port 8000 --served-model-name main \
  --tensor-parallel-size 1 \
  --max-model-len "$context" --max-num-batched-tokens "$batch_tokens" \
  --max-num-seqs "$sequences" --kv-cache-memory-bytes "$cache_bytes" \
  --kv-cache-dtype fp8 --limit-mm-per-prompt '{"image":0}' \
  --enable-prefix-caching --async-scheduling \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  "${runtime_args[@]}"
