#!/usr/bin/env bash
# High-concurrency TensorRT batch inference on N GPUs: one process per GPU,
# questions sharded round-robin with intra-GPU multi-question batching, shards merged.
#
#   bash scripts/run_batch_inference.sh <num_gpus> --model <ckpt> --input <questions.jsonl> --output <out.jsonl> [args]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
N="${1:?number of GPUs}"; shift
OUTPUT=""
HAS_BACKEND=0
HAS_QBATCH=0
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --output) OUTPUT="${ARGS[$((i + 1))]}" ;;
    --backend) HAS_BACKEND=1 ;;
    --question_batch_size|--question-batch-size|--concurrency) HAS_QBATCH=1 ;;
  esac
done
[ -n "$OUTPUT" ] || { echo "--output is required"; exit 1; }
mkdir -p "$(dirname "$OUTPUT")"
EXTRA_DEFAULTS=()
if [ "$HAS_BACKEND" -eq 0 ]; then
  EXTRA_DEFAULTS+=("--backend" "${BACKEND:-tensorrt}")
fi
if [ "$HAS_QBATCH" -eq 0 ]; then
  EXTRA_DEFAULTS+=("--question_batch_size" "${QUESTION_BATCH_SIZE:-32}")
fi
PIDS=()
for ((g = 0; g < N; g++)); do
  SHARD_ARGS=()
  for ((i = 0; i < ${#ARGS[@]}; i++)); do
    if [ "${ARGS[$i]}" = "--output" ]; then
      SHARD_ARGS+=("--output" "$OUTPUT.shard$g")
      i=$((i + 1))
    else
      SHARD_ARGS+=("${ARGS[$i]}")
    fi
  done
  CUDA_VISIBLE_DEVICES="$g" python -m hybrid_inference.run_batch "${EXTRA_DEFAULTS[@]}" "${SHARD_ARGS[@]}" --shard_id "$g" --num_shards "$N" &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
cat "$OUTPUT".shard* > "$OUTPUT"
echo "merged $(wc -l < "$OUTPUT") records into $OUTPUT"
