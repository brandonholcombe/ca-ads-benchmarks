#!/usr/bin/env bash
# Run the benchmark in the documented RAPIDS 26.08 environment.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {preflight|run} --backend {cpu|gpu} [benchmark options]" >&2
  exit 2
fi

backend=""
backend_count=0
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --backend)
      if ((i + 1 >= ${#args[@]})); then
        echo "--backend needs cpu or gpu" >&2
        exit 2
      fi
      backend="${args[$((i + 1))]}"
      backend_count=$((backend_count + 1))
      ;;
    --backend=*)
      backend="${args[$i]#--backend=}"
      backend_count=$((backend_count + 1))
      ;;
  esac
done

if ((backend_count != 1)) || [[ "$backend" != cpu && "$backend" != gpu ]]; then
  echo "Pass exactly one --backend cpu or --backend gpu." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
image="${ADS_IMAGE:-nvcr.io/nvidia/rapidsai/base:26.08-cuda13-py3.13}"
if ! image_id="$(docker image inspect --format '{{.Id}}' "$image")"; then
  echo "Pull the documented image first: docker pull $image" >&2
  exit 1
fi
image_digests="$(docker image inspect --format '{{json .RepoDigests}}' "$image")"
docker_args=(
  run --rm
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  --env "ADS_CONTAINER_IMAGE=$image"
  --env "ADS_CONTAINER_IMAGE_ID=$image_id"
  --env "ADS_CONTAINER_REPODIGESTS=$image_digests"
  --mount "type=bind,source=$repo_root,target=/workspace"
  --workdir /workspace
)

if [[ "$backend" == gpu ]]; then
  gpu_id="${ADS_GPU_ID:-0}"
  if [[ ! "$gpu_id" =~ ^[0-9]+$ && ! "$gpu_id" =~ ^GPU-[[:xdigit:]-]+$ ]]; then
    echo "ADS_GPU_ID must be a nonnegative GPU index or NVIDIA GPU UUID." >&2
    exit 2
  fi
  docker_args+=(--gpus "device=$gpu_id")
fi

exec docker "${docker_args[@]}" "$image_id" python -m nca_ads_bench "$@"
