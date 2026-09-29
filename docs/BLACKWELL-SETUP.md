# RTX PRO 6000 Blackwell benchmark setup

This is an operator recipe for an **existing, authorized** Akamai Linux instance with an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU. It does not provision an instance or alter its host driver. The benchmark has not yet been run on that GPU; save the preflight output and results from the actual host before reporting measurements. Use the benchmark repository root as the working directory. Commands in this page are for that remote Linux host, not the course authoring Mac.

## Compatibility decision

The RTX PRO 6000 Blackwell Server Edition is CUDA compute capability **12.0** (`sm_120`). RAPIDS **26.08** lists Blackwell 120 in its supported compute capabilities and supports Python 3.11–3.14 on Linux x86_64/aarch64. The selected NVIDIA RAPIDS base image is `nvcr.io/nvidia/rapidsai/base:26.08-cuda13-py3.13`, a tag shown in NVIDIA's own deployment documentation. It fixes the RAPIDS release, CUDA major version, and Python minor version. NVIDIA's 26.08 compatibility table gives **580+** for CUDA 13; the exact image runtime still must pass the container and benchmark preflight on the host. CUDA 13.0 GA's toolkit driver floor is **580.65.06**. For this recipe, arrange a current compatible 580-series or newer host driver with the Akamai instance owner if the host is below that floor. The driver remains a host responsibility; the image supplies CUDA userspace. [RAPIDS platform support](https://docs.nvidia.com/datascience/platform-support/), [NVIDIA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus), [RAPIDS 26.08 base image example](https://docs.nvidia.com/datascience/deployment/stable/hpc/), [CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

The broad RAPIDS table also lists CUDA 12.2 and driver 535 as *generic* CUDA 12 lower bounds. Those numbers alone do **not** establish support for this `sm_120` GPU. NVIDIA's GPU Operator requires at least driver **575.57.08** for the RTX PRO 6000 Blackwell Server Edition, and NVIDIA identifies CUDA **12.8** as the first toolkit with Blackwell support. Akamai's optional GPU `cloud-config` example installs CUDA 12.8; inspect the resulting **driver version** before choosing this CUDA 13 image. Do not infer the driver from the host toolkit or from the `CUDA Version` maximum shown by `nvidia-smi`. [GPU Operator platform support](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/26.3/platform-support.html), [NVIDIA CUDA 13.0 overview](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/), [Akamai onboarding](https://techdocs.akamai.com/cloud-computing/docs/nvidia-rtx-pro-6000-blackwell-gpu-onboarding).

RAPIDS 26.08 requires **pandas 3.0+**, **CuPy 14.0.1+**, and **NumPy 2**. The image is expected to supply cuDF and its compatible dependencies; verify all four imports and versions in the pulled image before running the benchmark. Keep the CPU baseline in the same image so Python and package versions match. [RAPIDS pandas notice](https://docs.nvidia.com/datascience/notices/rsn0059/), [RAPIDS CuPy/NumPy notice](https://docs.nvidia.com/datascience/notices/rsn0061/).

## Inspect the existing host

Run these read-only commands on the GPU host. Record their output with the benchmark report, omitting unrelated host details if the report is shared.

```bash
uname -m
cat /etc/os-release
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total --format=csv
docker version
```

Expect Linux x86_64 or aarch64, the named Blackwell GPU with compute capability 12.0, a compatible host driver, and Docker with NVIDIA Container Toolkit configured. If `nvidia-smi` fails or Docker cannot expose a GPU, ask the instance owner to repair the host driver/container runtime using [NVIDIA's installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). This recipe performs no host package installation. Akamai says the Blackwell plan is of limited availability; existing account access does not itself imply that a compatible instance or driver is ready. [Akamai onboarding](https://techdocs.akamai.com/cloud-computing/docs/nvidia-rtx-pro-6000-blackwell-gpu-onboarding).

## Pull and inspect the fixed image

From the cloned benchmark repository root on the remote host:

```bash
IMAGE=nvcr.io/nvidia/rapidsai/base:26.08-cuda13-py3.13
docker pull "$IMAGE"
docker image inspect --format '{{json .RepoDigests}}' "$IMAGE"
docker run --rm --gpus 'device=0' "$IMAGE" python -c 'import sys, numpy, pandas, cupy, cudf; print("Python", sys.version); print("NumPy", numpy.__version__); print("pandas", pandas.__version__); print("CuPy", cupy.__version__); print("cuDF", cudf.__version__); print("GPU", cupy.cuda.runtime.getDeviceProperties(0)["name"])'
```

Select a different device index or GPU UUID if GPU 0 is occupied. The `--gpus device=...` selection exposes only that device to the container; no multi-GPU framework is needed. NVIDIA documents this NGC image tag, and [the official RAPIDS Docker Hub tag API](https://hub.docker.com/v2/repositories/rapidsai/base/tags/26.08-cuda13-py3.13) returned an active multi-architecture tag on 2026-09-28. **The NGC registry digest and actual package contents have not been verified in this workspace.** Record the digest returned by the remote NGC pull, or the local image ID if `RepoDigests` is empty. If the import probe fails, stop and resolve the image/package mismatch before running or publishing numbers. Do not silently replace the image with `latest`. [NVIDIA Container Toolkit device selection](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).

## Run from this repository

`environments/run-in-container.sh` mounts only the benchmark repository at `/workspace`, runs as the invoking user's UID and GID so `results/` stays user-owned, and selects one GPU for GPU commands. It does not use privileged mode, mount host driver directories, or install packages on the host. It inspects the selected local image, records its ID and available repository digests, and runs that exact image ID. The default is the fixed release tag above; `ADS_IMAGE` may select a previously captured NGC digest for a repeat run. The host's NVIDIA Container Toolkit passes its driver and selected device into the container. Use a private, idle or explicitly shared GPU slot and record other GPU activity when interpreting timings.

```bash
cd /path/to/nca-ads-benchmarks
mkdir -p results
chmod +x environments/run-in-container.sh
ADS_GPU_ID=0 ./environments/run-in-container.sh preflight --backend gpu --output results/preflight.json
ADS_GPU_ID=0 ./environments/run-in-container.sh run --backend gpu --rows 10000 100000 --repeat 5 --warmup 2 --seed 20260928 --output results
```

The GPU run includes the CPU baseline in the same software environment. The benchmark report must keep **GPU resident compute** and **host-to-host including transfer** timing separate, compare output correctness against the CPU path, and retain hardware/software versions, seed, row counts, warmup and repeat counts. These CLI paths are implemented and CPU-tested locally. Actual GPU output remains unverified until the remote preflight and run pass. For a CPU-only check in the same image, use:

```bash
mkdir -p results/cpu
./environments/run-in-container.sh preflight --backend cpu --output results/preflight-cpu.json
./environments/run-in-container.sh run --backend cpu --rows 10000 100000 --repeat 5 --warmup 2 --seed 20260928 --output results/cpu
```

The CPU commands deliberately request no GPU. A successful CPU run establishes only the baseline and code path, not RAPIDS execution on Blackwell. Preserve results under distinct names or directories when comparing runs. Do not present anticipated speedups as measured data.

## Troubleshooting boundaries

- `nvidia-smi` fails on host: host driver/device issue; contact the instance owner. Do not attempt an in-container fix.
- Docker reports `could not select device driver` or no GPU devices: the NVIDIA Container Toolkit/Docker integration needs owner attention. Check the [toolkit guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- Container rejects the driver or reports `no kernel image` / unsupported architecture: record host driver, image digest, GPU model, and package versions. Recheck the CUDA 13 driver requirement and `sm_120` support; do not claim the generic CUDA 12.2 floor is sufficient.
- Import probe succeeds but GPU preflight fails: preserve the preflight JSON/error and stop before benchmark timings.

Source details and retrieval date are in [SOURCES.md](SOURCES.md).
