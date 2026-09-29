# Benchmark environment source register

Checked 2026-09-28. These are primary vendor sources for the Blackwell container recipe. Statements about the image's *runtime contents* remain unverified until the remote import probe runs; documentation of a tag does not prove a successful pull on Brandon's host.

| Source | Evidence used |
|---|---|
| [NVIDIA RAPIDS platform support](https://docs.nvidia.com/datascience/platform-support/) | RAPIDS 26.08 Linux architectures, Python 3.11–3.14, CUDA 12.2–12.9 / 13.0–13.3 ranges, driver 535+ / 580+ generic columns, and Blackwell CC 100/120 listing. |
| [NVIDIA CUDA GPU compute capability](https://developer.nvidia.com/cuda/gpus) | RTX PRO 6000 Blackwell Server Edition is CC 12.0 (`sm_120`). |
| [NVIDIA RAPIDS HPC deployment](https://docs.nvidia.com/datascience/deployment/stable/hpc/) | Official example names `nvcr.io/nvidia/rapidsai/base:26.08-cuda13-py3.13`. This validates the written tag, not a registry digest or runtime pull. |
| [Official RAPIDS Docker Hub tag API](https://hub.docker.com/v2/repositories/rapidsai/base/tags/26.08-cuda13-py3.13) | Read-only GET on 2026-09-28 returned an active tag with linux/amd64 and linux/arm64 manifests. This corroborates the release tag; it is not the digest of the selected NGC URL. |
| [NVIDIA CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html) | CUDA 13.0 GA toolkit driver floor is 580.65.06; verify image-specific requirements at runtime. |
| [NVIDIA GPU Operator platform support](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/26.3/platform-support.html) | RTX PRO 6000 Blackwell Server Edition requires driver 575.57.08 or newer in the operator's platform note. |
| [NVIDIA CUDA 13.0 overview](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/) | CUDA Toolkit 12.8 first supported Blackwell; CUDA 13.0 supports RTX PRO Blackwell. |
| [RAPIDS support notice 59](https://docs.nvidia.com/datascience/notices/rsn0059/) | cuDF 26.08 requires pandas 3.0+. |
| [RAPIDS support notice 61](https://docs.nvidia.com/datascience/notices/rsn0061/) | RAPIDS 26.08 requires CuPy 14.0.1+ and NumPy 2. |
| [NVIDIA Container Toolkit Docker GPU selection](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html) | `--gpus device=...` can select a specific GPU index or UUID. |
| [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Host driver and NVIDIA container runtime/Docker configuration are host prerequisites. |
| [Akamai RTX PRO 6000 Blackwell onboarding](https://techdocs.akamai.com/cloud-computing/docs/nvidia-rtx-pro-6000-blackwell-gpu-onboarding) | Limited availability, existing instance onboarding, optional cloud-init CUDA 12.8 example, and owner/operator setup context. |

Pending evidence from the existing authorized GPU host: `uname -m`, `/etc/os-release`, `nvidia-smi` GPU/driver output, `docker version`, image digest or ID after pull, NumPy/pandas/CuPy/cuDF import versions, GPU preflight JSON, correctness results, and timed benchmark outputs. No GPU deployment or benchmark execution has occurred in this workspace.
