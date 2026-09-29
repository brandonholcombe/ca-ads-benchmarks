# NCA-ADS CPU/GPU benchmarks

Portable correctness checks and measured CPU-versus-GPU experiments for the NCA-ADS course.
The first target is an existing Akamai Linux host with an NVIDIA RTX PRO 6000 Blackwell GPU.
This repository contains synthetic data generators and benchmark code only; it does not
provision cloud resources or change host drivers. **No Blackwell measurements have been
collected yet.** Local CPU validation cannot establish GPU correctness or speedup.

## What the first suite measures

| Workload | CPU / GPU | Output checked |
|---|---|---|
| Three array transforms | NumPy / CuPy, float32 | Every output, shape, dtype and finite-value status; rtol 1e-5, atol 1e-6 |
| Numeric cleaning and grouping | pandas / cuDF | Exact integer sales totals by region after excluding missing, negative and fractional quantities |
| Unique-key join and grouping | pandas / cuDF | Complete joined-row multiset, duplicate multiplicity, dtypes and exact regional totals |

Before timing, small fixtures check the M02 cuDF sum `2+4+6=12` and the M03 join's four-row
and six-row results. On the CPU path, the GPU sum is explicitly absent. Seeded inputs are
generated in memory. These are new experiments inspired by the lessons; they do not turn
the textbooks' hypothetical timing budgets into measurements of those exact scenarios.

## Run on the Akamai GPU host

Clone your private GitHub repository and work from its root. Read
[Blackwell setup](docs/BLACKWELL-SETUP.md) first: it checks Linux, the selected GPU, compatible
host drivers and Docker's NVIDIA integration. The documented environment is RAPIDS 26.08,
CUDA 13 and Python 3.13. An older generic CUDA minimum is not sufficient evidence of Blackwell
support. The setup guide contains current NVIDIA/Akamai sources and troubleshooting steps.

```bash
git clone https://github.com/brandonholcombe/ca-ads-benchmarks.git nca-ads-benchmarks
cd nca-ads-benchmarks
docker pull nvcr.io/nvidia/rapidsai/base:26.08-cuda13-py3.13
mkdir -p results

# Use an available, allocated device. The container sees it as CUDA device 0.
ADS_GPU_ID=0 bash environments/run-in-container.sh preflight \
  --backend gpu --output results/preflight-first.json

# Start small. A GPU run includes the CPU baseline in the same container and host.
ADS_GPU_ID=0 bash environments/run-in-container.sh run \
  --backend gpu --rows 10000 100000 --repeat 5 --warmup 2 --output results
```

Proceed to larger data only when preflight and the small run pass:

```bash
ADS_GPU_ID=0 bash environments/run-in-container.sh run \
  --backend gpu --rows 100000 1000000 5000000 \
  --repeat 7 --warmup 3 --seed 20260928 --output results
```

Each run creates a new directory. Preflight refuses to replace an existing file; use a new
filename for each attempt. The current cap is five million rows per size, with 3–100 timed
repeats and 1–20 warmups. Start with one GPU. Multi-GPU scaling and model-training benchmarks
are future extensions, not claims made by this version.

The wrapper records the local image ID and available repository digests, then executes that
exact local image ID. It uses your UID/GID and mounts only this repository. To repeat a captured
image, set `ADS_IMAGE` to the recorded NGC digest reference after pulling it. The default tag
fixes release labels but is not immutable. No image pull or GPU execution has been verified
from the authoring Mac.

## Read and return the results

Each run directory contains:

- `results.json`: raw timing samples, correctness evidence, parameters, source hash/commit,
  package versions, CPU context, GPU/runtime details and available container identity. Git
  commit fields may be null when Git is unavailable; the checkout content hash is still recorded.
- `summary.csv`: per-size workload medians and speedup ratios where correctness passed.
- `summary.md`: a human-readable table and timing boundaries.

Send the preflight JSON and the complete run directory back with any notes about concurrent
GPU workloads, instance configuration or changes between runs. A convenient archive is:

```bash
tar -czf benchmark-results.tar.gz -C results .
```

Use a new archive name for each handoff. `results/` and archives are ignored by Git so raw runs
do not silently become source commits. You can attach the archive in this chat. No script
automatically uploads results, creates issues, contacts a reporting service or reads credentials.

A nonzero exit means validation or execution failed. Preserve its error JSON and summary;
do not quote speedups from an unsuccessful run. Missing GPU dependencies cause an error,
not a fallback result labeled as GPU. Review results before changing any course claims.

## Timing boundaries

**CPU** starts with the seeded host inputs and returns usable host output. **GPU resident**
starts with already transferred device inputs and ends with completed device output.
**GPU host-to-host** includes input conversion/transfer, compute, and output transfer back
to the host. All exclude data generation, disk I/O, imports and first process startup.
Device synchronization occurs immediately before and after each GPU measurement. Output
comparison happens outside the timers. Speedup is CPU median divided by the respective GPU
median; a result below one means the GPU path was slower for that case.

Warmups are untimed; allocator and cache reuse are possible. CPU/GPU order alternates between
repeats, while resident and host-to-host GPU paths have a fixed order. Raw samples and quartiles
are retained; these are warm application-level comparisons, not isolated kernel timings,
confidence intervals, cold-start measurements or universal hardware speedups. pandas is
the selected CPU baseline, not a claim to represent the fastest possible CPU engine.
GPU models, CPU allocations, shared load, power/thermal state and software versions all matter.

## Local CPU validation and development

Python 3.11+ is required. In a separate development environment:

Supported execution uses this source checkout, directly or as an editable install. The source
fingerprint includes the Python package, container runner and project metadata from the checkout;
a standalone wheel installation is not the deployment method for this suite.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m nca_ads_bench preflight --backend cpu --output results/local-preflight.json
.venv/bin/python -m nca_ads_bench run --backend cpu --rows 10000 100000 --output results
```

Do not install GPU packages into the course's Manim environment. CPU tests require NumPy and
pandas only; the GPU backend loads CuPy and cuDF only when explicitly requested. The remote
container runs directly from source without an editable install. Full CLI help is available
with `python -m nca_ads_bench --help` and each subcommand's `--help`.

See [implementation review](docs/IMPLEMENTATION-REVIEW.md), [source verification](docs/SOURCES.md)
and [agent rules](AGENTS.md). GitHub publication status is recorded in [handoff](docs/HANDOFF.md).
