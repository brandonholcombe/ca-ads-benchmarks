# Independent benchmark implementation review

**Status: PASS for the documented source-checkout CPU workflow, pending the authorized Blackwell GPU preflight.** Rechecked 2026-09-28 against `nca_ads_bench/*.py`, `tests/test_benchmark.py`, and `environments/run-in-container.sh`. The bundled Python runtime passed **18 CPU tests**; `bash -n` passed for the container runner. No Docker image was pulled and no NVIDIA GPU, CUDA, or cuDF benchmark was executed here.

## Resolved review findings

1. **Output parity and JSON safety.** `compare_join()` now sorts and compares the full joined row multiset, including row values and multiplicity, and checks exact integer group totals. The test changes a joined `customer_id` while leaving totals intact and correctly fails. `compare_array()` rejects NaN and infinities and records a JSON-safe error; tests cover all three nonfinite values.
2. **Environment provenance.** `source_sha256` now covers the benchmark Python modules, `pyproject.toml`, and the container runner. The benchmark directory has been initialized as Git; the run records commit and dirty state when Git is available. The runner inspects the locally pulled image, passes only `ADS_CONTAINER_IMAGE`, `ADS_CONTAINER_IMAGE_ID`, and `ADS_CONTAINER_REPODIGESTS`, and executes the inspected image ID. A test confirms unrelated environment values are not copied into metadata. This records the local image identity even when an NGC repository digest is unavailable.
3. **Failure rows.** The result plan now contains every requested row-count/workload combination. A failed case is marked `error`, later combinations stay `not_run`, and JSON, CSV and Markdown keep the full plan without invented timings. Tests exercise GPU preflight failure, nonfinite correctness failure, and a mid-case compute exception. Eager input-generation failures are attributed to the row count and phase, without incorrectly blaming the first workload.
4. **Invalid seed.** The CLI now limits the seed to `0..2**32-1`; `--seed -1` exits with an argument error before creating a run directory.

## Other checks

- GPU mode imports CuPy and cuDF, requires a visible CUDA device, selects one device, and synchronizes before and after timed GPU work. A missing GPU produces an error report with no CPU fallback.
- GPU resident timing excludes initial host-to-device preparation and later correctness conversion. Host-to-host timing includes transfer, compute, and conversion. CPU/GPU trial order alternates, warmups are untimed, and allocator reuse is disclosed.
- The runner mounts the benchmark repository, uses the invoking user's UID/GID for output ownership, and requests only the selected GPU in GPU mode. CPU mode requests no GPU. A fake-Docker check by the parent exercised argument assembly; this is not a real container run.

## Limits of this review

- CUDA/cuDF package imports, `sm_120` execution, host driver compatibility, timing values, and GPU result parity require the actual Akamai host. Do not report GPU speedups before its preflight and benchmark pass.
- `source_hash()` assumes execution from the documented source checkout because it reads adjacent `pyproject.toml` and `environments/run-in-container.sh`. A standalone installed wheel lacks those files and would fail before report creation. The README's clone or editable-source path satisfies this assumption; wheel execution is outside the reviewed path.

This review supersedes the earlier three-finding draft. Those findings were repaired and verified by source inspection and CPU tests.

## Remaining-module extension — 2026-09-29

GPT-6 Sol high implemented the ML and data workload files in two bounded assignments. Each
worker then reviewed the other implementation. A separate Sol high review checked the parent
batch runner; the parent integrated and verified fixes. Subscription allowance usage was not
measured. No paid API, new GPU allocation, or host changes were made.

Review corrections accepted:
- Handle a partial JSON report from a killed child per case; continue independent workloads.
- Reject out-of-bounds data case sizes at the factory as well as the CLI.
- Remove unsupported cuML 26.08 StandardScaler `output_type` constructor argument.
- Use explicit `k-means++` initialization and three starts for both KMeans implementations.
- Preserve JSON-safe error evidence for nonfinite model round-trip predictions.

Validation: 44 unit/failure-path tests passed in isolated `.venv-course` with scikit-learn 1.9.1;
all 14 default CPU workload/size combinations passed at 10,000 and 100,000 rows. Targeted PCA
and KMeans reruns validate the final initialization/API-contract changes on CPU. A real child
process timeout test confirms process termination and preserved logs. Mock tests cover GPU
absence, constructor contracts, partial reports and budget exhaustion; none is GPU evidence.
Container wrapper Bash syntax passes. The code's GPU paths require the exact published revision
to run on the allocated Blackwell host. Prior first-suite GPU execution is user-reported only;
its returned artifacts have not yet been reviewed.

Scope: seven representative workloads, a same-runtime in-memory model round trip, per-case and
total time budgets, and preserved run artifacts. See COURSE-BATCH.md for module mapping and
explicit gaps. This does not validate distributed Dask scaling, durable deployment, forecast
models, or every future textbook example.
