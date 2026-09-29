# Collect the remaining-module evidence in one GPU session

Brandon reports executing the first suite; its raw reports have not yet been reviewed here.
This additional batch **does not rerun the original three workloads by default**. It collects
representative evidence for the remaining module outlines while one already allocated GPU is
available. These are bounded synthetic experiments, not exhaustive certification coverage.

## Run now

From the existing `nca-ads-benchmarks` checkout on the GPU host:

```bash
git pull --ff-only
ADS_GPU_ID=0 bash environments/run-in-container.sh course-batch \
  --backend gpu --rows 10000 100000 --repeat 3 --warmup 1 \
  --case-timeout 120 --max-seconds 900 --output results
```

Use the same working container image as the first run; the wrapper's `ADS_IMAGE` override can
pin its recorded digest. No image pull is needed if that image is still available locally.
The new cases require scikit-learn and cuML; the graph case additionally requires cuGraph.
Missing dependencies fail visibly and are recorded. Do not spend the session repeatedly
installing or retrying a failed case—return its report/log for diagnosis. The RAPIDS base image
is documented to include the RAPIDS libraries; actual imported versions are recorded per case.

The default batch executes seven new workloads at two sizes, sequentially on one GPU. Each
child runs a CPU baseline and both GPU timing boundaries. Data generation, imports, warmups,
validation, and the untimed model audit count toward the process time limit. Each process is
limited to 120 seconds; the batch has a 900-second total budget. The final child gets only the
remaining budget. These are stop limits, not an estimate that the run should take 15 minutes.
Host scheduling and process cleanup can add overhead. **Stopping this command does not stop
the cloud instance or its billing.** Release the instance through your normal cloud workflow
when the files have been copied and no other work needs it.

If a smaller workload fails or times out, its larger case is skipped. Independent workloads
continue within the budget. A failed global GPU preflight stops the whole batch. There is no
CPU fallback, automatic retry, cloud provisioning, or external upload.

## What each module can use

| Module | Evidence collected | Limits / work that does not require this GPU session |
|---|---|---|
| M00–M02 | Existing first suite and environment metadata | Do not repeat unless earlier reports failed or environment changed |
| M03 preparation | Existing cleaning and complete join comparisons | Storage/CSV/Parquet ingestion not measured; depends on the future lesson and storage setup |
| M04 features | Train-only scaling and PCA, held-out distribution shift; train/test split | Sampling/class balance/leakage explanations can be checked on CPU |
| M05 EDA | Exact 32-bin histogram with missing values and out-of-range observations | Plotting/rendering not timed; a histogram does not cover all EDA statistics |
| M06 ML | Linear regression, logistic classification, KMeans | Small fixed feature counts; quality checks and solver differences are documented; no hyperparameter search |
| M07 workflows | Ordered scaling→PCA computation and separate transfer boundaries; batch failure/skip records | This is not a Dask, cached pipeline, distributed scaling, or production retraining benchmark |
| M08 MLOps | Raw timings, source/environment metadata, CPU/GPU model serialization round trips | Round trip uses self-created in-memory pickle bytes in the same runtime, not deployment portability or storage latency; drift demonstrations can use CPU |
| M09 time/graphs | Chronologically sorted trailing observation windows with missing values; directed in/out degree | No forecast model, calendar-window resampling, PageRank, or multi-GPU graph claim |

See [ML definitions and quality gates](ML-WORKLOADS.md) and
[data definitions and correctness gates](DATA-WORKLOADS.md). Most module textbooks are still
unwritten; later lesson-specific claims may need a targeted follow-up. Avoid reserving GPU
time for CPU-only arithmetic, documentation review, plotting, or animation rendering.

## Return the evidence

The command prints `results/course_<UTC>_<id>`. That directory contains:

- `batch.json`: complete plan, limits, per-case status, relative report paths, source identity.
- `summary.md`: completion table; incomplete batches exit nonzero.
- `preflight.json` and `preflight.log`.
- One directory per attempted workload/size, with `process.log` and the standard JSON/CSV/Markdown
  run reports. A timed-out process may have only its log and timeout entry; earlier cases remain.

Create a uniquely named archive after the command completes (including an incomplete run):

```bash
archive="nca-ads-results-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
if [ ! -e "$archive" ]; then tar -czf "$archive" -C results .; fi
```

Return the archive in this chat together with the first-run reports. Results and archives stay
ignored by Git. Preserve the originals. We will review correctness and measurement boundaries
before citing any speedup in the course. No remote GPU success is claimed by local unit tests.

## Rerun only a selected case, if needed

After diagnosis, select a workload explicitly so completed tests are not repeated:

```bash
ADS_GPU_ID=0 bash environments/run-in-container.sh course-batch \
  --backend gpu --workloads supervised_linear_regression \
  --rows 10000 --repeat 3 --warmup 1 \
  --case-timeout 120 --max-seconds 180 --output results
```

`course-batch --workloads` accepts the original names as well. It requires ascending, unique
row sizes; new data cases are limited to 100–100000 rows and ML cases to 1000–100000 rows. A new unique directory is always
created. Single `run --workload NAME` remains available but has no process time budget; use
`course-batch` for the bounded cloud workflow.

## Local development

Install the optional CPU course dependencies into a separate environment:

```bash
python -m venv .venv-course
.venv-course/bin/python -m pip install -e '.[course]'
.venv-course/bin/python -m unittest discover -s tests -v
.venv-course/bin/python -m nca_ads_bench course-batch \
  --backend cpu --rows 10000 100000 --output results
```

The original three workloads still require only NumPy and pandas. The expanded test collection
requires the optional course dependencies. GPU libraries are loaded only in the GPU path.
