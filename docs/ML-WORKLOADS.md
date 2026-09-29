# Optional ML and feature benchmark cases

These four cases extend the NCA-ADS benchmark harness at the `Case` boundary. They use seeded,
bounded synthetic `float32` features, 1,000–100,000 rows, and one GPU when requested. The first
tuple item always has the requested row count; the harness may report it consistently. A single
case's host input is created before timing. CPU uses scikit-learn; GPU uses explicit single-GPU
cuML estimators with CuPy inputs and outputs. Neither path enables `cuml.accel`. The CPU baseline
must run in the same remote environment as the GPU trial. `dependency_versions(gpu=True)`
checks imports and exposes the scikit-learn/cuML versions at preflight; a missing GPU dependency
must end the GPU run visibly.

| Case | Timed operation | Correctness check outside timing |
|---|---|---|
| `train_only_standardize_pca` | Fit scaling on first 80%, scale both splits, fit two-component PCA on scaled training rows | Train means and population scales against direct statistics; held-out feature shift remains visible; PCA subspace projector and explained variance compared within tolerances |
| `supervised_linear_regression` | Fit ordinary least squares on 80%, predict held-out 20% | Held-out RMSE, coefficient error against known data-generating coefficients, and prediction RMSE against CPU |
| `supervised_logistic_classification` | Fit binary logistic regression on 80%, predict held-out 20% | Accuracy, precision and recall against known labels, plus CPU/GPU label disagreement |
| `unsupervised_kmeans` | Fit three-centroid K-means (`init="k-means++"`, three starts, at most 100 iterations) and predict all rows | Adjusted Rand index against known membership and CPU membership, plus centroid distances; cluster labels may permute |

Each timed compute creates a fresh model, including during warmup and repeats. The runner's
GPU-resident timer begins with CuPy input already prepared and ends after synchronized model
output. Its host-to-host timer also includes transfer and conversion. Input generation, imports,
quality checks, and the regression persistence audit occur outside both timers. The outputs
are arrays kept only for validation; comparison evidence in reports is compact scalar data.
No score is a benchmark result until the exact code revision passes on the intended NVIDIA GPU.

The regression `audit()` hook trains fresh models outside timing, serializes each with pickle
protocol 5 to bytes in memory, reloads those bytes in the same process, and checks held-out
predictions. It returns serializable pass/fail evidence and byte count for CPU and, when selected,
GPU. This tests a same-version in-memory round trip. It does not measure storage performance,
cross-version compatibility, or deployment on another host. It loads only bytes it just created;
untrusted pickle data must never be loaded.

The synthetic tasks are deliberately easy so they can detect gross correctness failures. Their
quality thresholds are gates, not claims that CPU and GPU optimizers produce bit-identical
coefficients, PCA signs, K-means label IDs, or equal timing. The train-only PCA case includes
a held-out distribution shift specifically to expose leakage. The regression is low-noise,
classification labels follow a linear boundary, and clusters are well separated. These cases
do not establish performance or model quality on real datasets, address class imbalance, or
measure cross-validation, tuning, distributed execution, tracking services, or durable artifacts.

The estimator classes and their input/output types come from [cuML's API reference](https://docs.rapids.ai/api/cuml/stable/api/),
which lists `StandardScaler`, `PCA`, `LinearRegression`, `LogisticRegression`, and single-GPU
`KMeans`. [cuML's output-type documentation](https://docs.rapids.ai/api/cuml/stable/api/generated/cuml.set_global_output_type/)
explains that CuPy outputs remain device arrays and NumPy conversion transfers data to host.
[NVIDIA's model serialization guide](https://docs.rapids.ai/api/cuml/stable/pickling_cuml_models/)
documents same-version single-GPU pickle round trips and the risk of loading untrusted bytes.
These API references were reviewed on 2026-09-29. The actual RAPIDS 26.08 image contents and
GPU behavior remain to be checked by the remote preflight and run.

Release-specific checks: the [26.08 StandardScaler constructor](https://docs.nvidia.com/cuml/26.08/api/generated/cuml.preprocessing.StandardScaler/)
does not accept the estimator `output_type` keyword, so the scaler receives CuPy inputs directly.
The [26.08 KMeans API](https://docs.nvidia.com/cuml/26.08/api/generated/cuml.cluster.KMeans/)
supports the explicit initialization used on both paths. Matching initialization names and seeds
does not guarantee identical initial centers or solver implementations across libraries.
