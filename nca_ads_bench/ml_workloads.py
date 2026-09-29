"""Bounded ML cases; model fitting and inference share the timed compute call.

Data generation, transfer, quality checks, and the pickle audit are outside it.
Only the requested GPU path imports cuML. No CPU fallback is attempted there.
"""

from __future__ import annotations

import pickle
from functools import partial

import numpy as np

from .workloads import Case

WORKLOAD_NAMES = (
    "train_only_standardize_pca",
    "supervised_linear_regression",
    "supervised_logistic_classification",
    "unsupervised_kmeans",
)
MIN_ROWS = 1_000
MAX_ROWS = 100_000
TRAIN_FRACTION = 0.8


def dependency_versions(gpu: bool) -> dict[str, str]:
    """Fail at preflight if any estimator used by these cases cannot import."""
    import sklearn
    from sklearn.cluster import KMeans  # noqa: F401
    from sklearn.decomposition import PCA  # noqa: F401
    from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401

    versions = {"scikit-learn": sklearn.__version__}
    if gpu:
        import cuml
        from cuml.cluster import KMeans as GPUKMeans  # noqa: F401
        from cuml.decomposition import PCA as GPUPCA  # noqa: F401
        from cuml.linear_model import LinearRegression as GPULinearRegression  # noqa: F401
        from cuml.linear_model import LogisticRegression as GPULogisticRegression  # noqa: F401
        from cuml.preprocessing import StandardScaler as GPUStandardScaler  # noqa: F401

        versions["cuml"] = cuml.__version__
    return versions


def _split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cut = int(len(x) * TRAIN_FRACTION)
    return x[:cut], x[cut:]


def _data(name: str, rows: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    if name == WORKLOAD_NAMES[0]:
        latent = rng.normal(size=(rows, 2)).astype(np.float32)
        noise = rng.normal(0, 0.03, size=(rows, 4)).astype(np.float32)
        basis = np.array([[2.0, 0.3, 1.0, -0.5], [-0.2, 1.5, 0.4, 1.2]], dtype=np.float32)
        x = latent @ basis + noise
        _, test = _split(x)
        # A held-out shift makes accidental fitting on all rows observable.
        test[:, 0] += np.float32(6.0)
        return (x,)
    if name == WORKLOAD_NAMES[1]:
        x = rng.normal(size=(rows, 6)).astype(np.float32)
        coefficients = np.array([1.5, -2.0, 0.75, 0.4, -1.2, 2.5], dtype=np.float32)
        y = x @ coefficients + np.float32(0.7)
        y += rng.normal(0, 0.05, rows).astype(np.float32)
        return x, y
    if name == WORKLOAD_NAMES[2]:
        x = rng.normal(size=(rows, 5)).astype(np.float32)
        margin = 2.5 * x[:, 0] - 1.8 * x[:, 1] + 1.2 * x[:, 2] + 0.3
        y = (margin > 0).astype(np.int32)
        return x, y
    if name == WORKLOAD_NAMES[3]:
        centers = np.array([[-5.0, -5.0], [0.0, 6.0], [6.0, -3.0]], dtype=np.float32)
        truth = np.arange(rows, dtype=np.int32) % 3
        x = centers[truth] + rng.normal(0, 0.30, size=(rows, 2)).astype(np.float32)
        order = rng.permutation(rows)
        return np.ascontiguousarray(x[order]), np.ascontiguousarray(truth[order])
    raise ValueError(f"unknown ML workload: {name}")


def _pca_compute(data, scaler_type, pca_type):
    train, test = _split(data[0])
    scaler = scaler_type()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    pca = pca_type(n_components=2)
    pca.fit(train_scaled)
    return {"mean": scaler.mean_, "scale": scaler.scale_,
            "test_scaled_mean": test_scaled.mean(axis=0),
            "components": pca.components_, "variance_ratio": pca.explained_variance_ratio_}


def _regression_compute(data, model_type):
    train, test = _split(data[0])
    y_train, _ = _split(data[1])
    model = model_type()
    model.fit(train, y_train)
    return {"prediction": model.predict(test), "coefficients": model.coef_}


def _classification_compute(data, model_type):
    train, test = _split(data[0])
    y_train, _ = _split(data[1])
    model = model_type(max_iter=200)
    model.fit(train, y_train)
    return {"prediction": model.predict(test)}


def _cluster_compute(data, model_type, seed: int):
    x, _ = data
    model = model_type(n_clusters=3, init="k-means++", n_init=3, max_iter=100, random_state=seed)
    model.fit(x)
    return {"labels": model.predict(x), "centers": model.cluster_centers_}


def _host_result(result: dict, cp) -> dict[str, np.ndarray]:
    return {key: cp.asnumpy(value) for key, value in result.items()}


def _finite_array(value, shape) -> bool:
    array = np.asarray(value)
    return array.shape == shape and np.issubdtype(array.dtype, np.number) and bool(np.isfinite(array).all())


def _compare_pca(actual, expected, data) -> dict:
    train, _ = _split(data[0])
    mean = np.asarray(actual["mean"])
    scale = np.asarray(actual["scale"])
    test_mean = np.asarray(actual["test_scaled_mean"])
    components = np.asarray(actual["components"])
    ratio = np.asarray(actual["variance_ratio"])
    shapes = (_finite_array(mean, (4,)) and _finite_array(scale, (4,))
              and _finite_array(test_mean, (4,)) and _finite_array(components, (2, 4))
              and _finite_array(ratio, (2,)))
    if not shapes:
        return {"passed": False, "reason": "invalid shape or nonfinite PCA output"}
    independent_mean = train.mean(axis=0)
    independent_scale = train.std(axis=0)
    mean_error = float(np.max(np.abs(mean - independent_mean)))
    scale_error = float(np.max(np.abs(scale - independent_scale)))
    test_mean_error = float(np.max(np.abs(test_mean - np.asarray(expected["test_scaled_mean"]))))
    projector = components.T @ components
    expected_components = np.asarray(expected["components"])
    expected_projector = expected_components.T @ expected_components
    projector_error = float(np.max(np.abs(projector - expected_projector)))
    variance_error = float(np.max(np.abs(ratio - np.asarray(expected["variance_ratio"]))))
    passed = (mean_error < 0.01 and scale_error < 0.01 and test_mean_error < 0.1
              and projector_error < 0.1 and variance_error < 0.05
              and test_mean[0] > 2.0 and ratio.sum() > 0.95)
    return {"passed": bool(passed), "train_mean_max_abs_error": mean_error,
            "train_scale_max_abs_error": scale_error,
            "held_out_scaled_mean_first_feature": float(test_mean[0]),
            "test_mean_max_abs_error_vs_cpu": test_mean_error,
            "pca_projector_max_abs_error_vs_cpu": projector_error,
            "variance_ratio_max_abs_error_vs_cpu": variance_error}


def _compare_regression(actual, expected, data) -> dict:
    _, y_test = _split(data[1])
    prediction = np.asarray(actual["prediction"]).reshape(-1)
    coefficients = np.asarray(actual["coefficients"]).reshape(-1)
    if not (_finite_array(prediction, y_test.shape) and _finite_array(coefficients, (6,))):
        return {"passed": False, "reason": "invalid shape or nonfinite regression output"}
    rmse = float(np.sqrt(np.mean((prediction - y_test) ** 2)))
    cpu_rmse = float(np.sqrt(np.mean((prediction - np.asarray(expected["prediction"]).reshape(-1)) ** 2)))
    true_coefficients = np.array([1.5, -2.0, 0.75, 0.4, -1.2, 2.5])
    coefficient_error = float(np.max(np.abs(coefficients - true_coefficients)))
    return {"passed": bool(rmse < 0.15 and cpu_rmse < 0.15 and coefficient_error < 0.15),
            "held_out_rmse": rmse, "prediction_rmse_vs_cpu": cpu_rmse,
            "max_coefficient_error_vs_truth": coefficient_error}


def _compare_classification(actual, expected, data) -> dict:
    _, y_test = _split(data[1])
    prediction = np.asarray(actual["prediction"]).reshape(-1)
    if not _finite_array(prediction, y_test.shape) or not np.isin(prediction, [0, 1]).all():
        return {"passed": False, "reason": "invalid classification labels or shape"}
    accuracy = float(np.mean(prediction == y_test))
    positive = prediction == 1
    true_positive = int(np.count_nonzero(positive & (y_test == 1)))
    false_positive = int(np.count_nonzero(positive & (y_test == 0)))
    false_negative = int(np.count_nonzero(~positive & (y_test == 1)))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    disagreement = float(np.mean(prediction != np.asarray(expected["prediction"]).reshape(-1)))
    return {"passed": bool(accuracy > 0.94 and precision > 0.93 and recall > 0.93
                           and disagreement < 0.06), "held_out_accuracy": accuracy,
            "held_out_precision": precision, "held_out_recall": recall,
            "label_disagreement_vs_cpu": disagreement}


def _compare_cluster(actual, expected, data) -> dict:
    from sklearn.metrics import adjusted_rand_score

    x, truth = data
    labels = np.asarray(actual["labels"]).reshape(-1)
    centers = np.asarray(actual["centers"])
    if not (_finite_array(labels, truth.shape) and _finite_array(centers, (3, 2))):
        return {"passed": False, "reason": "invalid cluster labels or centers"}
    if len(np.unique(labels)) != 3:
        return {"passed": False, "reason": "expected three nonempty clusters"}
    truth_centers = np.array([[-5, -5], [0, 6], [6, -3]], dtype=np.float32)
    distances = np.linalg.norm(centers[:, None, :] - truth_centers[None, :, :], axis=2)
    center_error = float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))
    ari_truth = float(adjusted_rand_score(truth, labels))
    ari_cpu = float(adjusted_rand_score(np.asarray(expected["labels"]).reshape(-1), labels))
    return {"passed": bool(center_error < 0.5 and ari_truth > 0.98 and ari_cpu > 0.98),
            "max_nearest_center_distance": center_error, "adjusted_rand_vs_truth": ari_truth,
            "adjusted_rand_vs_cpu": ari_cpu}


def _pickle_audit(data, model_type, to_host, backend: str) -> dict:
    train, test = _split(data[0])
    y_train, _ = _split(data[1])
    model = model_type()
    model.fit(train, y_train)
    before = np.asarray(to_host(model.predict(test))).reshape(-1)
    payload = pickle.dumps(model, protocol=5)
    restored = pickle.loads(payload)  # Only bytes created above in this process.
    after = np.asarray(to_host(restored.predict(test))).reshape(-1)
    valid = (before.shape == after.shape and before.shape == (len(test),)
             and np.isfinite(before).all() and np.isfinite(after).all())
    max_error = float(np.max(np.abs(before - after))) if valid else None
    passed = bool(max_error is not None and np.isfinite(max_error) and max_error < 1e-5)
    if max_error is not None and not np.isfinite(max_error):
        max_error = None
    return {"passed": passed, "backend": backend, "pickle_protocol": 5,
            "serialized_bytes": len(payload), "prediction_max_abs_error": max_error,
            "checked_predictions": int(len(before)),
            "versions": dependency_versions(backend == "gpu")}


def make_case(name: str, rows: int, seed: int, gpu=None) -> Case:
    if name not in WORKLOAD_NAMES:
        raise ValueError(f"unknown ML workload: {name}")
    if not MIN_ROWS <= rows <= MAX_ROWS:
        raise ValueError(f"ML rows must be between {MIN_ROWS} and {MAX_ROWS}: {rows}")
    dependency_versions(gpu is not None)
    data = _data(name, rows, seed)
    if gpu is None:
        def unavailable(*_):
            raise RuntimeError("GPU backend unavailable")
        gpu_prepare = gpu_compute = gpu_to_host = unavailable
    else:
        cp, _cudf = gpu
        def gpu_prepare(host):
            return tuple(cp.asarray(value) for value in host)
        gpu_to_host = lambda result: _host_result(result, cp)
        if name == WORKLOAD_NAMES[0]:
            from cuml.decomposition import PCA as GPUPCA
            from cuml.preprocessing import StandardScaler as GPUStandardScaler
            gpu_compute = lambda dev: _pca_compute(
                dev, GPUStandardScaler,
                partial(GPUPCA, output_type="cupy"))
        elif name == WORKLOAD_NAMES[1]:
            from cuml.linear_model import LinearRegression as GPULinearRegression
            gpu_compute = lambda dev: _regression_compute(dev, partial(GPULinearRegression, output_type="cupy"))
        elif name == WORKLOAD_NAMES[2]:
            from cuml.linear_model import LogisticRegression as GPULogisticRegression
            gpu_compute = lambda dev: _classification_compute(dev, partial(GPULogisticRegression, output_type="cupy"))
        else:
            from cuml.cluster import KMeans as GPUKMeans
            gpu_compute = lambda dev: _cluster_compute(dev, partial(GPUKMeans, output_type="cupy"), seed)

    if name == WORKLOAD_NAMES[0]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        cpu = lambda host: _pca_compute(host, StandardScaler, PCA)
        compare = lambda actual, expected: _compare_pca(actual, expected, data)
    elif name == WORKLOAD_NAMES[1]:
        from sklearn.linear_model import LinearRegression
        cpu = lambda host: _regression_compute(host, LinearRegression)
        compare = lambda actual, expected: _compare_regression(actual, expected, data)
    elif name == WORKLOAD_NAMES[2]:
        from sklearn.linear_model import LogisticRegression
        cpu = lambda host: _classification_compute(host, LogisticRegression)
        compare = lambda actual, expected: _compare_classification(actual, expected, data)
    else:
        from sklearn.cluster import KMeans
        cpu = lambda host: _cluster_compute(host, KMeans, seed)
        compare = lambda actual, expected: _compare_cluster(actual, expected, data)

    case = Case(name, data, cpu, gpu_prepare, gpu_compute, gpu_to_host, compare)
    if name == WORKLOAD_NAMES[1]:
        def audit():
            from sklearn.linear_model import LinearRegression
            result = {"cpu": _pickle_audit(data, LinearRegression, lambda value: value, "cpu")}
            if gpu is not None:
                from cuml.linear_model import LinearRegression as GPULinearRegression
                device_data = gpu_prepare(data)
                result["gpu"] = _pickle_audit(device_data, partial(GPULinearRegression, output_type="cupy"),
                                               gpu[0].asnumpy, "gpu")
            return {"passed": all(item["passed"] for item in result.values()), "paths": result}
        case.audit = audit
    return case
