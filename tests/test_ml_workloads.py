"""Independent quality and rejection checks for the optional ML suite."""

import unittest

import numpy as np

from nca_ads_bench.ml_workloads import WORKLOAD_NAMES, _pickle_audit, dependency_versions, make_case


class NonfiniteModel:
    def fit(self, _x, _y):
        return self

    def predict(self, x):
        return np.full(len(x), np.nan)


class MLWorkloadTests(unittest.TestCase):
    def test_gpu_pca_constructor_contract_without_cuda(self):
        # CPU stand-ins check constructor keywords, not GPU behavior.
        import sys
        from types import SimpleNamespace, ModuleType
        from unittest.mock import patch
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        preprocessing = ModuleType("cuml.preprocessing")
        preprocessing.StandardScaler = StandardScaler
        decomposition = ModuleType("cuml.decomposition")
        def gpu_pca(*, output_type, **kwargs):
            self.assertEqual(output_type, "cupy")
            return PCA(**kwargs)
        decomposition.PCA = gpu_pca
        cp = SimpleNamespace(asarray=np.asarray, asnumpy=np.asarray)
        with patch.dict(sys.modules, {"cuml.preprocessing": preprocessing, "cuml.decomposition": decomposition}), patch("nca_ads_bench.ml_workloads.dependency_versions", return_value={}):
            case = make_case(WORKLOAD_NAMES[0], 1000, 42, gpu=(cp, None))
            output = case.gpu_to_host(case.gpu_compute(case.gpu_prepare(case.host_input)))
            self.assertTrue(case.compare(output, case.cpu(case.host_input))["passed"])

    def test_versions_and_bounds(self):
        self.assertIn("scikit-learn", dependency_versions(False))
        with self.assertRaises(ValueError):
            make_case(WORKLOAD_NAMES[0], 999, 42)
        with self.assertRaises(ValueError):
            make_case(WORKLOAD_NAMES[0], 100_001, 42)
        with self.assertRaises(ValueError):
            make_case("does_not_exist", 100, 42)

    def test_deterministic_inputs_and_all_cpu_quality(self):
        for name in WORKLOAD_NAMES:
            with self.subTest(name=name):
                case = make_case(name, 1000, 42)
                again = make_case(name, 1000, 42)
                self.assertEqual(len(case.host_input[0]), 1000)
                for left, right in zip(case.host_input, again.host_input):
                    np.testing.assert_array_equal(left, right)
                reference = case.cpu(case.host_input)
                self.assertTrue(case.compare(reference, reference)["passed"])
                with self.assertRaises(RuntimeError):
                    case.gpu_prepare(case.host_input)

    def test_scaler_rejects_full_data_fit_leakage(self):
        case = make_case(WORKLOAD_NAMES[0], 1000, 42)
        expected = case.cpu(case.host_input)
        full = case.host_input[0]
        train, test = full[:800], full[800:]
        leaked = dict(expected)
        leaked["mean"] = np.concatenate((train, test)).mean(axis=0)
        self.assertFalse(case.compare(leaked, expected)["passed"])
        bad = dict(expected)
        bad["components"] = np.zeros((2, 4))
        self.assertFalse(case.compare(bad, expected)["passed"])

    def test_regression_rejects_wrong_predictions_and_roundtrip(self):
        case = make_case(WORKLOAD_NAMES[1], 1000, 42)
        expected = case.cpu(case.host_input)
        wrong = dict(expected)
        wrong["prediction"] = np.zeros_like(expected["prediction"])
        self.assertFalse(case.compare(wrong, expected)["passed"])
        audit = case.audit()
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["paths"]["cpu"]["passed"])
        self.assertGreater(audit["paths"]["cpu"]["serialized_bytes"], 0)

    def test_roundtrip_audit_failure_is_json_safe(self):
        import json

        case = make_case(WORKLOAD_NAMES[1], 1000, 42)
        audit = _pickle_audit(case.host_input, NonfiniteModel, lambda value: value, "cpu")
        self.assertFalse(audit["passed"])
        self.assertIsNone(audit["prediction_max_abs_error"])
        json.dumps(audit, allow_nan=False)

    def test_classifier_rejects_trivial_majority_prediction(self):
        case = make_case(WORKLOAD_NAMES[2], 1000, 42)
        expected = case.cpu(case.host_input)
        wrong = {"prediction": np.zeros_like(expected["prediction"])}
        comparison = case.compare(wrong, expected)
        self.assertFalse(comparison["passed"])
        self.assertLess(comparison["held_out_recall"], 0.1)

    def test_clusters_accept_permutation_and_reject_single_label(self):
        case = make_case(WORKLOAD_NAMES[3], 1000, 42)
        expected = case.cpu(case.host_input)
        permuted = {"labels": (np.asarray(expected["labels"]) + 1) % 3,
                    "centers": np.asarray(expected["centers"])[[1, 2, 0]]}
        self.assertTrue(case.compare(permuted, expected)["passed"])
        wrong = {"labels": np.zeros_like(expected["labels"]), "centers": expected["centers"]}
        self.assertFalse(case.compare(wrong, expected)["passed"])


if __name__ == "__main__":
    unittest.main()
