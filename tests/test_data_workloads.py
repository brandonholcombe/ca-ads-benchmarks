import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nca_ads_bench import data_workloads as dw


class DataWorkloadTests(unittest.TestCase):
    def test_cpu_cases_are_deterministic_and_complete(self):
        for name in dw.WORKLOAD_NAMES:
            first = dw.make_case(name, 100, 9)
            second = dw.make_case(name, 100, 9)
            if isinstance(first.host_input, pd.DataFrame):
                pd.testing.assert_frame_equal(first.host_input, second.host_input)
            else:
                np.testing.assert_array_equal(first.host_input, second.host_input)
            result = first.cpu(first.host_input)
            self.assertTrue(first.compare(result, result)["passed"], name)
        self.assertEqual(set(dw.dependency_versions(False)), {"numpy", "pandas"})

    def test_case_factory_rejects_unbounded_sizes(self):
        for name in dw.WORKLOAD_NAMES:
            for rows in (99, 100_001):
                with self.assertRaisesRegex(ValueError, "data rows"):
                    dw.make_case(name, rows, 7)

    def test_histogram_fixed_edges_missing_and_outliers(self):
        fixture = np.array([0.0, 0.1, 0.5, 1.0, np.nan, -0.1, 1.1], dtype=np.float32)
        counts, edges = dw.histogram_compute(fixture, np)
        self.assertEqual(int(counts.sum()), 4)
        self.assertEqual(counts[[0, 3, 16, 31]].tolist(), [1, 1, 1, 1])
        self.assertEqual(float(edges[0]), 0.0)
        self.assertEqual(float(edges[-1]), 1.0)
        wrong = counts.copy()
        wrong[0] -= 1
        wrong[1] += 1
        self.assertFalse(dw.compare_histogram((wrong, edges), (counts, edges))["passed"])
        invalid = counts.astype(float)
        invalid[0] = np.nan
        self.assertFalse(dw.compare_histogram((invalid, edges), (counts, edges))["passed"])

    def test_rolling_is_chronological_trailing_and_missing_aware(self):
        frame = pd.DataFrame({"minute": [40, 10, 60, 20, 50, 30],
                              "value": [1000.0, 2.0, 30.0, 4.0, np.nan, 6.0]})
        result = dw.rolling_compute(frame)
        self.assertEqual(result.minute.tolist(), [10, 20, 30, 40, 50, 60])
        self.assertTrue(np.isnan(result.rolling_mean.iloc[0]))
        self.assertTrue(np.isnan(result.rolling_mean.iloc[1]))
        self.assertEqual(result.rolling_mean.iloc[2], 4.0)
        self.assertEqual(result.rolling_mean.iloc[4], 253.0)
        self.assertEqual(result.rolling_mean.iloc[5], 208.4)
        # Changing a later observation cannot alter any earlier window.
        changed = frame.copy()
        changed.loc[changed.minute == 60, "value"] = -9999.0
        other = dw.rolling_compute(changed)
        np.testing.assert_array_equal(result.rolling_mean.iloc[:5], other.rolling_mean.iloc[:5])
        self.assertFalse(dw.compare_rolling(result.iloc[::-1], result)["passed"])

    def test_graph_directed_sink_and_order_independent_comparison(self):
        edges = pd.DataFrame({"src": np.array([0, 0, 1, 2], dtype=np.int32),
                              "dst": np.array([1, 2, 2, 3], dtype=np.int32)})
        got = dw.graph_cpu(edges)
        self.assertEqual(got.to_dict("list"), {"vertex": [0, 1, 2, 3],
                                               "in_degree": [0, 1, 2, 1],
                                               "out_degree": [2, 1, 1, 0]})
        self.assertTrue(dw.compare_graph(got.iloc[::-1], got)["passed"])
        wrong = got.copy()
        wrong.loc[wrong.vertex == 3, "out_degree"] = 1
        self.assertFalse(dw.compare_graph(wrong, got)["passed"])
        renamed = got.copy()
        renamed.loc[renamed.vertex == 3, "vertex"] = 99
        self.assertFalse(dw.compare_graph(renamed, got)["passed"])

    def test_graph_gpu_contract_uses_directed_ids_without_renumber(self):
        events = []
        edges = pd.DataFrame({"src": [0, 1], "dst": [1, 2]})
        class FakeGraph:
            def __init__(self, directed):
                events.append(("directed", directed))
            def from_cudf_edgelist(self, value, source, destination, renumber):
                events.append(("edgelist", value is edges, source, destination, renumber))
            def degrees(self):
                events.append(("degrees",))
                return pd.DataFrame({"vertex": [2, 1, 0],
                                     "in_degree": [1, 1, 0],
                                     "out_degree": [0, 1, 1]})
        class FakeCuGraph:
            Graph = FakeGraph
        result = dw.graph_gpu(edges, FakeCuGraph)
        self.assertTrue(dw.compare_graph(result, dw.graph_cpu(edges))["passed"])
        self.assertEqual(events, [("directed", True),
                                  ("edgelist", True, "src", "dst", False), ("degrees",)])

    def test_generated_graph_is_capped_unique_and_has_sink(self):
        for rows in (100, 10_000, 100_000):
            edges = dw.graph_input(rows, 7)
            self.assertEqual(len(edges), rows)
            self.assertFalse(edges.duplicated().any())
            self.assertFalse((edges.src == edges.dst).any())
            degree = dw.graph_cpu(edges)
            self.assertEqual(int(degree.in_degree.sum()), len(edges))
            self.assertEqual(int(degree.out_degree.sum()), len(edges))
            self.assertTrue(((degree.in_degree > 0) & (degree.out_degree == 0)).any())
            np.testing.assert_array_equal(degree.vertex, np.arange(len(degree), dtype=np.int32))
        for rows in (99, 100_001):
            with self.assertRaisesRegex(ValueError, "directed graph rows"):
                dw.graph_input(rows, 7)

    def test_gpu_dependency_failure_is_visible(self):
        with patch.object(dw, "import_module", side_effect=ModuleNotFoundError("No module named 'cupy'")):
            with self.assertRaises(ModuleNotFoundError):
                dw.dependency_versions(True)
        class Version:
            __version__ = "fixture"
        with patch.object(dw, "import_module", return_value=Version()) as imported:
            self.assertEqual(set(dw.dependency_versions(True, name=dw.WORKLOAD_NAMES[0])),
                             {"numpy", "pandas", "cupy", "cudf"})
            self.assertEqual(imported.call_count, 2)
        with patch.object(dw, "import_module", return_value=Version()) as imported:
            self.assertIn("cugraph", dw.dependency_versions(True, name=dw.WORKLOAD_NAMES[2]))
            self.assertEqual(imported.call_count, 3)
        with self.assertRaises(ValueError):
            dw.make_case("unknown", 10, 1)
        case = dw.make_case(dw.WORKLOAD_NAMES[2], 100, 1)
        with self.assertRaisesRegex(RuntimeError, "GPU backend unavailable"):
            case.gpu_prepare(case.host_input)


if __name__ == "__main__":
    unittest.main()
