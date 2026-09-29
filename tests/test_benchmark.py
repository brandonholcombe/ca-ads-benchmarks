import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nca_ads_bench.cli import bounded_seed, main, positive_rows, repeat_count, warmup_count
from nca_ads_bench.runner import base_metadata, describe, preflight, run, source_hash, timed
from nca_ads_bench.workloads import (WORKLOAD_NAMES, Case, array_input, cleanup_compute, cleanup_input,
                                     compare_array, compare_join, fixture_checks,
                                     join_compute, join_input, make_cases)


class WorkloadTests(unittest.TestCase):
    def test_deterministic_inputs_and_references(self):
        self.assertTrue(np.array_equal(array_input(50, 7), array_input(50, 7)))
        self.assertTrue(cleanup_input(50, 7).equals(cleanup_input(50, 7)))
        self.assertTrue(all(a.equals(b) for a, b in zip(join_input(50, 7), join_input(50, 7))))
        for case in make_cases(50, 7):
            value = case.cpu(case.host_input)
            self.assertTrue(case.compare(value, value)["passed"], case.name)

    def test_fixture_preserves_duplicate_multiplicity(self):
        self.assertEqual(fixture_checks()["unique_lookup_rows"], 4)
        self.assertEqual(fixture_checks()["duplicate_lookup_rows"], 6)

    def test_cleanup_quarantines_missing_fractional_negative(self):
        case = make_cases(100, 3)[1]
        frame = case.host_input
        result = case.cpu(frame)
        valid = frame.quantity.notna() & (frame.quantity > 0) & ((frame.quantity % 1) == 0)
        self.assertLess(valid.sum(), len(frame))
        self.assertEqual(sum(result), int((frame.loc[valid, "quantity"].astype("int64") * frame.loc[valid, "unit_cents"]).sum()))

    def test_join_key_cardinality(self):
        case = make_cases(100, 3)[2]
        left, right = case.host_input
        self.assertTrue(right.customer_id.is_unique)
        joined, totals = case.cpu((left, right))
        self.assertEqual(len(joined), len(left))
        self.assertEqual(int(totals.sum()), int(left.amount_cents.sum()))

    def test_fixed_cleanup_and_join_outputs(self):
        frame = pd.DataFrame({"region": [0, 0, 1, 1],
                              "quantity": [2.0, np.nan, -1.0, 3.0],
                              "unit_cents": [100, 200, 300, 400]})
        self.assertEqual(cleanup_compute(frame).to_dict(), {0: 200, 1: 1200})
        left = pd.DataFrame({"customer_id": [1, 1, 2], "amount_cents": [10, 20, 30]})
        right = pd.DataFrame({"customer_id": [1, 2], "region": [0, 1]})
        joined, totals = join_compute((left, right))
        self.assertEqual(len(joined), 3)
        self.assertEqual(totals.to_dict(), {0: 30, 1: 30})

    def test_join_rejects_wrong_rows_with_same_totals(self):
        case = make_cases(40, 4)[2]
        expected = case.cpu(case.host_input)
        wrong = expected[0].copy()
        wrong.loc[0, "customer_id"] = -123
        check = compare_join((wrong, expected[1]), expected)
        self.assertFalse(check["passed"])
        self.assertFalse(check["row_multiset_match"])
        self.assertEqual(check["actual_totals"], check["expected_totals"])

    def test_array_nonfinite_errors_are_json_safe(self):
        expected = np.array([1.0, 2.0], dtype=np.float32)
        for bad in (np.nan, np.inf, -np.inf):
            result = compare_array(np.array([1.0, bad], dtype=np.float32), expected)
            self.assertFalse(result["passed"])
            self.assertEqual(result["nonfinite_count"], 1)
            self.assertIsNone(result["max_abs_error"])
            json.dumps(result, allow_nan=False)


class ReportTests(unittest.TestCase):
    def test_input_generation_failure_does_not_blame_first_workload(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "nca_ads_bench.runner.make_cases", side_effect=MemoryError("fixture allocation failed")
        ):
            directory, report = run("cpu", [20, 40], 3, 1, 9, Path(tmp))
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["error"]["phase"], "input_generation")
            self.assertEqual(report["error"]["rows"], 20)
            self.assertIsNone(report["error"]["workload"])
            self.assertEqual(len(report["results"]), 6)
            self.assertTrue(all(row["status"] == "not_run" for row in report["results"]))
            self.assertTrue((directory / "results.json").exists())

    def test_stats_interpolation(self):
        stat = describe([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual([stat["q25_ms"], stat["median_ms"], stat["q75_ms"]], [2, 3, 4])
        self.assertEqual(stat["samples_ms"], [1, 2, 3, 4, 5])

    def test_timer_sync_boundary(self):
        events = []
        def sync(): events.append("sync")
        def work(): events.append("work"); return 12
        value, elapsed = timed(work, sync)
        self.assertEqual(value, 12)
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(events, ["sync", "work", "sync"])

    def test_cpu_run_schema_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, report = run("cpu", [20], 3, 1, 9, Path(tmp))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(len(report["results"]), 3)
            self.assertEqual([r["status"] for r in report["results"]], ["passed"] * 3)
            self.assertTrue(all(r["correctness_passed"] for r in report["results"]))
            self.assertTrue(all("gpu_resident_ms" not in r for r in report["results"]))
            self.assertEqual(json.loads((directory / "results.json").read_text())["schema_version"], 1)
            self.assertTrue((directory / "summary.csv").exists())
            self.assertTrue((directory / "summary.md").exists())
            other, _ = run("cpu", [20], 3, 1, 9, Path(tmp))
            self.assertNotEqual(directory, other)

    def test_gpu_failure_still_records_artifacts_no_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, patch("nca_ads_bench.runner.gpu_modules", side_effect=RuntimeError("no GPU")):
            directory, report = run("gpu", [20], 3, 1, 9, Path(tmp))
            self.assertEqual(report["status"], "error")
            self.assertEqual(len(report["results"]), 3)
            self.assertEqual([r["status"] for r in report["results"]], ["not_run"] * 3)
            self.assertIn("no GPU", (directory / "summary.md").read_text())
            self.assertEqual(json.loads((directory / "results.json").read_text())["status"], "error")

    def test_failed_correctness_still_records_valid_json(self):
        case = Case(WORKLOAD_NAMES[0], np.array([1.0, 2.0], dtype=np.float32),
                    lambda x: np.array([np.nan, np.inf], dtype=np.float32),
                    None, None, None, compare_array)
        with tempfile.TemporaryDirectory() as tmp, patch("nca_ads_bench.runner.make_cases", return_value=[case]):
            directory, report = run("cpu", [2], 3, 1, 9, Path(tmp))
            self.assertEqual(report["status"], "error")
            self.assertFalse(report["results"][0]["correctness_passed"])
            self.assertEqual([r["status"] for r in report["results"]], ["error", "not_run", "not_run"])
            loaded = json.loads((directory / "results.json").read_text())
            self.assertEqual(loaded["status"], "error")
            self.assertIsNone(loaded["results"][0]["comparisons"][0]["max_abs_error"])
            self.assertTrue((directory / "summary.md").exists())
            self.assertTrue((directory / "summary.csv").exists())

    def test_mid_case_failure_names_case_and_preserves_full_plan(self):
        original = make_cases
        def cases(size, seed, gpu):
            generated = original(size, seed, gpu)
            if size == 30:
                cpu = generated[1].cpu
                calls = 0
                def fail_after_reference(value):
                    nonlocal calls
                    calls += 1
                    if calls > 1:
                        raise RuntimeError("injected compute failure")
                    return cpu(value)
                generated[1].cpu = fail_after_reference
            return generated
        with tempfile.TemporaryDirectory() as tmp, patch("nca_ads_bench.runner.make_cases", side_effect=cases):
            directory, report = run("cpu", [20, 30, 40], 3, 1, 9, Path(tmp))
            self.assertEqual(report["status"], "error")
            self.assertEqual(len(report["results"]), 9)
            self.assertEqual([r["status"] for r in report["results"]],
                             ["passed", "passed", "passed", "passed", "error", "not_run", "not_run", "not_run", "not_run"])
            self.assertEqual(report["error"]["rows"], 30)
            self.assertEqual(report["error"]["workload"], WORKLOAD_NAMES[1])
            self.assertEqual(report["error"]["phase"], "measurement")
            self.assertNotIn("cpu_ms", report["results"][4])
            self.assertNotIn("speedup_cpu_over_gpu_host_to_host", report["results"][4])
            loaded = json.loads((directory / "results.json").read_text())
            self.assertEqual(len(loaded["results"]), 9)
            summary = (directory / "summary.csv").read_text()
            self.assertIn("injected compute failure", summary)
            self.assertIn("not_run", summary)
            markdown = (directory / "summary.md").read_text()
            self.assertIn("Failed case: rows=30, workload=numeric_cleanup_groupby, phase=measurement", markdown)
            self.assertIn("injected compute failure", markdown)

    def test_source_hash_covers_runner_files(self):
        self.assertEqual(len(source_hash()), 64)
        from nca_ads_bench import runner
        root = Path(runner.__file__).resolve().parent.parent
        wrapper = root / "environments" / "run-in-container.sh"
        # Compute the expected hash independently using the declared file set.
        import hashlib
        files = list((root / "nca_ads_bench").glob("*.py")) + [root / "pyproject.toml", wrapper]
        digest = hashlib.sha256()
        for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        self.assertEqual(source_hash(), digest.hexdigest())

    def test_container_metadata_allowlist(self):
        with patch.dict("os.environ", {"ADS_CONTAINER_IMAGE": "rapids:test",
                                    "ADS_CONTAINER_IMAGE_ID": "sha256:abc",
                                    "ADS_CONTAINER_REPODIGESTS": "rapids@sha256:abc",
                                    "UNRELATED_SECRET": "do-not-record"}):
            metadata = base_metadata("cpu")
        self.assertEqual(set(metadata["container"]), {"ADS_CONTAINER_IMAGE", "ADS_CONTAINER_IMAGE_ID", "ADS_CONTAINER_REPODIGESTS"})
        self.assertNotIn("UNRELATED_SECRET", json.dumps(metadata))

    def test_preflight_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preflight.json"
            self.assertEqual(preflight("cpu", path)["status"], "passed")
            with self.assertRaises(FileExistsError):
                preflight("cpu", path)

    def test_invalid_arguments(self):
        for fn, val in [(positive_rows, "0"), (positive_rows, "5000001"),
                        (repeat_count, "2"), (warmup_count, "0"),
                        (bounded_seed, "-1"), (bounded_seed, str(2**32))]:
            with self.assertRaises(Exception): fn(val)
        self.assertEqual(bounded_seed("0"), 0)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "must-not-exist"
            with self.assertRaises(SystemExit):
                main(["run", "--backend", "cpu", "--seed", "-1", "--output", str(output)])
            self.assertFalse(output.exists())
        with self.assertRaises(SystemExit):
            main(["run", "--backend", "cpu", "--rows", "10", "10"])


if __name__ == "__main__":
    unittest.main()
