import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from nca_ads_bench.batch import execute, run_batch
from nca_ads_bench.cli import main
from nca_ads_bench.runner import run
from nca_ads_bench.workloads import make_cases


class BatchTests(unittest.TestCase):
    def args(self, output):
        return argparse.Namespace(backend="cpu", workloads=["array_three_transforms", "numeric_cleanup_groupby"],
            rows=[100, 200], repeat=3, warmup=1, seed=9, case_timeout=2, max_seconds=30, output=Path(output))

    def fake_execute(self, fail=None):
        def execute_fake(command, log, seconds):
            log.write_text("test child log\n")
            if "--workload" not in command:
                return {"status": "passed", "exit_code": 0}
            name = command[command.index("--workload") + 1]
            output = Path(command[command.index("--output") + 1]) / "unique"
            output.mkdir()
            status = "error" if name == fail else "passed"
            (output / "results.json").write_text(json.dumps({"status": status}))
            return {"status": status, "exit_code": int(status != "passed")}
        return execute_fake

    def test_failure_preserves_passed_and_skips_larger_failed_only(self):
        with tempfile.TemporaryDirectory() as tmp, patch("nca_ads_bench.batch.execute", self.fake_execute("numeric_cleanup_groupby")):
            directory, report = run_batch(self.args(tmp))
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual([x["status"] for x in report["cases"]], ["passed", "error", "passed", "skipped"])
            saved = json.loads((directory / "batch.json").read_text())
            self.assertEqual(saved["cases"], report["cases"])
            self.assertTrue((directory / saved["cases"][0]["results_json"]).exists())
            self.assertTrue((directory / "summary.md").exists())

    def test_new_batch_does_not_overwrite_old(self):
        with tempfile.TemporaryDirectory() as tmp, patch("nca_ads_bench.batch.execute", self.fake_execute()):
            first, _ = run_batch(self.args(tmp))
            before = (first / "batch.json").read_bytes()
            second, report = run_batch(self.args(tmp))
            self.assertEqual(report["status"], "passed")
            self.assertNotEqual(first, second)
            self.assertEqual(before, (first / "batch.json").read_bytes())

    def test_partial_timeout_report_does_not_stop_independent_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            delegate = self.fake_execute()
            def execute_partial(command, log, seconds):
                result = delegate(command, log, seconds)
                if "--workload" in command and command[command.index("--workload") + 1] == "numeric_cleanup_groupby":
                    output = Path(command[command.index("--output") + 1]) / "unique" / "results.json"
                    output.write_text('{"status":')
                    return {"status": "timeout", "error": "timed out"}
                return result
            with patch("nca_ads_bench.batch.execute", execute_partial):
                _, report = run_batch(self.args(tmp))
            self.assertEqual([e["status"] for e in report["cases"]], ["passed", "timeout", "passed", "skipped"])
            self.assertIn("partial", report["cases"][1]["report_error"])

    def test_missing_gpu_stops_batch_no_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, patch("nca_ads_bench.batch.execute", return_value={"status": "error", "exit_code": 1}) as mocked:
            args = self.args(tmp)
            args.backend = "gpu"
            _, report = run_batch(args)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(report["status"], "incomplete")
            self.assertTrue(all(x["status"] == "skipped" for x in report["cases"]))

    def test_expired_budget_launches_no_more_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = [0.0]
            def execute_fake(*_):
                clock[0] = 31.0
                return {"status": "passed", "exit_code": 0}
            with patch("nca_ads_bench.batch.time.monotonic", side_effect=lambda: clock[0]), patch("nca_ads_bench.batch.execute", side_effect=execute_fake) as mocked:
                _, report = run_batch(self.args(tmp))
            self.assertEqual(mocked.call_count, 1)
            self.assertTrue(all(x["status"] == "budget_exhausted" for x in report["cases"]))
            self.assertEqual(report["status"], "incomplete")

    def test_real_subprocess_timeout_preserves_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "process.log"
            result = execute([sys.executable, "-u", "-c", "import time; print('started'); time.sleep(30)"], log, 0.4)
            self.assertEqual(result["status"], "timeout")
            self.assertIn("started", log.read_text())

    def test_success_without_report_is_error(self):
        with tempfile.TemporaryDirectory() as tmp, patch("nca_ads_bench.batch.execute", return_value={"status": "passed", "exit_code": 0}):
            _, report = run_batch(self.args(tmp))
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("without a results", report["cases"][0]["error"])

    def test_selected_workload_records_untimed_audit(self):
        case = make_cases(100, 9)[0]
        case.audit = lambda: {"passed": True, "round_trip": "checked"}
        with tempfile.TemporaryDirectory() as tmp, patch("nca_ads_bench.registry.make_selected", return_value=case):
            _, report = run("cpu", [100], 3, 1, 9, Path(tmp), case.name)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(len(report["results"]), 1)
            self.assertEqual(report["results"][0]["audit"]["round_trip"], "checked")

    def test_failed_audit_blocks_timing(self):
        case = make_cases(100, 9)[0]
        case.audit = lambda: {"passed": False}
        with tempfile.TemporaryDirectory() as tmp, patch("nca_ads_bench.registry.make_selected", return_value=case), patch("nca_ads_bench.runner._measure_case") as measure:
            _, report = run("cpu", [100], 3, 1, 9, Path(tmp), case.name)
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["error"]["phase"], "artifact_audit")
            measure.assert_not_called()

    def test_batch_argument_bounds(self):
        for options in (["--rows", "200", "100"], ["--rows", "100001"], ["--max-seconds", "0"],
                        ["--workloads", "eda_summary_stats", "eda_summary_stats"]):
            with self.assertRaises(SystemExit):
                main(["course-batch", "--backend", "cpu"] + options)
