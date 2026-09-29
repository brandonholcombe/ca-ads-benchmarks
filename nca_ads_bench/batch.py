"""Sequential, time-bounded child processes; preserve every finished case."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

from .runner import base_metadata


def checkpoint(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _terminate(process):
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def execute(command, log, seconds):
    """Timeout includes imports/data generation/checks, not only timed samples."""
    with log.open("x") as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=(os.name == "posix"))
        try:
            code = child.wait(timeout=seconds)
            return {"status": "passed" if code == 0 else "error", "exit_code": code}
        except subprocess.TimeoutExpired:
            _terminate(child)
            return {"status": "timeout", "error": f"Child exceeded {seconds:.3f} seconds"}
        except BaseException:
            _terminate(child)
            raise


def run_batch(args):
    started = time.monotonic()
    deadline = started + args.max_seconds
    directory = args.output / ("course_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    manifest = directory / "batch.json"
    report = {"schema_version": 1, "kind": "course_batch", "status": "running",
              "metadata": base_metadata(args.backend),
              "parameters": {key: getattr(args, key) for key in
                             ("backend", "workloads", "rows", "repeat", "warmup", "seed", "case_timeout", "max_seconds")},
              "preflight": {"status": "not_run"},
              "cases": [{"name": name, "rows": size, "status": "not_run"}
                        for size in args.rows for name in args.workloads]}
    checkpoint(manifest, report)
    active = None
    command_base = [sys.executable, "-m", "nca_ads_bench"]
    try:
        pre = report["preflight"]
        pre["status"] = "running"
        checkpoint(manifest, report)
        pre_command = command_base + ["preflight", "--backend", args.backend,
                                     "--output", str(directory / "preflight.json")]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pre.update(status="budget_exhausted")
        else:
            pre.update(execute(pre_command, directory / "preflight.log", min(args.case_timeout, remaining)))
        checkpoint(manifest, report)
        if pre["status"] != "passed":
            for entry in report["cases"]:
                entry.update(status="skipped", reason="preflight did not pass")
        else:
            failed_names = set()
            for index, entry in enumerate(report["cases"]):
                active = entry
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    entry.update(status="budget_exhausted", reason="No batch time remains")
                    continue
                if entry["name"] in failed_names:
                    entry.update(status="skipped", reason="Smaller case of this workload did not pass")
                    continue
                relative = f"{index + 1:02d}_{entry['name']}_{entry['rows']}"
                case_directory = directory / relative
                case_directory.mkdir()
                entry.update(status="running", directory=relative)
                command = command_base + ["run", "--backend", args.backend,
                          "--workload", entry["name"], "--rows", str(entry["rows"]),
                          "--repeat", str(args.repeat), "--warmup", str(args.warmup),
                          "--seed", str(args.seed), "--output", str(case_directory)]
                entry["command"] = command
                checkpoint(manifest, report)
                print(f"[{index + 1}/{len(report['cases'])}] {entry['name']} rows={entry['rows']}", flush=True)
                begin = time.monotonic()
                entry.update(execute(command, case_directory / "process.log", min(args.case_timeout, remaining)))
                entry["elapsed_seconds"] = round(time.monotonic() - begin, 3)
                artifacts = list(case_directory.glob("*/results.json"))
                if artifacts:
                    artifact = artifacts[0]
                    entry["results_json"] = str(artifact.relative_to(directory))
                    try:
                        payload = json.loads(artifact.read_text())
                        if not isinstance(payload, dict):
                            raise ValueError("Child report must be a JSON object")
                        if entry["status"] == "passed" and payload.get("status") != "passed":
                            entry.update(status="error", error="Child report did not pass")
                    except (OSError, ValueError) as exc:
                        entry["report_error"] = f"Unreadable or partial child report: {exc}"
                        if entry["status"] == "passed":
                            entry["status"] = "error"
                elif entry["status"] == "passed":
                    entry.update(status="error", error="Child exited without a results report")
                if entry["status"] != "passed":
                    failed_names.add(entry["name"])
                print(f"  {entry['status']} ({entry['elapsed_seconds']:.1f}s)", flush=True)
                checkpoint(manifest, report)
        report["status"] = "passed" if all(e["status"] == "passed" for e in report["cases"]) else "incomplete"
    except KeyboardInterrupt:
        if active is not None and active["status"] == "running":
            active.update(status="interrupted")
        elif report["preflight"]["status"] == "running":
            report["preflight"]["status"] = "interrupted"
        report["status"] = "interrupted"
    except Exception as exc:
        report.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
        if active is not None and active["status"] == "running":
            active["status"] = "error"
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["metadata"]["utc_end"] = datetime.now(timezone.utc).isoformat()
    checkpoint(manifest, report)
    lines = [f"# Course batch: {report['status']}", "", f"Elapsed seconds: {report['elapsed_seconds']}", "",
             "| Workload | Rows | Status | Reports/logs |", "|---|---:|---|---|"]
    for entry in report["cases"]:
        link = f"[{entry['directory']}]({entry['directory']})" if "directory" in entry else entry.get("reason", "")
        lines.append(f"| {entry['name']} | {entry['rows']} | {entry['status']} | {link} |")
    lines += ["", "Only passed case reports contain usable timing evidence. Review correctness, environment and boundaries before course claims.",
              "Each finished child report is preserved. A killed child may have only its process log and manifest timeout status.",
              "The time budget stops benchmark work; it does not stop the cloud instance or its billing.", ""]
    (directory / "summary.md").write_text("\n".join(lines))
    return directory, report
