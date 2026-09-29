"""Execution, timing boundaries, hardware inventory, and durable reports."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd

from . import __version__
from .workloads import WORKLOAD_NAMES, fixture_checks, make_cases


class BenchmarkError(RuntimeError):
    pass


class CorrectnessError(BenchmarkError):
    def __init__(self, comparison: dict):
        self.comparison = comparison
        super().__init__(f"Correctness failed in {comparison['path']}, repeat {comparison['repeat']}")


def source_hash() -> str:
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    files = list((root / "nca_ads_bench").glob("*.py"))
    files += [root / "pyproject.toml", root / "environments" / "run-in-container.sh"]
    for file in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def command_output(args: list[str], timeout: int = 5) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_info() -> dict:
    root = Path(__file__).resolve().parent.parent
    commit = command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    dirty = command_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"])
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None,
            "source_sha256": source_hash()}


def cpu_environment() -> dict:
    model = platform.processor() or None
    if sys.platform == "linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith(("model name", "Hardware")) and ":" in line:
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    elif sys.platform == "darwin":
        model = command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or model
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity_count = None
    except OSError:
        affinity_count = None
    quota_cores = None
    if sys.platform == "linux":
        try:
            quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
            if quota != "max" and int(period) > 0:
                quota_cores = int(quota) / int(period)
        except (OSError, ValueError, IndexError):
            try:
                quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
                period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
                if quota > 0 and period > 0:
                    quota_cores = quota / period
            except (OSError, ValueError):
                pass
    return {"model": model, "affinity_cpu_count": affinity_count,
            "cgroup_cpu_quota_cores": quota_cores}


def base_metadata(backend: str) -> dict:
    thread_names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    container_names = ("ADS_CONTAINER_IMAGE", "ADS_CONTAINER_IMAGE_ID", "ADS_CONTAINER_REPODIGESTS")
    return {"harness_version": __version__, "backend_requested": backend,
            "utc_start": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "cpu": cpu_environment(),
            "thread_environment": {k: os.environ[k] for k in thread_names if k in os.environ},
            "container": {k: os.environ[k] for k in container_names if k in os.environ},
            "git": git_info()}


def gpu_modules() -> tuple:
    try:
        import cupy as cp
        import cudf
        count = cp.cuda.runtime.getDeviceCount()
        if count < 1:
            raise BenchmarkError("No CUDA device visible to CuPy")
        cp.cuda.Device(0).use()
        cp.cuda.runtime.deviceSynchronize()
        return cp, cudf
    except Exception as exc:
        raise BenchmarkError(f"GPU preflight failed: {type(exc).__name__}: {exc}") from exc


def gpu_metadata(cp, cudf) -> dict:
    device = cp.cuda.runtime.getDeviceProperties(0)
    def decode(v):
        return v.decode(errors="replace") if isinstance(v, bytes) else str(v)
    free, total = cp.cuda.runtime.memGetInfo()
    try:
        bus_id = decode(cp.cuda.runtime.deviceGetPCIBusId(0))
    except Exception:
        bus_id = None
    smi = command_output(["nvidia-smi", "-i", bus_id,
                          "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw",
                          "--format=csv,noheader,nounits"]) if bus_id else None
    # CUDA ordinal zero can map to another physical GPU under CUDA_VISIBLE_DEVICES.
    # A PCI bus ID scopes nvidia-smi to the actual selected device; absence is reported as unknown.
    return {"device_ordinal": 0, "name": decode(device["name"]),
            "pci_bus_id": bus_id,
            "compute_capability": f"{device['major']}.{device['minor']}",
            "driver_version_cuda_api": cp.cuda.runtime.driverGetVersion(),
            "runtime_version_cuda_api": cp.cuda.runtime.runtimeGetVersion(),
            "total_memory_bytes": int(total), "free_memory_bytes_at_preflight": int(free),
            "cupy": cp.__version__, "cudf": cudf.__version__,
            "nvidia_smi_query_header": "index,name,uuid,pci.bus_id,driver_version,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "nvidia_smi_selected_row": smi.strip() if smi else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def sync(cp) -> None:
    cp.cuda.runtime.deviceSynchronize()


def timed(call, sync_fn=None):
    if sync_fn is not None:
        sync_fn()
    start = time.perf_counter_ns()
    value = call()
    if sync_fn is not None:
        sync_fn()
    return value, (time.perf_counter_ns() - start) / 1e6


def describe(samples: list[float]) -> dict:
    if not samples:
        raise ValueError("at least one timing sample required")
    ordered = sorted(samples)
    def quantile(fraction):
        pos = (len(ordered) - 1) * fraction
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return {"samples_ms": samples, "median_ms": statistics.median(samples),
            "q25_ms": quantile(.25), "q75_ms": quantile(.75),
            "min_ms": ordered[0], "max_ms": ordered[-1]}


def _write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def preflight(backend: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    report = {"schema_version": 1, "kind": "preflight", "status": "error", "metadata": base_metadata(backend)}
    try:
        if backend == "gpu":
            cp, cudf = gpu_modules()
            report["metadata"]["gpu"] = gpu_metadata(cp, cudf)
            report["fixtures"] = fixture_checks(cudf)
            sync(cp)
        else:
            report["fixtures"] = fixture_checks()
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(output, report)
    return report


def _measure_case(case, reference, backend, repeat, warmup, cp):
    cpu_samples, resident_samples, end_to_end_samples = [], [], []
    comparisons = []
    if backend == "gpu":
        device_input = case.gpu_prepare(case.host_input)
        sync(cp)
    else:
        device_input = None
    # CPU and GPU warmups are deliberately untimed. GPU transfers are warmed separately.
    for _ in range(warmup):
        case.cpu(case.host_input)
        if cp is not None:
            case.gpu_compute(device_input)
            sync(cp)
            case.gpu_to_host(case.gpu_compute(case.gpu_prepare(case.host_input)))
            sync(cp)
    def checked(path, index, actual):
        comparison = {"path": path, "repeat": index, **case.compare(actual, reference)}
        comparisons.append(comparison)
        if not comparison["passed"]:
            raise CorrectnessError(comparison)
    for index in range(repeat):
        def cpu_trial():
            result, elapsed = timed(lambda: case.cpu(case.host_input))
            cpu_samples.append(elapsed)
            checked("cpu", index, result)
        def gpu_trial():
            resident, elapsed = timed(lambda: case.gpu_compute(device_input), lambda: sync(cp))
            resident_samples.append(elapsed)
            checked("gpu_resident", index, case.gpu_to_host(resident))
            def transfer_pipeline():
                dev = case.gpu_prepare(case.host_input)
                result = case.gpu_compute(dev)
                return case.gpu_to_host(result)
            host, elapsed = timed(transfer_pipeline, lambda: sync(cp))
            end_to_end_samples.append(elapsed)
            checked("gpu_host_to_host", index, host)
        if cp is None:
            cpu_trial()
        elif index % 2 == 0:
            cpu_trial(); gpu_trial()
        else:
            gpu_trial(); cpu_trial()
    result = {"name": case.name, "rows": len(case.host_input[0]) if isinstance(case.host_input, tuple) else len(case.host_input),
              "correctness_passed": True, "comparisons": comparisons,
              "cpu_ms": describe(cpu_samples)}
    if cp is not None:
        result["gpu_resident_ms"] = describe(resident_samples)
        result["gpu_host_to_host_ms"] = describe(end_to_end_samples)
        result["speedup_cpu_over_gpu_resident"] = result["cpu_ms"]["median_ms"] / result["gpu_resident_ms"]["median_ms"]
        result["speedup_cpu_over_gpu_host_to_host"] = result["cpu_ms"]["median_ms"] / result["gpu_host_to_host_ms"]["median_ms"]
    return result


def _write_reports(directory: Path, report: dict) -> None:
    _write_json_exclusive(directory / "results.json", report)
    with (directory / "summary.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rows", "workload", "status", "correctness_passed", "cpu_median_ms", "gpu_resident_median_ms", "gpu_host_to_host_median_ms", "speedup_resident", "speedup_host_to_host", "error_type", "error_message"])
        writer.writeheader()
        for row in report.get("results", []):
            writer.writerow({"rows": row["rows"], "workload": row["name"],
                             "status": row["status"],
                             "correctness_passed": row.get("correctness_passed"),
                             "cpu_median_ms": row.get("cpu_ms", {}).get("median_ms"),
                             "gpu_resident_median_ms": row.get("gpu_resident_ms", {}).get("median_ms"),
                             "gpu_host_to_host_median_ms": row.get("gpu_host_to_host_ms", {}).get("median_ms"),
                             "speedup_resident": row.get("speedup_cpu_over_gpu_resident"),
                             "speedup_host_to_host": row.get("speedup_cpu_over_gpu_host_to_host"),
                             "error_type": row.get("error", {}).get("type"),
                             "error_message": row.get("error", {}).get("message")})
    lines = [f"# NCA-ADS benchmark run: {report['status']}", "",
             f"Backend requested: {report['metadata']['backend_requested']}",
             f"UTC start: {report['metadata']['utc_start']}",
             f"Source SHA-256: {report['metadata']['git']['source_sha256']}", ""]
    if "error" in report:
        lines.append(f"Error: {report['error']['type']}: {report['error']['message']}")
        if "rows" in report["error"]:
            lines.append(f"Failed case: rows={report['error']['rows']}, workload={report['error'].get('workload') or 'input generation'}, phase={report['error']['phase']}")
    if report.get("results"):
        lines += ["| Rows | Workload | Status | Correct | CPU median ms | GPU resident median ms | GPU host-to-host median ms | CPU/GPU host-to-host | Error |",
                  "|---:|---|---|---|---:|---:|---:|---:|---|"]
        for row in report["results"]:
            def fmt(value):
                return f"{value:.3f}" if value is not None else "—"
            correct = row.get("correctness_passed")
            correct_text = str(correct) if correct is not None else "—"
            case_error = row.get("error", {})
            error_text = f"{case_error['type']}: {case_error['message']}" if case_error else ""
            error_text = error_text.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['rows']} | {row['name']} | {row['status']} | {correct_text} | {fmt(row.get('cpu_ms', {}).get('median_ms'))} | "
                         f"{fmt(row.get('gpu_resident_ms', {}).get('median_ms'))} | "
                         f"{fmt(row.get('gpu_host_to_host_ms', {}).get('median_ms'))} | "
                         f"{fmt(row.get('speedup_cpu_over_gpu_host_to_host'))} | {error_text} |")
    lines += ["", "Each GPU resident timing starts with prepared device inputs and ends with device outputs; conversion for correctness is outside timing.",
              "GPU host-to-host includes host-to-device conversion, compute, and device-to-host conversion; input generation and disk I/O are excluded.",
              "Warmups are untimed; CPU/GPU order alternates per repeat. Allocators and caches may reuse memory after warmup.",
              "GPU results, if present, apply only to the recorded device and environment. CPU-only runs do not validate GPU execution.", ""]
    with (directory / "summary.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(backend: str, rows: list[int], repeat: int, warmup: int, seed: int, output: Path) -> tuple[Path, dict]:
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12])
    run_dir.mkdir(exist_ok=False)
    report = {"schema_version": 1, "kind": "run", "status": "error",
              "metadata": base_metadata(backend), "parameters": {"rows": rows, "repeat": repeat, "warmup": warmup, "seed": seed},
              "timing_method": {"unit": "milliseconds", "clock": "time.perf_counter_ns",
                                "gpu_sync": "selected CUDA device before and after each timing",
                                "order": "CPU first on even repeats, GPU first on odd repeats",
                                "warmup": "untimed", "allocator_reuse": "possible after warmup"},
              "results": [{"rows": size, "name": name, "status": "not_run"}
                          for size in rows for name in WORKLOAD_NAMES]}
    planned = {(item["rows"], item["name"]): item for item in report["results"]}
    active = None
    phase = "preflight"
    try:
        cp = cudf = None
        if backend == "gpu":
            cp, cudf = gpu_modules()
            report["metadata"]["gpu"] = gpu_metadata(cp, cudf)
        report["fixtures"] = fixture_checks(cudf)
        for size in rows:
            phase = "input_generation"
            active = {"rows": size, "workload": None}
            for case in make_cases(size, seed, (cp, cudf) if cp is not None else None):
                entry = planned[(size, case.name)]
                active = {"rows": size, "workload": case.name}
                entry["status"] = "running"
                phase = "reference"
                reference = case.cpu(case.host_input)
                phase = "measurement"
                result = _measure_case(case, reference, backend, repeat, warmup, cp)
                entry.update(result)
                entry["status"] = "passed"
        report["status"] = "passed"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "phase": phase}
        if active is not None:
            error.update(active)
            if active["workload"] is not None:
                entry = planned[(active["rows"], active["workload"])]
                entry["status"] = "error"
                entry["error"] = {"type": type(exc).__name__, "message": str(exc), "phase": phase}
                if isinstance(exc, CorrectnessError):
                    entry["correctness_passed"] = False
                    entry["comparisons"] = [exc.comparison]
            # Input generation constructs a size's cases together. A failure
            # there belongs to the row size, not an arbitrarily named workload;
            # all its cases remain not_run and the top-level error has the phase.
        report["error"] = error
    report["metadata"]["utc_end"] = datetime.now(timezone.utc).isoformat()
    _write_reports(run_dir, report)
    return run_dir, report
