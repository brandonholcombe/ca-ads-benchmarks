"""Bounded M05/M09 data workloads; GPU libraries are loaded only for GPU cases."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd

from .workloads import Case


WORKLOAD_NAMES = ("eda_summary_stats", "chronological_rolling_mean", "directed_graph_degree")
HISTOGRAM_BINS = 32
ROLLING_OBSERVATIONS = 8
ROLLING_MIN_PERIODS = 3
MIN_GRAPH_EDGES = 100
MAX_GRAPH_EDGES = 100_000


def histogram_input(rows: int, seed: int) -> np.ndarray:
    if rows < 1:
        raise ValueError("rows must be positive")
    values = np.random.default_rng(seed + 101).normal(0.5, 0.2, rows).astype(np.float32)
    values[::29] = np.nan
    values[::43] = -0.25
    values[::47] = 1.25
    return values


def histogram_compute(values, xp):
    # The fixed range deliberately excludes finite outliers; NaNs are ignored.
    return xp.histogram(values[xp.isfinite(values)], bins=HISTOGRAM_BINS, range=(0.0, 1.0))


def compare_histogram(actual, expected) -> dict:
    counts, edges = (np.asarray(x) for x in actual)
    want_counts, want_edges = (np.asarray(x) for x in expected)
    shape_ok = counts.shape == want_counts.shape == (HISTOGRAM_BINS,) and edges.shape == want_edges.shape == (HISTOGRAM_BINS + 1,)
    integral = np.issubdtype(counts.dtype, np.integer)
    finite_counts = np.issubdtype(counts.dtype, np.number) and bool(np.all(np.isfinite(counts)))
    finite_edges = np.issubdtype(edges.dtype, np.number) and bool(np.all(np.isfinite(edges)))
    counts_ok = shape_ok and integral and np.array_equal(counts, want_counts)
    edges_ok = shape_ok and finite_edges and np.allclose(edges, want_edges, rtol=0, atol=1e-7)
    return {"passed": bool(counts_ok and edges_ok), "counts_match": bool(counts_ok),
            "edges_match": bool(edges_ok), "actual_total": int(counts.sum()) if finite_counts else None,
            "expected_total": int(want_counts.sum()),
            "actual_counts": counts.astype(int).tolist() if finite_counts and integral else None,
            "expected_counts": want_counts.astype(int).tolist(),
            "comparison": "all 32 integer bin counts exactly; all 33 fixed bin edges to 1e-7"}


def rolling_input(rows: int, seed: int) -> pd.DataFrame:
    if rows < 1:
        raise ValueError("rows must be positive")
    rng = np.random.default_rng(seed + 102)
    minute = np.cumsum(rng.integers(1, 4, rows, dtype=np.int64))
    values = rng.normal(10.0, 2.0, rows).astype(np.float64)
    values[::11] = np.nan
    order = rng.permutation(rows)
    return pd.DataFrame({"minute": minute[order], "value": values[order]})


def rolling_compute(frame):
    ordered = frame.sort_values("minute").reset_index(drop=True)
    mean = ordered["value"].rolling(window=ROLLING_OBSERVATIONS,
                                     min_periods=ROLLING_MIN_PERIODS,
                                     center=False).mean()
    return ordered[["minute"]].assign(rolling_mean=mean)


def compare_rolling(actual, expected) -> dict:
    got_minute = np.asarray(actual["minute"])
    want_minute = np.asarray(expected["minute"])
    got = np.asarray(actual["rolling_mean"], dtype=np.float64)
    want = np.asarray(expected["rolling_mean"], dtype=np.float64)
    times_ok = np.array_equal(got_minute, want_minute)
    missing_ok = np.array_equal(np.isnan(got), np.isnan(want))
    values_ok = got.shape == want.shape and np.allclose(got, want, rtol=1e-10, atol=1e-10, equal_nan=True)
    return {"passed": bool(times_ok and missing_ok and values_ok),
            "timestamps_match": bool(times_ok), "missing_mask_match": bool(missing_ok),
            "values_match": bool(values_ok), "actual_rows": len(got),
            "expected_rows": len(want), "rtol": 1e-10, "atol": 1e-10,
            "comparison": "all sorted time keys, null positions, and trailing rolling means"}


def graph_input(rows: int, seed: int) -> pd.DataFrame:
    """Unique, loop-free directed edges; a final sink vertex has zero out-degree."""
    if not MIN_GRAPH_EDGES <= rows <= MAX_GRAPH_EDGES:
        raise ValueError(f"directed graph rows must be {MIN_GRAPH_EDGES}..{MAX_GRAPH_EDGES}")
    count = rows
    # Keep vertex IDs contiguous so cuGraph can disable internal renumbering.
    vertices = min(max(8, (count + 3) // 4 + 1), count + 1)
    i = np.arange(count, dtype=np.int64)
    src = (i % (vertices - 1)).astype(np.int32)
    offset = 1 + i // (vertices - 1)
    dst = ((src + offset) % (vertices - 1)).astype(np.int32)
    # One incoming edge goes to a sink; replacing rather than adding preserves the edge cap.
    dst[-1] = vertices - 1
    # IDs and edge order vary with seed but topology remains duplicate-free.
    order = np.random.default_rng(seed + 103).permutation(count)
    return pd.DataFrame({"src": src[order], "dst": dst[order]})


def graph_cpu(edges: pd.DataFrame) -> pd.DataFrame:
    vertices = np.union1d(edges["src"].to_numpy(), edges["dst"].to_numpy())
    incoming = edges.groupby("dst").size()
    outgoing = edges.groupby("src").size()
    return pd.DataFrame({"vertex": vertices.astype(np.int32),
                         "in_degree": incoming.reindex(vertices, fill_value=0).to_numpy(dtype=np.int64),
                         "out_degree": outgoing.reindex(vertices, fill_value=0).to_numpy(dtype=np.int64)})


def graph_gpu(edges, cugraph):
    graph = cugraph.Graph(directed=True)
    graph.from_cudf_edgelist(edges, source="src", destination="dst", renumber=False)
    return graph.degrees()


def compare_graph(actual, expected) -> dict:
    columns = ("vertex", "in_degree", "out_degree")
    got = actual.loc[:, columns].sort_values("vertex").reset_index(drop=True)
    want = expected.loc[:, columns].sort_values("vertex").reset_index(drop=True)
    ids_ok = np.array_equal(got["vertex"].to_numpy(), want["vertex"].to_numpy())
    in_ok = np.array_equal(got["in_degree"].to_numpy(), want["in_degree"].to_numpy())
    out_ok = np.array_equal(got["out_degree"].to_numpy(), want["out_degree"].to_numpy())
    return {"passed": bool(ids_ok and in_ok and out_ok), "vertex_ids_match": bool(ids_ok),
            "in_degree_match": bool(in_ok), "out_degree_match": bool(out_ok),
            "actual_vertices": len(got), "expected_vertices": len(want),
            "actual_in_total": int(got["in_degree"].sum()),
            "expected_in_total": int(want["in_degree"].sum()),
            "actual_out_total": int(got["out_degree"].sum()),
            "expected_out_total": int(want["out_degree"].sum()),
            "comparison": "all vertex IDs and directed in/out degrees exactly, independent of output row order"}


def dependency_versions(gpu: bool, name: str | None = None) -> dict[str, str]:
    if name is not None and name not in WORKLOAD_NAMES:
        raise ValueError(f"Unknown data workload: {name}")
    versions = {"numpy": np.__version__, "pandas": pd.__version__}
    if gpu:
        dependencies = ("cupy", "cudf")
        if name is None or name == WORKLOAD_NAMES[2]:
            dependencies += ("cugraph",)
        for dependency in dependencies:
            module = import_module(dependency)  # Missing GPU dependencies fail visibly.
            versions[dependency] = module.__version__
    return versions


def make_case(name: str, rows: int, seed: int, gpu=None) -> Case:
    if name not in WORKLOAD_NAMES:
        raise ValueError(f"Unknown data workload: {name}")
    if not 100 <= rows <= 100_000:
        raise ValueError("data rows must be between 100 and 100000")

    def missing(*_):
        raise RuntimeError("GPU backend unavailable")

    if name == WORKLOAD_NAMES[0]:
        values = histogram_input(rows, seed)
        if gpu is None:
            return Case(name, values, lambda x: histogram_compute(x, np), missing, missing, missing, compare_histogram)
        cp, _ = gpu
        return Case(name, values, lambda x: histogram_compute(x, np), cp.asarray,
                    lambda x: histogram_compute(x, cp),
                    lambda pair: (cp.asnumpy(pair[0]), cp.asnumpy(pair[1])), compare_histogram)

    if name == WORKLOAD_NAMES[1]:
        frame = rolling_input(rows, seed)
        if gpu is None:
            return Case(name, frame, rolling_compute, missing, missing, missing, compare_rolling)
        _, cudf = gpu
        return Case(name, frame, rolling_compute,
                    lambda x: cudf.from_pandas(x, nan_as_null=True), rolling_compute,
                    lambda x: x.to_pandas(), compare_rolling)

    edges = graph_input(rows, seed)
    if gpu is None:
        return Case(name, edges, graph_cpu, missing, missing, missing, compare_graph)
    _, cudf = gpu
    cugraph = import_module("cugraph")
    return Case(name, edges, graph_cpu, cudf.from_pandas,
                lambda x: graph_gpu(x, cugraph), lambda x: x.to_pandas(), compare_graph)
