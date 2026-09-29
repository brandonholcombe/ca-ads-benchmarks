"""Deterministic host inputs, CPU references, and lazy optional GPU operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WORKLOAD_NAMES = ("array_three_transforms", "numeric_cleanup_groupby", "unique_lookup_join_groupby")


@dataclass
class Case:
    name: str
    host_input: object
    cpu: object
    gpu_prepare: object
    gpu_compute: object
    gpu_to_host: object
    compare: object


def array_input(rows: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.01, 10.0, rows).astype(np.float32)


def cleanup_input(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    qty = rng.integers(1, 12, rows).astype(np.float64)
    qty[::17] = np.nan
    qty[::29] = -1
    qty[::41] = 2.5
    return pd.DataFrame({
        "region": rng.integers(0, 8, rows, dtype=np.int32),
        "quantity": qty,
        "unit_cents": rng.integers(100, 5000, rows, dtype=np.int64),
    })


def join_input(rows: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 2)
    keys = max(2, min(rows, rows // 4))
    orders = pd.DataFrame({
        "customer_id": rng.integers(0, keys, rows, dtype=np.int32),
        "amount_cents": rng.integers(100, 5000, rows, dtype=np.int64),
    })
    lookup = pd.DataFrame({
        "customer_id": np.arange(keys, dtype=np.int32),
        "region": np.arange(keys, dtype=np.int32) % 8,
    })
    return orders, lookup


def array_compute(x, xp):
    y = xp.sqrt(x * x + xp.float32(1.0))
    y = xp.log1p(y)
    return y / (y + xp.float32(1.0))


def cleanup_compute(frame):
    q = frame["quantity"]
    valid = q.notna() & (q > 0) & ((q % 1) == 0)
    clean = frame.loc[valid, ["region", "quantity", "unit_cents"]].copy()
    clean["quantity"] = clean["quantity"].astype("int64")
    clean["sales_cents"] = clean["quantity"] * clean["unit_cents"]
    return clean.groupby("region")["sales_cents"].sum()


def join_compute(pair):
    left, right = pair
    joined = left.merge(right, on="customer_id", how="inner")
    totals = joined.groupby("region")["amount_cents"].sum()
    return joined, totals


def canonical_series(value) -> dict[int, int]:
    return {int(k): int(v) for k, v in value.items()}


def compare_array(actual, expected) -> dict:
    actual = np.asarray(actual)
    numeric = np.issubdtype(actual.dtype, np.number)
    nonfinite_count = int(np.size(actual) - np.count_nonzero(np.isfinite(actual))) if numeric else None
    finite = numeric and nonfinite_count == 0
    passed = (actual.dtype == np.dtype("float32") and actual.shape == expected.shape
              and finite and np.allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=False))
    max_abs = None
    if actual.shape == expected.shape and finite:
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        if np.all(np.isfinite(difference)):
            max_abs = float(np.max(difference))
    return {"passed": bool(passed), "dtype": str(actual.dtype), "expected_dtype": "float32",
            "rtol": 1e-5, "atol": 1e-6, "max_abs_error": max_abs,
            "all_finite": bool(finite), "nonfinite_count": nonfinite_count}


def compare_cleanup(actual, expected) -> dict:
    got, want = canonical_series(actual), canonical_series(expected)
    dtype_ok = str(actual.dtype) == "int64"
    return {"passed": got == want and dtype_ok, "actual": got, "expected": want,
            "actual_dtype": str(actual.dtype), "expected_dtype": "int64",
            "value_type": "int64 cents", "comparison": "exact integer counts and sums"}


def compare_join(actual, expected) -> dict:
    got_join, got_totals = actual
    want_join, want_totals = expected
    got, want = canonical_series(got_totals), canonical_series(want_totals)
    dtype_ok = str(got_totals.dtype) == "int64"
    row_multiset_match = False
    if list(got_join.columns) == list(want_join.columns) and len(got_join) == len(want_join):
        columns = list(want_join.columns)
        actual_sorted = got_join.sort_values(columns, kind="mergesort").reset_index(drop=True)
        expected_sorted = want_join.sort_values(columns, kind="mergesort").reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(actual_sorted, expected_sorted, check_dtype=True,
                                          check_exact=True, check_names=True)
            row_multiset_match = True
        except AssertionError:
            pass
    passed = row_multiset_match and got == want and dtype_ok
    return {"passed": passed, "actual_rows": len(got_join), "expected_rows": len(want_join),
            "actual_totals": got, "expected_totals": want,
            "actual_dtype": str(got_totals.dtype), "expected_dtype": "int64",
            "row_multiset_match": row_multiset_match,
            "comparison": "sorted full joined rows with duplicate multiplicity and dtypes, exact integer sums"}


def make_cases(rows: int, seed: int, gpu=None) -> list[Case]:
    arr = array_input(rows, seed)
    clean = cleanup_input(rows, seed)
    join = join_input(rows, seed)
    if not join[1]["customer_id"].is_unique:
        raise AssertionError("right join key must be unique")

    def missing(*_):
        raise RuntimeError("GPU backend unavailable")

    if gpu is None:
        prepare_array = compute_array = host_array = missing
        prepare_df = compute_df = host_df = missing
        prepare_join = compute_join = host_join = missing
    else:
        cp, cudf = gpu
        prepare_array = cp.asarray
        compute_array = lambda x: array_compute(x, cp)
        host_array = cp.asnumpy
        # Explicitly map pandas NaN quantities to cuDF nulls so both filters
        # quarantine the same rows even if a cuDF release changes its default.
        prepare_df = lambda frame: cudf.from_pandas(frame, nan_as_null=True)
        compute_df = cleanup_compute
        host_df = lambda x: x.to_pandas()
        prepare_join = lambda pair: (cudf.from_pandas(pair[0]), cudf.from_pandas(pair[1]))
        compute_join = join_compute
        host_join = lambda x: (x[0].to_pandas(), x[1].to_pandas())
    return [
        Case(WORKLOAD_NAMES[0], arr, lambda x: array_compute(x, np),
             prepare_array, compute_array, host_array, compare_array),
        Case(WORKLOAD_NAMES[1], clean, cleanup_compute,
             prepare_df, compute_df, host_df, compare_cleanup),
        Case(WORKLOAD_NAMES[2], join, join_compute,
             prepare_join, compute_join, host_join, compare_join),
    ]


def fixture_checks(cudf=None) -> dict:
    """The M02 smoke and M03 cardinality fixtures run before timing."""
    left = pd.DataFrame({"order_id": ["O1", "O2", "O3", "O4"],
                         "customer_id": ["C1", "C1", "C2", "C3"],
                         "amount": [10, 20, 30, 40]})
    right = pd.DataFrame({"customer_id": ["C1", "C2", "C3", "C4"],
                          "segment": ["Retail", "Enterprise", "Retail", "Public"]})
    duplicated = pd.concat([right, pd.DataFrame({"customer_id": ["C1"],
                                                 "segment": ["Wholesale"]})], ignore_index=True)
    if cudf is None:
        got4 = left.merge(right, on="customer_id", how="inner")
        got6 = left.merge(duplicated, on="customer_id", how="inner")
        smoke = None
    else:
        smoke = int(cudf.Series([2, 4, 6], dtype="int64").sum())
        got4 = cudf.from_pandas(left).merge(cudf.from_pandas(right), on="customer_id", how="inner").to_pandas()
        got6 = cudf.from_pandas(left).merge(cudf.from_pandas(duplicated), on="customer_id", how="inner").to_pandas()
    def pairs(frame):
        return sorted(tuple(row) for row in frame[["order_id", "customer_id", "segment"]].itertuples(index=False, name=None))
    expected4 = pairs(left.merge(right, on="customer_id", how="inner"))
    expected6 = pairs(left.merge(duplicated, on="customer_id", how="inner"))
    passed = (smoke in (None, 12) and len(got4) == 4 and len(got6) == 6
              and pairs(got4) == expected4 and pairs(got6) == expected6)
    if not passed:
        raise AssertionError("M02 sum or M03 four/six-row fixture failed")
    return {"passed": True, "cudf_sum": smoke, "unique_lookup_rows": 4,
            "duplicate_lookup_rows": 6, "compared_sorted_pairs_with_multiplicity": True}
