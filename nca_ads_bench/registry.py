"""Lazy registry: the original suite remains the default run command."""
from .workloads import WORKLOAD_NAMES as ORIGINAL_NAMES

ML_NAMES = ("train_only_standardize_pca", "supervised_linear_regression",
            "supervised_logistic_classification", "unsupervised_kmeans")
DATA_NAMES = ("eda_summary_stats", "chronological_rolling_mean", "directed_graph_degree")
REMAINING_NAMES = DATA_NAMES + ML_NAMES
ALL_NAMES = ORIGINAL_NAMES + REMAINING_NAMES


def module_for(name):
    if name in ML_NAMES:
        from . import ml_workloads
        return ml_workloads
    if name in DATA_NAMES:
        from . import data_workloads
        return data_workloads
    raise ValueError(f"Unknown additional workload: {name}")


def make_selected(name, rows, seed, gpu):
    if name in ORIGINAL_NAMES:
        from .workloads import make_cases
        return next(case for case in make_cases(rows, seed, gpu) if case.name == name)
    return module_for(name).make_case(name, rows, seed, gpu)


def dependency_versions(name, gpu):
    if name in ORIGINAL_NAMES:
        return {}
    module = module_for(name)
    return module.dependency_versions(gpu, name=name) if name in DATA_NAMES else module.dependency_versions(gpu)
