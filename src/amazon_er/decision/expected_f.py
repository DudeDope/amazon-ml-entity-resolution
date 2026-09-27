"""Exact top-k expected-Fbeta selection under independent pair probabilities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - slower fallback
    def njit(*args, **kwargs):
        def decorator(function):
            return function

        return decorator if not (args and callable(args[0])) else args[0]


@njit(cache=True)
def _poisson_binomial(probability):
    count = probability.shape[0]
    distribution = np.zeros(count + 1)
    distribution[0] = 1.0
    for index in range(count):
        value = probability[index]
        for successes in range(index + 1, 0, -1):
            distribution[successes] = (
                distribution[successes] * (1.0 - value)
                + distribution[successes - 1] * value
            )
        distribution[0] *= 1.0 - value
    return distribution


@njit(cache=True)
def _best_k(probability, extra_miss, beta_squared, allow_empty=True):
    count = probability.shape[0]
    best_k = 0 if allow_empty else 1
    none = 1.0
    for value in probability:
        none *= 1.0 - value
    best_expected = none * np.exp(-extra_miss) if allow_empty else -1.0
    for selected in range(1, count + 1):
        inside = _poisson_binomial(probability[:selected])
        outside = _poisson_binomial(probability[selected:])
        expected = 0.0
        for true_positive in range(1, selected + 1):
            if inside[true_positive] < 1e-12:
                continue
            for false_negative in range(0, count - selected + 1):
                if outside[false_negative] < 1e-12:
                    continue
                fn = false_negative + extra_miss
                denominator = (
                    (1.0 + beta_squared) * true_positive
                    + beta_squared * fn
                    + selected
                    - true_positive
                )
                expected += (
                    inside[true_positive]
                    * outside[false_negative]
                    * ((1.0 + beta_squared) * true_positive / denominator)
                )
        if expected > best_expected:
            best_expected = expected
            best_k = selected
    return best_k, best_expected


@njit(cache=True)
def _select_groups(group_starts, sorted_probability, extra_miss, beta_squared, k_max):
    counts = np.zeros(group_starts.shape[0] - 1, dtype=np.int32)
    for group in range(len(counts)):
        start, stop = group_starts[group], group_starts[group + 1]
        limit = min(stop - start, k_max)
        counts[group] = _best_k(
            sorted_probability[start : start + limit], extra_miss, beta_squared, True
        )[0]
    return counts


def expected_f_select(
    q: np.ndarray,
    probability: np.ndarray,
    extra_miss: float = 0.0,
    beta: float = 0.5,
    k_max: int = 25,
    probability_floor: float = 1e-3,
) -> np.ndarray:
    if len(q) != len(probability):
        raise ValueError("q and probability lengths differ")
    if len(q) == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((-probability, q))
    sorted_q = q[order]
    sorted_probability = probability[order].astype(np.float64)
    sorted_probability[sorted_probability < probability_floor] = 0.0
    starts = np.r_[0, np.flatnonzero(sorted_q[1:] != sorted_q[:-1]) + 1, len(sorted_q)].astype(np.int64)
    counts = _select_groups(starts, sorted_probability, float(extra_miss), beta * beta, int(k_max))
    ranks = np.arange(len(sorted_q)) - np.repeat(starts[:-1], np.diff(starts))
    selected_sorted = ranks < np.repeat(counts, np.diff(starts))
    selected = np.zeros(len(q), dtype=bool)
    selected[order] = selected_sorted
    return selected


@njit(cache=True)
def _select_groups_with_existence(group_starts, probability, existence, extra_miss, beta_squared, k_max):
    counts = np.zeros(len(existence), dtype=np.int32)
    for group in range(len(existence)):
        start, stop = group_starts[group], group_starts[group + 1]
        limit = min(stop - start, k_max)
        if limit == 0 or existence[group] <= 1e-9:
            continue
        conditional = np.empty(limit)
        for index in range(limit):
            conditional[index] = min(probability[start + index] / existence[group], 1.0)
        selected, expected = _best_k(conditional, extra_miss, beta_squared, False)
        if existence[group] * expected > 1.0 - existence[group]:
            counts[group] = selected
    return counts


def pi_select(
    q: np.ndarray,
    probability: np.ndarray,
    existence_by_q: Mapping[object, float],
    extra_miss: float = 0.0,
    beta: float = 0.5,
    k_max: int = 25,
    probability_floor: float = 1e-3,
) -> np.ndarray:
    if len(q) != len(probability):
        raise ValueError("q and probability lengths differ")
    if len(q) == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((-probability, q))
    sorted_q = q[order]
    sorted_probability = probability[order].astype(np.float64)
    sorted_probability[sorted_probability < probability_floor] = 0.0
    starts = np.r_[0, np.flatnonzero(sorted_q[1:] != sorted_q[:-1]) + 1, len(sorted_q)].astype(np.int64)
    unique_q = sorted_q[starts[:-1]]
    existence = np.asarray([existence_by_q.get(value, 0.0) for value in unique_q], dtype=np.float64)
    maximum = np.maximum.reduceat(sorted_probability, starts[:-1]) if len(sorted_probability) else np.zeros(0)
    existence = np.clip(np.maximum(existence, maximum), 1e-9, 1.0)
    counts = _select_groups_with_existence(
        starts, sorted_probability, existence, float(extra_miss), beta * beta, int(k_max)
    )
    ranks = np.arange(len(sorted_q)) - np.repeat(starts[:-1], np.diff(starts))
    selected_sorted = ranks < np.repeat(counts, np.diff(starts))
    selected = np.zeros(len(q), dtype=bool)
    selected[order] = selected_sorted
    return selected
