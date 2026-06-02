"""Temporal smoothing primitives for post-processing."""

from typing import List


def majority_vote_smoothing(
    values: List[bool],
    window_size: int = 5,
    threshold: float = 0.5,
) -> List[bool]:
    """Symmetric sliding-window majority vote.

    At each position t, the True ratio within [t-half, t+half] (out-of-range
    positions count as False) decides the output. Recovers isolated FNs inside
    T-runs and rejects short FP bursts.
    """
    n = len(values)

    if n == 0 or window_size <= 1:
        return list(values)

    half = window_size // 2
    full_window = 2 * half + 1

    result: List[bool] = []
    for t in range(n):
        # Count True frames in the window
        count = 0
        for k in range(t - half, t + half + 1):
            if 0 <= k < n and values[k]:
                count += 1
        ratio = count / full_window
        result.append(ratio >= threshold)
    return result


def dilate_confirmed_runs(
    values: List[bool],
    min_run_length: int = 3,
    dilation: int = 1,
) -> List[bool]:
    """Symmetric dilation applied only to True runs of length >= min_run_length.

    Compensates VLM's conservative entry/exit boundaries: a confirmed True
    segment is extended by `dilation` frames on each side. Short True runs
    (length < min_run_length) are left untouched so residual FP bursts are
    not amplified. Edge clamps at sequence boundaries.
    """
    n = len(values)
    if n == 0 or dilation <= 0 or min_run_length <= 0:
        return list(values)

    result = list(values)
    i = 0
    while i < n:
        if not values[i]:
            i += 1
            continue
        j = i
        while j < n and values[j]:
            j += 1
        # True run is [i, j-1]; length = j - i
        if (j - i) >= min_run_length:
            start = max(0, i - dilation)
            end = min(n, j + dilation)
            for k in range(start, end):
                result[k] = True
        i = j
    return result