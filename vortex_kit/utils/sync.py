"""
vortex_kit.utils.sync
======================
Synchronization utilities for asynchronous data.

Functions for refresh-time synchronization of multiple price series.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _refresh_time_core(
    timestamps_list: list[np.ndarray],
    values_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Refresh-time synchronization core — returns synchronized timestamps
    and values as a 2-D array.
    """
    k = len(timestamps_list)
    # Get initial pointers
    ptrs = np.zeros(k, dtype=np.int64)
    n_obs = np.array([len(ts) for ts in timestamps_list], dtype=np.int64)

    sync_ts = []
    sync_vals = []

    while True:
        # Find current max timestamp across all series
        current_max = -np.inf
        for i in range(k):
            if ptrs[i] < n_obs[i]:
                if timestamps_list[i][ptrs[i]] > current_max:
                    current_max = timestamps_list[i][ptrs[i]]

        if current_max == -np.inf:
            break  # All series exhausted

        # Collect values at this refresh time
        row_vals = np.zeros(k, dtype=np.float64)
        valid = True
        for i in range(k):
            # Move pointer forward until we have timestamp >= current_max
            while ptrs[i] < n_obs[i] and timestamps_list[i][ptrs[i]] < current_max:
                ptrs[i] += 1

            if ptrs[i] >= n_obs[i]:
                valid = False
                break

            # Use the last observation at or before current_max
            if ptrs[i] > 0:
                row_vals[i] = values_list[i][ptrs[i] - 1]
            else:
                row_vals[i] = values_list[i][0]

        if valid:
            sync_ts.append(current_max)
            sync_vals.append(row_vals.copy())

        # Advance all pointers that are at current_max
        for i in range(k):
            if ptrs[i] < n_obs[i] and timestamps_list[i][ptrs[i]] == current_max:
                ptrs[i] += 1

    if len(sync_ts) == 0:
        return np.array([], dtype=np.float64), np.zeros((0, k), dtype=np.float64)

    return np.array(sync_ts, dtype=np.float64), np.array(sync_vals, dtype=np.float64)


def refresh_time(
    timestamps_list: list[np.ndarray],
    values_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Refresh-time synchronisation of multiple asynchronous series.

    For each refresh time (the max of current timestamps across all series),
    take the most recent observation from each series.

    Parameters
    ----------
    timestamps_list : list of ndarray
        List of K sorted timestamp arrays (one per asset).
    values_list : list of ndarray
        List of K value arrays corresponding to timestamps.

    Returns
    -------
    tuple
        (sync_timestamps, sync_values) where sync_values is shape (N, K).
    """
    return _refresh_time_core(timestamps_list, values_list)
