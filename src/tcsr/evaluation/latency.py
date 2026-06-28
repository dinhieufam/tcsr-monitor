"""Per-frame monitor latency benchmark."""

from __future__ import annotations

import time

import numpy as np


def benchmark_monitor(monitor, X: np.ndarray, n_warmup: int = 10, n_runs: int = 100) -> dict:
    """
    Measure wall-clock inference latency of monitor.predict_proba(x).
    Reports per-frame latency in milliseconds.
    """
    # Warmup
    for _ in range(n_warmup):
        monitor.predict_proba(X[:1])

    times_ms = []
    for _ in range(n_runs):
        start = time.perf_counter()
        monitor.predict_proba(X[:1])
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)

    arr = np.array(times_ms)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "std_ms": float(arr.std()),
        "n_runs": n_runs,
    }
