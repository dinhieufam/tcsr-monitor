"""Warning lead-time analysis: frames/seconds before sustained failure onset."""

from __future__ import annotations

import numpy as np


def compute_lead_times(
    alarms: np.ndarray,
    labels: np.ndarray,
    video_ids: np.ndarray,
    fps: float = 25.0,
    min_failure_run: int = 3,
) -> dict:
    """
    For each sustained failure run (>= min_failure_run consecutive failures),
    compute the number of frames (and seconds) between the first alarm and
    the start of the failure run.

    Negative lead time means the alarm came AFTER the failure started.
    """
    lead_frames = []
    unique_videos = np.unique(video_ids)

    for vid in unique_videos:
        mask = video_ids == vid
        vid_labels = labels[mask]
        vid_alarms = alarms[mask]

        # Find starts of sustained failure runs
        failure_runs = _find_runs(vid_labels, value=1, min_len=min_failure_run)
        for start, _ in failure_runs:
            # Look for the most recent alarm before this failure start
            alarm_indices = np.where(vid_alarms[:start])[0]
            if len(alarm_indices) == 0:
                continue
            last_alarm = alarm_indices[-1]
            lead = start - last_alarm
            lead_frames.append(lead)

    if not lead_frames:
        return {"mean_lead_frames": float("nan"), "mean_lead_seconds": float("nan"), "n_events": 0}

    lead_frames_arr = np.array(lead_frames)
    return {
        "mean_lead_frames": float(lead_frames_arr.mean()),
        "mean_lead_seconds": float(lead_frames_arr.mean() / fps),
        "median_lead_frames": float(np.median(lead_frames_arr)),
        "n_events": len(lead_frames),
        "frac_early_warning": float((lead_frames_arr > 0).mean()),
    }


def _find_runs(arr: np.ndarray, value: int, min_len: int) -> list[tuple[int, int]]:
    runs = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == value:
            j = i
            while j < n and arr[j] == value:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs
