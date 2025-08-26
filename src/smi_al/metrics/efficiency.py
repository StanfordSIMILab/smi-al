def perf_per_minute(delta_perf: float, minutes: float) -> float:
    return delta_perf / max(minutes, 1e-6)
