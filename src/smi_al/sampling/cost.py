from typing import Dict
import numpy as np

def predict_time_proxy(instance_count: int, boundary_length: float, class_mix_entropy: float) -> float:
    # Simple linear proxy (placeholder coefficients)
    return 0.05 * instance_count + 0.001 * boundary_length + 0.1 * class_mix_entropy + 0.2

def greedy_knapsack(values: np.ndarray, costs: np.ndarray, budget: float) -> np.ndarray:
    # value per cost ratio
    ratio = values / (costs + 1e-8)
    order = np.argsort(-ratio)
    chosen = []
    spent = 0.0
    for i in order:
        if spent + costs[i] <= budget:
            chosen.append(i)
            spent += costs[i]
    return np.array(chosen, dtype=int)
