import numpy as np
from typing import Dict, List

GRID_SIZE = 10
N_NEURONS = GRID_SIZE * GRID_SIZE

RAW_PATTERNS: Dict[str, List[List[int]]] = {
    "К": [
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    ],
    "О": [
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    ],
    "Л": [
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    ],
}

NOISE_LEVELS: List[int] = [10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]


def get_bipolar_patterns() -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for name, grid in RAW_PATTERNS.items():
        arr = np.array(grid, dtype=int).flatten()
        result[name] = np.where(arr == 1, 1, -1)
    return result


def apply_noise(vector: np.ndarray, noise_percent: float, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    noisy = vector.copy()
    n_total = len(noisy)
    n_flip = int(round((noise_percent / 100.0) * 0.55 * n_total))
    n_flip = max(0, min(n_total, n_flip))

    if n_flip > 0:
        flip_indices = rng.choice(n_total, size=n_flip, replace=False)
        noisy[flip_indices] *= -1
    return noisy


def hamming_distance(v1: np.ndarray, v2: np.ndarray) -> int:
    return int(np.sum(v1 != v2))


def pattern_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.mean(v1 == v2) * 100.0)