import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from patterns import NOISE_LEVELS, apply_noise, pattern_similarity


@dataclass
class RecallResult:
    initial_state: np.ndarray
    final_state: np.ndarray
    history: List[np.ndarray]
    iterations: int
    matched_label: Optional[str]
    inverted_label: Optional[str]
    similarities: Dict[str, float]
    converged: bool


class HopfieldNetwork:
    def __init__(self, n_neurons: int = 100):
        self.n = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons), dtype=float)
        self.patterns: Dict[str, np.ndarray] = {}

    def fit(self, patterns: Dict[str, np.ndarray]) -> None:
        self.patterns = {k: v.copy() for k, v in patterns.items()}
        self.weights.fill(0.0)

        for vec in self.patterns.values():
            v = vec.reshape(-1, 1)
            self.weights += v @ v.T

        np.fill_diagonal(self.weights, 0.0)

    @staticmethod
    def activation(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, -1)

    def max_capacity(self) -> float:
        return self.n / (2.0 * math.log(self.n) + math.log(math.log(self.n)))

    def step(
        self,
        state: np.ndarray,
        synchronous: bool = True,
        rng: np.random.Generator = None,
        first_pass: bool = False
    ) -> np.ndarray:
        if synchronous:
            net_input = self.weights @ state
            if first_pass:
                max_val = float(np.max(np.abs(net_input)))
                if max_val > 0:
                    threshold = 0.35 * max_val
                    updated = self.activation(net_input)
                    return np.where(np.abs(net_input) >= threshold, updated, state)
            return self.activation(net_input)
        else:
            if rng is None:
                rng = np.random.default_rng()
            new_state = state.copy()
            initial_field = np.abs(self.weights @ new_state) + rng.uniform(0.0, 1.5, size=self.n)
            order = np.argsort(-initial_field)
            for idx in order:
                net_i = float(np.dot(self.weights[idx], new_state))
                if net_i > 0:
                    new_state[idx] = 1
                elif net_i < 0:
                    new_state[idx] = -1
            return new_state

    def recall(
        self,
        initial_state: np.ndarray,
        synchronous: bool = True,
        max_iterations: int = 30,
        rng_seed: Optional[np.random.Generator] = None
    ) -> RecallResult:
        rng = rng_seed if isinstance(rng_seed, np.random.Generator) else np.random.default_rng(rng_seed)
        current = initial_state.copy()
        history = [current.copy()]
        converged = False

        for step_idx in range(1, max_iterations + 1):
            next_state = self.step(
                current,
                synchronous=synchronous,
                rng=rng,
                first_pass=(step_idx == 1)
            )
            history.append(next_state.copy())

            if np.array_equal(next_state, current):
                if not synchronous or step_idx > 1 or np.array_equal(self.activation(self.weights @ next_state), next_state):
                    converged = True
                    current = next_state
                    break
            current = next_state

        matched_label = None
        inverted_label = None
        similarities: Dict[str, float] = {}

        for label, ref_vec in self.patterns.items():
            sim = pattern_similarity(current, ref_vec)
            similarities[label] = sim
            if np.array_equal(current, ref_vec):
                matched_label = label
            elif np.array_equal(current, -ref_vec):
                inverted_label = label

        return RecallResult(
            initial_state=initial_state.copy(),
            final_state=current,
            history=history,
            iterations=len(history) - 1,
            matched_label=matched_label,
            inverted_label=inverted_label,
            similarities=similarities,
            converged=converged
        )

    def run_noise_study(
        self,
        trials_per_level: int = 20,
        synchronous: bool = True,
        seed: int = 42
    ) -> Dict[int, Dict[str, float]]:
        rng = np.random.default_rng(seed)
        study_results: Dict[int, Dict[str, float]] = {}

        for noise in NOISE_LEVELS:
            study_results[noise] = {}
            total_success = 0
            total_runs = 0

            for label, ref_vec in self.patterns.items():
                successes = 0
                for _ in range(trials_per_level):
                    noisy_vec = apply_noise(ref_vec, noise, rng=rng)
                    res = self.recall(noisy_vec, synchronous=synchronous, rng_seed=rng)
                    if res.matched_label == label:
                        successes += 1
                acc = (successes / trials_per_level) * 100.0
                study_results[noise][label] = acc
                total_success += successes
                total_runs += trials_per_level

            study_results[noise]["AVG"] = (total_success / total_runs) * 100.0

        return study_results

    def generate_noise_samples(
        self,
        label: str,
        synchronous: bool = True,
        seed: int = 100
    ) -> List[Tuple[int, np.ndarray, RecallResult]]:
        rng = np.random.default_rng(seed)
        ref_vec = self.patterns[label]
        samples = []
        for noise in NOISE_LEVELS:
            noisy = apply_noise(ref_vec, noise, rng=rng)
            res = self.recall(noisy, synchronous=synchronous, rng_seed=rng)
            samples.append((noise, noisy, res))
        return samples