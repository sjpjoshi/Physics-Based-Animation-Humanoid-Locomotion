from __future__ import annotations

import gymnasium as gym
import numpy as np


class Discretizer:
    """Bins a continuous Box observation into a tuple of bin indices."""

    def __init__(self, low, high, n_bins) -> None:
        # low / high: per-dimension FINITE ranges to bin over. CartPole's velocity
        # dims (idx 1 and 3) are unbounded, so the runner passes hand-picked clip
        # ranges for them. n_bins: an int (same for every dim) or a per-dim sequence.
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        n_dims = len(self.low)
        self.n_bins = [n_bins] * n_dims if isinstance(n_bins, int) else list(n_bins)

    def __call__(self, obs) -> tuple[int, ...]:
        obs = np.asarray(obs, dtype=np.float64)
        clipped = np.clip(obs, self.low, self.high)

        bins = []
        for i in range(len(clipped)):
            edges = np.linspace(
                self.low[i],
                self.high[i],
                self.n_bins[i] + 1,
            )[1:-1]

            bin_idx = int(np.digitize(clipped[i], edges))
            bins.append(bin_idx)

        return tuple(bins)

class DiscretizedObservation(gym.ObservationWrapper):
    """Applies a ``Discretizer`` to every observation.

    With this wrapper, a tabular agent sees hashable tuple states straight out of
    ``reset()`` / ``step()``, so the agent code needs no changes.
    """

    def __init__(self, env: gym.Env, discretizer: Discretizer) -> None:
        super().__init__(env)
        self._discretizer = discretizer

    def observation(self, observation):
        return self._discretizer(observation)
