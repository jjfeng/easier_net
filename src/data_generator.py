import numpy as np
import scipy.linalg

import torch
from torch.utils.data import TensorDataset


class DataGenerator:
    def __init__(
        self,
        num_p,
        func,
        x_scale: float = 1,
        y_scale: float = 1,
        y_shift: float = 0,
        is_classification=False,
        snr: float = 1,
        sigma_eps: float = None,
        correlation: float = 0,
        num_true: int = 1,
        num_corr: int = 0,
    ):
        self.num_p = num_p
        self.func = func
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.y_shift = y_shift
        self.is_classification = is_classification
        self.snr = snr
        self.sigma_eps = sigma_eps
        self.correlation = correlation
        self.num_true = num_true
        self.num_corr = num_corr

    def create_data(self, n_obs, max_gen_obs: int = 10000):
        assert n_obs > 0

        max_gen_obs = max(max_gen_obs, n_obs)

        xs = np.random.rand(max_gen_obs, self.num_p) * self.x_scale
        for idx in range(self.num_true):
            for j in range(self.num_corr):
                similar_idx = (j + 1) * self.num_true + idx
                xs[:, similar_idx] = (1 - self.correlation) * xs[
                    :, similar_idx
                ] + self.correlation * xs[:, idx]

        if not self.is_classification:
            # regression
            true_ys = self.func(xs)
            true_ys = np.reshape(true_ys, (true_ys.size, 1))
            eps = np.random.randn(xs.shape[0], 1)
            eps_norm = np.sqrt(np.var(eps))
            y_norm = np.sqrt(np.var(true_ys))
            sigma_eps = (
                (1.0 / self.snr * y_norm / eps_norm)
                if self.snr is not None
                else self.sigma_eps
            )
            print("SIGMA", sigma_eps)
            y = true_ys + sigma_eps * eps
        else:
            # classification
            true_ys = self.func(xs)
            true_ys = np.reshape(true_ys, (true_ys.size, 1))
            y = np.array(
                np.random.random_sample((true_ys.size, 1)) < true_ys, dtype=int
            )

        y = (y - self.y_shift) / self.y_scale
        true_ys = (true_ys - self.y_shift) / self.y_scale

        return xs[:n_obs], y[:n_obs], true_ys[:n_obs], sigma_eps
