"""Plain dense autoencoder. Reconstruction error is the anomaly score.

This is not a VAE: there is no latent distribution and no KL term. The loss
is mean squared error between the input and the reconstruction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector:
    def __init__(
        self,
        seed: int = 42,
        epochs: int = 15,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        val_fraction: float = 0.1,
        threshold_quantile: float = 0.99,
    ) -> None:
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_fraction = val_fraction
        self.threshold_quantile = threshold_quantile
        self.threshold = 0.0
        self.final_loss = None
        self.net: DenseAutoencoder | None = None

    def fit(self, x_normal: np.ndarray) -> "AutoencoderDetector":
        torch.manual_seed(self.seed)
        x_normal = np.asarray(x_normal, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(x_normal))
        n_val = max(1, int(self.val_fraction * len(x_normal)))
        if n_val >= len(x_normal):
            n_val = max(1, len(x_normal) // 5)
        val = x_normal[order[:n_val]]
        train = x_normal[order[n_val:]]
        self.net = DenseAutoencoder(x_normal.shape[1])
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        train_tensor = torch.from_numpy(train)
        self.net.train()
        last_loss = 0.0
        for _epoch in range(self.epochs):
            permutation = torch.randperm(train_tensor.shape[0])
            total = 0.0
            seen = 0
            for start in range(0, train_tensor.shape[0], self.batch_size):
                batch_idx = permutation[start : start + self.batch_size]
                batch = train_tensor[batch_idx]
                optimizer.zero_grad()
                rebuilt = self.net(batch)
                loss = loss_fn(rebuilt, batch)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(batch)
                seen += len(batch)
            last_loss = total / max(seen, 1)
        self.final_loss = float(last_loss)
        val_error = self.anomaly_scores(val)
        self.threshold = float(np.quantile(val_error, self.threshold_quantile))
        return self

    def anomaly_scores(self, x: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("autoencoder is not fit")
        self.net.eval()
        values = np.asarray(x, dtype=np.float32)
        scores = np.empty(len(values), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch = torch.from_numpy(values[start : start + self.batch_size])
                rebuilt = self.net(batch)
                error = torch.mean((rebuilt - batch) ** 2, dim=1)
                scores[start : start + len(batch)] = error.numpy()
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.anomaly_scores(x) >= self.threshold).astype(int)
