"""GRU-based temporal monitor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class _GRUNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)  # scalar per sequence


class _WindowDataset(Dataset):
    def __init__(self, X_seq: list[np.ndarray], y_seq: list[np.ndarray], window: int):
        self.windows: list[tuple[np.ndarray, float]] = []
        for X, y in zip(X_seq, y_seq):
            for t in range(len(y)):
                start = max(0, t - window + 1)
                chunk = X[start : t + 1]
                pad = np.zeros((window - len(chunk), X.shape[1]), dtype=np.float32)
                padded = np.vstack([pad, chunk])
                self.windows.append((padded.astype(np.float32), float(y[t])))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, y = self.windows[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class GRUMonitor:
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        window_size: int = 10,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 50,
        patience: int = 10,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.window_size = window_size
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.net: _GRUNet | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X_seq: list[np.ndarray], y_seq: list[np.ndarray]) -> "GRUMonitor":
        input_size = X_seq[0].shape[1]
        self.net = _GRUNet(input_size, self.hidden_size, self.num_layers, self.dropout)
        self.net.to(self.device)
        dataset = _WindowDataset(X_seq, y_seq, self.window_size)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        best_loss, no_improve = float("inf"), 0
        for epoch in range(self.max_epochs):
            self.net.train()
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                out = self.net(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X = X.astype(np.float32)
        results = []
        T = len(X)
        for t in range(T):
            start = max(0, t - self.window_size + 1)
            chunk = X[start : t + 1]
            pad = np.zeros((self.window_size - len(chunk), X.shape[1]), dtype=np.float32)
            padded = np.vstack([pad, chunk])
            x = torch.from_numpy(padded[None]).to(self.device)
            logit = self.net(x).item()
            results.append(torch.sigmoid(torch.tensor(logit)).item())
        return np.array(results, dtype=np.float32)

    def feature_importances(self, feature_names: list[str]) -> dict[str, float]:
        return {}  # GRU doesn't have direct feature importances
