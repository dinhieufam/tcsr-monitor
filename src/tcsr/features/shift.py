"""Distribution-shift features: Mahalanobis and kNN distance to train embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class ShiftDetector:
    """Fit on train embeddings, then score test embeddings."""

    def __init__(self, method: str = "mahalanobis", n_neighbors: int = 10):
        if method not in ("mahalanobis", "knn"):
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.n_neighbors = n_neighbors
        self._mean: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None
        self._knn: NearestNeighbors | None = None

    def fit(self, embeddings: np.ndarray) -> None:
        """embeddings: (N, D) float."""
        self._mean = embeddings.mean(0)
        if self.method == "mahalanobis":
            cov = np.cov(embeddings.T) + np.eye(embeddings.shape[1]) * 1e-6
            self._cov_inv = np.linalg.inv(cov)
        else:
            self._knn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm="ball_tree")
            self._knn.fit(embeddings)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return per-sample distance scores (N,)."""
        if self.method == "mahalanobis":
            diff = embeddings - self._mean
            return np.array([
                float(np.sqrt(np.dot(d, self._cov_inv @ d))) for d in diff
            ])
        else:
            dists, _ = self._knn.kneighbors(embeddings)
            return dists.mean(1)


def extract_encoder_embedding(
    model,
    imgs: "torch.Tensor",
    layer: str = "layer4",
    device: str = "cuda",
) -> np.ndarray:
    """Pool the encoder's feature map at `layer` to a (B, D) vector."""
    import torch

    hooks = []
    activations: dict[str, "torch.Tensor"] = {}

    def hook_fn(name):
        def fn(module, inp, out):
            activations[name] = out.detach()
        return fn

    encoder = model.encoder if hasattr(model, "encoder") else model
    target_layer = getattr(encoder, layer, None)
    if target_layer is None:
        raise AttributeError(f"Encoder has no attribute {layer!r}")
    hooks.append(target_layer.register_forward_hook(hook_fn(layer)))

    with torch.no_grad():
        model(imgs.to(device))

    for h in hooks:
        h.remove()

    feat = activations[layer]  # (B, C, H, W)
    return feat.mean(dim=(2, 3)).cpu().numpy()
