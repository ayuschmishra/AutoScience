"""PyTorch tabular MLP behind the scikit-learn estimator interface.

Design constraints:
- Fully seeded: same ``random_state`` -> bit-identical weights and predictions
  (verified by tests), which the reproducibility study depends on.
- Mini-batch ``DataLoader`` training: memory stays bounded by batch size, so
  the same code path scales from wine (178 rows) to full HIGGS (11M rows).
- Early stopping on an internal validation split with best-weights restore.
- MC-dropout predictive uncertainty for regression (Phase 5 hooks).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_VALIDATION_FRACTION = 0.1
_MIN_ROWS_FOR_EARLY_STOPPING = 100


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class _MLPModule(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        batch_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(width, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            width = hidden_dim
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class _TorchMLPBase(BaseEstimator):  # type: ignore[misc]  # sklearn is untyped
    """Shared training loop for the classifier and regressor."""

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        batch_norm: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
        device: str = "auto",
        random_state: int = 0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.random_state = random_state

    # -- subclass hooks ----------------------------------------------------
    def _out_dim(self) -> int:
        raise NotImplementedError

    def _loss_fn(self) -> nn.Module:
        raise NotImplementedError

    def _prepare_target(self, y: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    # -- training ----------------------------------------------------------
    def _fit_impl(self, x: np.ndarray, y_tensor: torch.Tensor) -> None:
        torch.manual_seed(self.random_state)
        dev = _resolve_device(self.device)
        x_tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

        n = len(x_tensor)
        generator = torch.Generator().manual_seed(self.random_state)
        use_early_stopping = n >= _MIN_ROWS_FOR_EARLY_STOPPING
        if use_early_stopping:
            n_val = max(int(n * _VALIDATION_FRACTION), 1)
            perm = torch.randperm(n, generator=generator)
            val_idx, train_idx = perm[:n_val], perm[n_val:]
        else:
            train_idx = torch.arange(n)
            val_idx = torch.arange(0)

        train_ds = TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
        loader = DataLoader(
            train_ds,
            batch_size=min(self.batch_size, len(train_ds)),
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=len(train_ds) > self.batch_size,
        )
        x_val = x_tensor[val_idx].to(dev)
        y_val = y_tensor[val_idx].to(dev)

        model = _MLPModule(
            in_dim=x.shape[1],
            out_dim=self._out_dim(),
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        ).to(dev)
        loss_fn = self._loss_fn()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        best_val = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        for _epoch in range(self.max_epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb.to(dev)), yb.to(dev))
                loss.backward()
                optimizer.step()
            scheduler.step()

            if not use_early_stopping:
                continue
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(x_val), y_val))
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if use_early_stopping:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.device_ = dev
        self.n_features_in_ = x.shape[1]

    def _forward_batched(self, x: np.ndarray) -> torch.Tensor:
        x_tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x_tensor), self.batch_size):
                batch = x_tensor[start : start + self.batch_size].to(self.device_)
                outputs.append(self.model_(batch).cpu())
        return torch.cat(outputs)


class TorchMLPClassifier(_TorchMLPBase, ClassifierMixin):  # type: ignore[misc]
    """Seeded MLP classifier with predict_proba."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> TorchMLPClassifier:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self._n_classes = len(self.classes_)
        self._fit_impl(x, torch.from_numpy(y_idx.astype(np.int64)))
        return self

    def _out_dim(self) -> int:
        return self._n_classes

    def _loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self._forward_batched(np.asarray(x, dtype=np.float32))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return np.asarray(self.classes_[proba.argmax(axis=1)])


class TorchMLPRegressor(_TorchMLPBase, RegressorMixin):  # type: ignore[misc]
    """Seeded MLP regressor with MC-dropout predictive uncertainty.

    Targets are standardized internally for stable optimization; predictions
    are returned on the original scale.
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> TorchMLPRegressor:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        y_standardized = (y - self._y_mean) / self._y_std
        self._fit_impl(x, torch.from_numpy(y_standardized).unsqueeze(1))
        return self

    def _out_dim(self) -> int:
        return 1

    def _loss_fn(self) -> nn.Module:
        return nn.MSELoss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = self._forward_batched(np.asarray(x, dtype=np.float32))
        return (out.squeeze(1).numpy() * self._y_std + self._y_mean).astype(np.float64)

    def predict_uncertainty(self, x: np.ndarray, n_samples: int = 30) -> dict[str, Any]:
        """MC-dropout: mean and standard deviation over stochastic passes.

        Dropout layers are put in train mode (BatchNorm stays in eval mode),
        and each pass is seeded, so the uncertainty itself is reproducible.
        """
        for module in self.model_.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        torch.manual_seed(self.random_state)
        draws = np.stack(
            [
                self._forward_batched(np.asarray(x, dtype=np.float32)).squeeze(1).numpy()
                for _ in range(n_samples)
            ]
        )
        self.model_.eval()
        draws = draws * self._y_std + self._y_mean
        return {"mean": draws.mean(axis=0), "std": draws.std(axis=0)}
