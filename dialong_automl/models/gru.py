"""GRU model wrapper for DiaLong-AutoML (Phase 5).

Uses the padded sequence representation (Phase 4).
Input shape: (N, T, F).

If torch cannot be imported (e.g. DLL blocked by OS Application Control
policy), this module still imports cleanly. The ImportError surfaces at
instantiation time (``__init__``) with a clear message.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from dialong_automl.models.base import BaseModel, REPRESENTATION_SEQUENCE

logger = logging.getLogger(__name__)

# Attempt torch import once at module level — failure is deferred to __init__
try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence as _pack
    _TORCH_OK = True
    _TORCH_ERR: Exception | None = None
except Exception as _e:
    _TORCH_OK = False
    _TORCH_ERR = _e


class _GRUNet:
    """Placeholder — only instantiated when torch is available."""
    pass


if _TORCH_OK:
    class _GRUNet(nn.Module):  # type: ignore[no-redef]
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.classifier = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor", lengths: "torch.Tensor") -> "torch.Tensor":
            packed = _pack(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(packed)
            return self.classifier(h_n[-1]).squeeze(1)


class GRUModel(BaseModel):
    """GRU sequence classifier (PyTorch).

    Raises ImportError at instantiation if torch is unavailable.
    """

    name = "gru"
    representation = REPRESENTATION_SEQUENCE

    _DEFAULTS: dict[str, Any] = {
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "epochs": 30,
        "patience": 5,
        "seed": 42,
    }

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        if not _TORCH_OK:
            raise ImportError(
                "torch is required for GRUModel but could not be imported "
                f"(possibly blocked by OS Application Control policy): {_TORCH_ERR}"
            )
        super().__init__({**self._DEFAULTS, **(hyperparams or {})})
        self._net: Any = None
        self._input_size: int | None = None
        self._device = torch.device("cpu")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray | None = None,
        lengths: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        mask_val: np.ndarray | None = None,
        lengths_val: np.ndarray | None = None,
        trial: Any = None,
    ) -> "GRUModel":
        hp = self.hyperparams
        seed = int(hp.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        N, T, F = X.shape
        self._input_size = F

        if lengths is None:
            lengths = (mask.sum(axis=1) if mask is not None else np.full(N, T)).astype(np.int32)
        lengths = np.maximum(lengths, 1)

        hidden_size = int(hp["hidden_size"])
        num_layers = int(hp["num_layers"])
        dropout = float(hp["dropout"]) if num_layers > 1 else 0.0
        lr = float(hp["learning_rate"])
        wd = float(hp["weight_decay"])
        batch_size = int(hp["batch_size"])
        epochs = int(hp["epochs"])
        patience = int(hp["patience"])

        self._net = _GRUNet(F, hidden_size, num_layers, dropout).to(self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.BCEWithLogitsLoss()

        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self._device)
        len_t = torch.tensor(lengths, dtype=torch.long, device=self._device)

        has_val = X_val is not None and y_val is not None
        if has_val:
            lv = (mask_val.sum(axis=1) if mask_val is not None else
                  np.full(len(X_val), X_val.shape[1])).astype(np.int32)
            lv = np.maximum(lv, 1)
            Xv_t = torch.tensor(X_val, dtype=torch.float32, device=self._device)
            yv_t = torch.tensor(y_val, dtype=torch.float32, device=self._device)
            lv_t = torch.tensor(lv, dtype=torch.long, device=self._device)

        best_val = -1.0
        best_state = None
        no_improve = 0
        epoch = 0

        for epoch in range(epochs):
            self._net.train()
            idx = np.random.permutation(N)
            for start in range(0, N, batch_size):
                b = idx[start: start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self._net(X_t[b], len_t[b]), y_t[b])
                loss.backward()
                optimizer.step()

            if has_val:
                val_auroc = self._val_auroc(Xv_t, yv_t, lv_t)
                if val_auroc > best_val:
                    best_val = val_auroc
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if trial is not None:
                    import optuna
                    trial.report(val_auroc, epoch)
                    if trial.should_prune():
                        break
                if no_improve >= patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._fitted = True
        logger.info("GRUModel fitted: F=%d, best_val_auroc=%.4f", F, best_val)
        return self

    def _val_auroc(self, Xv_t: Any, yv_t: Any, lv_t: Any) -> float:
        from dialong_automl.evaluation.metrics import auroc as _auroc
        self._net.eval()
        with torch.no_grad():
            proba = torch.sigmoid(self._net(Xv_t, lv_t)).cpu().numpy()
        y_true = yv_t.cpu().numpy()
        score = _auroc(y_true, proba)
        return score if score is not None else 0.5

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        N, T, F = X.shape
        lengths = np.full(N, T, dtype=np.int32)
        return self._predict(X, lengths)

    def predict_proba_with_mask(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._require_fitted()
        lengths = np.maximum(mask.sum(axis=1).astype(np.int32), 1)
        return self._predict(X, lengths)

    def _predict(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        self._net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
            len_t = torch.tensor(lengths, dtype=torch.long, device=self._device)
            proba = torch.sigmoid(self._net(X_t, len_t)).cpu().numpy().astype(np.float32)
        return proba

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "net_state": self._net.state_dict(),
            "hyperparams": self.hyperparams,
            "input_size": self._input_size,
        }, path)
        logger.info("GRUModel saved -> %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "GRUModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hp = ckpt["hyperparams"]
        obj = cls(hyperparams=hp)
        F = ckpt["input_size"]
        hs = int(hp["hidden_size"])
        nl = int(hp["num_layers"])
        do = float(hp["dropout"]) if nl > 1 else 0.0
        obj._net = _GRUNet(F, hs, nl, do)
        obj._net.load_state_dict(ckpt["net_state"])
        obj._net.eval()
        obj._input_size = F
        obj._fitted = True
        return obj
