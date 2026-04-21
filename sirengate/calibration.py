from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 300, lr: float = 0.01) -> "TemperatureScaler":
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)

        temp = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([temp], lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(max_iter):
            optimizer.zero_grad()
            loss = criterion(logits_t / temp.clamp(min=1e-3), labels_t)
            loss.backward()
            optimizer.step()

        self.temperature = float(temp.detach().clamp(min=1e-3).item())
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits / max(self.temperature, 1e-6)

    def transform_probs(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.transform_logits(logits)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        return exp / exp.sum(axis=1, keepdims=True)