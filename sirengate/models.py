from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader


class SmallAudioCNN(nn.Module):
    def __init__(self, num_classes: int = 10, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.features(x)
        emb = self.embedding(feats)
        logits = self.classifier(emb)
        return {"logits": logits, "embedding": emb}


@dataclass
class TrainArtifacts:
    best_model_path: str
    history: List[dict]
    val_accuracy: float


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    embeddings_all: List[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].numpy()
        out = model(x)

        logits_all.append(out["logits"].cpu().numpy())
        embeddings_all.append(out["embedding"].cpu().numpy())
        labels_all.append(y)

    return (
        np.concatenate(logits_all),
        np.concatenate(labels_all),
        np.concatenate(embeddings_all),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    output_dir: str | Path,
    device: torch.device,
) -> TrainArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_acc = -1.0
    best_model_path = output_dir / "best_model.pt"
    history: List[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        train_true: List[int] = []
        train_pred: List[int] = []

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out["logits"], y)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_true.extend(y.cpu().tolist())
            train_pred.extend(out["logits"].argmax(dim=1).cpu().tolist())

        model.eval()
        val_true: List[int] = []
        val_pred: List[int] = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                out = model(x)
                val_true.extend(y.cpu().tolist())
                val_pred.extend(out["logits"].argmax(dim=1).cpu().tolist())

        train_acc = accuracy_score(train_true, train_pred)
        val_acc = accuracy_score(val_true, val_pred)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
        }
        history.append(row)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    return TrainArtifacts(
        best_model_path=str(best_model_path),
        history=history,
        val_accuracy=float(best_acc),
    )