from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    seed: int = 537
    sample_rate: int = 16000
    clip_seconds: float = 4.0
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fold: int = 10

    # Simulation / middleware
    cloud_accuracy_boost: float = 0.18
    alert_payload_bytes: int = 64
    embedding_bytes: int = 256
    raw_audio_bytes_per_second: int = 32000

    # Adaptive policy
    initial_threshold: float = 0.72
    min_threshold: float = 0.45
    max_threshold: float = 0.95
    budget_upload_rate: float = 0.25
    adaptation_rate: float = 0.08
    sliding_window: int = 50

    # Class importance
    priority_weights: Dict[str, float] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f)
        return cls(**data)