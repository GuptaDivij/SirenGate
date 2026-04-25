from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    seed: int = 537

    # Audio preprocessing
    sample_rate: int = 16000
    clip_seconds: float = 4.0
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256

    # Edge model
    embedding_dim: int = 128
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fold: int = 10

    # Transmission sizes
    alert_payload_bytes: int = 64
    embedding_bytes: int = 256
    raw_audio_bytes_per_second: int = 32000

    # Simulated cloud verifier quality
    # Embedding verification is better than edge-only but weaker than full raw-audio cloud verification.
    embedding_cloud_accuracy_boost: float = 0.08
    raw_cloud_accuracy_boost: float = 0.18

    # Backward-compatible old field. Used as raw_cloud_accuracy_boost if present in older configs.
    cloud_accuracy_boost: float = 0.18

    # Adaptive middleware
    initial_threshold: float = 0.55
    min_threshold: float = 0.30
    max_threshold: float = 0.95
    budget_upload_rate: float = 0.40
    adaptation_rate: float = 0.06
    sliding_window: int = 50
    target_edge_accuracy: float = 0.80

    # Three-route triage behavior
    embedding_band_width: float = 0.15
    min_top2_margin_for_alert: float = 0.10
    margin_penalty_strength: float = 0.08

    threshold_sweep: list[float] | None = None

    # Class importance
    priority_weights: Dict[str, float] | None = None
    false_positive_costs: Dict[str, float] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        config = cls(**data)

        # Keep old config files working.
        if "raw_cloud_accuracy_boost" not in data and "cloud_accuracy_boost" in data:
            config.raw_cloud_accuracy_boost = float(data["cloud_accuracy_boost"])

        return config