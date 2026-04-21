from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


@dataclass
class RoutingDecision:
    route: str
    threshold_used: float
    transmitted_bytes: int
    used_cloud: bool


@dataclass
class AdaptiveRouter:
    alert_payload_bytes: int
    raw_audio_bytes_per_second: int
    clip_seconds: float
    initial_threshold: float
    min_threshold: float
    max_threshold: float
    budget_upload_rate: float
    adaptation_rate: float
    sliding_window: int
    priority_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.threshold = self.initial_threshold
        self.recent_uploads: Deque[int] = deque(maxlen=self.sliding_window)
        self.recent_correctness: Deque[int] = deque(maxlen=self.sliding_window)

    @property
    def raw_clip_bytes(self) -> int:
        return int(self.raw_audio_bytes_per_second * self.clip_seconds)

    def decide(self, confidence: float, predicted_label: str) -> RoutingDecision:
        priority = float(self.priority_weights.get(predicted_label, 1.0))

        # Higher-priority classes get slightly stricter routing,
        # meaning we more easily escalate them to the cloud.
        effective_threshold = min(self.max_threshold, self.threshold + 0.06 * (priority - 1.0))
        use_cloud = confidence < effective_threshold

        if use_cloud:
            route = "raw_audio"
            transmitted_bytes = self.raw_clip_bytes
        else:
            route = "alert_only"
            transmitted_bytes = self.alert_payload_bytes

        return RoutingDecision(
            route=route,
            threshold_used=float(effective_threshold),
            transmitted_bytes=transmitted_bytes,
            used_cloud=use_cloud,
        )

    def update(self, uploaded: bool, edge_correct: bool) -> None:
        self.recent_uploads.append(1 if uploaded else 0)
        self.recent_correctness.append(1 if edge_correct else 0)

        upload_rate = sum(self.recent_uploads) / max(len(self.recent_uploads), 1)
        edge_acc = sum(self.recent_correctness) / max(len(self.recent_correctness), 1)

        budget_error = upload_rate - self.budget_upload_rate
        acc_error = 0.80 - edge_acc

        # If we’re uploading too much, threshold rises.
        self.threshold += self.adaptation_rate * budget_error

        # If edge accuracy is weak, threshold falls so more clips go to cloud.
        self.threshold -= 0.5 * self.adaptation_rate * acc_error

        self.threshold = max(self.min_threshold, min(self.max_threshold, self.threshold))