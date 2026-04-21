from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


@dataclass
class RoutingDecision:
    route: str
    threshold_used: float
    margin_threshold_used: float
    transmitted_bytes: int
    used_cloud: bool
    sent_embedding: bool


@dataclass
class AdaptiveRouter:
    alert_payload_bytes: int
    embedding_bytes: int
    raw_audio_bytes_per_second: int
    clip_seconds: float
    initial_threshold: float
    initial_margin_threshold: float
    min_threshold: float
    max_threshold: float
    min_margin_threshold: float
    max_margin_threshold: float
    budget_upload_rate: float
    budget_embedding_rate: float
    adaptation_rate: float
    sliding_window: int
    target_edge_accuracy: float
    priority_weights: Dict[str, float] = field(default_factory=dict)
    false_positive_costs: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.threshold = self.initial_threshold
        self.margin_threshold = self.initial_margin_threshold
        self.recent_cloud_uploads: Deque[int] = deque(maxlen=self.sliding_window)
        self.recent_embedding_uploads: Deque[int] = deque(maxlen=self.sliding_window)
        self.recent_correctness: Deque[int] = deque(maxlen=self.sliding_window)

    @property
    def raw_clip_bytes(self) -> int:
        return int(self.raw_audio_bytes_per_second * self.clip_seconds)

    def decide(self, confidence: float, margin: float, predicted_label: str) -> RoutingDecision:
        priority = float(self.priority_weights.get(predicted_label, 1.0))
        fp_cost = float(self.false_positive_costs.get(predicted_label, priority))

        # Higher-priority or high-false-positive-cost classes escalate more aggressively.
        effective_threshold = min(self.max_threshold, self.threshold + 0.07 * (priority - 1.0) + 0.04 * (fp_cost - 1.0))
        effective_margin = min(
            self.max_margin_threshold,
            max(self.min_margin_threshold, self.margin_threshold + 0.03 * (priority - 1.0)),
        )

        if confidence < effective_threshold:
            route = "raw_audio"
            transmitted_bytes = self.raw_clip_bytes
            use_cloud = True
            sent_embedding = False
        elif margin < effective_margin:
            route = "embedding"
            transmitted_bytes = self.alert_payload_bytes + self.embedding_bytes
            use_cloud = False
            sent_embedding = True
        else:
            route = "alert_only"
            transmitted_bytes = self.alert_payload_bytes
            use_cloud = False
            sent_embedding = False

        return RoutingDecision(
            route=route,
            threshold_used=float(effective_threshold),
            margin_threshold_used=float(effective_margin),
            transmitted_bytes=transmitted_bytes,
            used_cloud=use_cloud,
            sent_embedding=sent_embedding,
        )

    def update(self, route: str, edge_correct: bool) -> None:
        self.recent_cloud_uploads.append(1 if route == "raw_audio" else 0)
        self.recent_embedding_uploads.append(1 if route == "embedding" else 0)
        self.recent_correctness.append(1 if edge_correct else 0)

        cloud_rate = sum(self.recent_cloud_uploads) / max(len(self.recent_cloud_uploads), 1)
        embedding_rate = sum(self.recent_embedding_uploads) / max(len(self.recent_embedding_uploads), 1)
        edge_acc = sum(self.recent_correctness) / max(len(self.recent_correctness), 1)

        cloud_budget_error = cloud_rate - self.budget_upload_rate
        embedding_budget_error = embedding_rate - self.budget_embedding_rate
        acc_error = self.target_edge_accuracy - edge_acc

        # If cloud use is too high, demand lower confidence before escalating to raw audio.
        self.threshold += self.adaptation_rate * cloud_budget_error

        # If edge accuracy is weak, escalate more examples.
        self.threshold -= 0.5 * self.adaptation_rate * acc_error
        self.margin_threshold += 0.5 * self.adaptation_rate * embedding_budget_error
        self.margin_threshold -= 0.25 * self.adaptation_rate * acc_error

        self.threshold = max(self.min_threshold, min(self.max_threshold, self.threshold))
        self.margin_threshold = max(self.min_margin_threshold, min(self.max_margin_threshold, self.margin_threshold))
