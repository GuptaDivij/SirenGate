from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


@dataclass
class RoutingDecision:
    route: str
    threshold_used: float
    raw_threshold_used: float
    transmitted_bytes: int
    used_cloud: bool


@dataclass
class AdaptiveRouter:
    """
    Adaptive Edge-AI router.

    Routes:
    - alert_only: high-confidence local prediction, send tiny JSON alert.
    - embedding: medium-confidence uncertainty, send compact embedding for cloud verification.
    - raw_audio: low-confidence or high-risk uncertainty, send full raw audio to cloud.

    The adaptive threshold changes online using a sliding window:
    - If bandwidth usage is too high, it lowers the escalation threshold.
    - If recent edge accuracy is too weak, it raises the escalation threshold.
    """

    alert_payload_bytes: int
    embedding_bytes: int
    raw_audio_bytes_per_second: int
    clip_seconds: float

    initial_threshold: float
    min_threshold: float
    max_threshold: float
    budget_upload_rate: float
    adaptation_rate: float
    sliding_window: int
    target_edge_accuracy: float

    embedding_band_width: float = 0.15
    min_top2_margin_for_alert: float = 0.10
    margin_penalty_strength: float = 0.08

    priority_weights: Dict[str, float] = field(default_factory=dict)
    false_positive_costs: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.threshold = float(self.initial_threshold)
        self.recent_bandwidth_fraction: Deque[float] = deque(maxlen=self.sliding_window)
        self.recent_correctness: Deque[int] = deque(maxlen=self.sliding_window)

    @property
    def raw_clip_bytes(self) -> int:
        return int(self.raw_audio_bytes_per_second * self.clip_seconds)

    def _effective_alert_threshold(self, predicted_label: str, top2_margin: float) -> float:
        priority = float(self.priority_weights.get(predicted_label, 1.0))
        fp_cost = float(self.false_positive_costs.get(predicted_label, priority))

        # Higher-priority and high-false-positive-cost classes require stronger confidence
        # before staying local. This directly handles classes like gun_shot and siren.
        priority_penalty = 0.05 * (priority - 1.0)
        fp_penalty = 0.03 * (fp_cost - 1.0)

        # If the top two classes are close, the model is uncertain even if top softmax is high.
        margin_gap = max(0.0, self.min_top2_margin_for_alert - float(top2_margin))
        margin_penalty = self.margin_penalty_strength * margin_gap / max(self.min_top2_margin_for_alert, 1e-6)

        threshold = self.threshold + priority_penalty + fp_penalty + margin_penalty
        return max(self.min_threshold, min(self.max_threshold, threshold))

    def decide(self, confidence: float, top2_margin: float, predicted_label: str) -> RoutingDecision:
        alert_threshold = self._effective_alert_threshold(predicted_label, top2_margin)
        raw_threshold = max(self.min_threshold, alert_threshold - self.embedding_band_width)

        if confidence >= alert_threshold and top2_margin >= self.min_top2_margin_for_alert:
            route = "alert_only"
            transmitted_bytes = self.alert_payload_bytes
            used_cloud = False
        elif confidence >= raw_threshold:
            route = "embedding"
            transmitted_bytes = self.embedding_bytes
            used_cloud = True
        else:
            route = "raw_audio"
            transmitted_bytes = self.raw_clip_bytes
            used_cloud = True

        return RoutingDecision(
            route=route,
            threshold_used=float(alert_threshold),
            raw_threshold_used=float(raw_threshold),
            transmitted_bytes=int(transmitted_bytes),
            used_cloud=bool(used_cloud),
        )

    def update(self, transmitted_bytes: int, edge_correct: bool) -> None:
        # Normalize bytes against full raw upload cost.
        bandwidth_fraction = float(transmitted_bytes) / max(float(self.raw_clip_bytes), 1.0)

        self.recent_bandwidth_fraction.append(bandwidth_fraction)
        self.recent_correctness.append(1 if edge_correct else 0)

        current_bandwidth_rate = sum(self.recent_bandwidth_fraction) / max(len(self.recent_bandwidth_fraction), 1)
        current_edge_accuracy = sum(self.recent_correctness) / max(len(self.recent_correctness), 1)

        budget_error = current_bandwidth_rate - self.budget_upload_rate
        accuracy_error = self.target_edge_accuracy - current_edge_accuracy

        # If bandwidth is too high, lower threshold so fewer samples escalate.
        self.threshold -= self.adaptation_rate * budget_error

        # If local model has been unreliable, raise threshold so more samples escalate.
        self.threshold += 0.5 * self.adaptation_rate * accuracy_error

        self.threshold = max(self.min_threshold, min(self.max_threshold, self.threshold))