from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .calibration import TemperatureScaler
from .metrics import classification_metrics, weighted_event_score, weighted_precision_recall_f1
from .middleware import AdaptiveRouter
from .models import collect_logits
from .utils import IDX_TO_CLASS, URBAN_SOUND_CLASSES


@dataclass
class SimulationResult:
    summary: Dict[str, float]
    events: pd.DataFrame


def _simulate_cloud_prediction(
    true_label: int,
    edge_pred: int,
    rng: np.random.Generator,
    boost: float,
    num_classes: int,
) -> int:
    if edge_pred == true_label:
        return true_label

    p_correct = min(0.98, 0.55 + boost)
    return true_label if rng.random() < p_correct else int(rng.integers(0, num_classes))


def _simulate_embedding_prediction(
    true_label: int,
    edge_pred: int,
    rng: np.random.Generator,
    boost: float,
    num_classes: int,
) -> int:
    if edge_pred == true_label:
        return true_label

    p_correct = min(0.92, 0.35 + boost)
    return true_label if rng.random() < p_correct else edge_pred if rng.random() < 0.6 else int(rng.integers(0, num_classes))


def evaluate_policies_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    clip_seconds: float,
    scaler: TemperatureScaler,
    config,
    output_dir: str | Path,
) -> Dict[str, SimulationResult]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    probs = scaler.transform_probs(logits)
    confidences = probs.max(axis=1)
    edge_preds = probs.argmax(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    num_classes = probs.shape[1]

    priority_weights = config.priority_weights or {}
    class_weights = {idx: float(priority_weights.get(name, 1.0)) for idx, name in IDX_TO_CLASS.items()}
    false_positive_costs = {
        idx: float((config.false_positive_costs or priority_weights).get(name, 1.0))
        for idx, name in IDX_TO_CLASS.items()
    }
    results: Dict[str, SimulationResult] = {}

    baselines = ["always_cloud", "edge_only", "fixed_threshold", "fixed_triage", "adaptive_triage"]

    for policy_name in baselines:
        router = AdaptiveRouter(
            alert_payload_bytes=config.alert_payload_bytes,
            embedding_bytes=config.embedding_bytes,
            raw_audio_bytes_per_second=config.raw_audio_bytes_per_second,
            clip_seconds=clip_seconds,
            initial_threshold=config.initial_threshold,
            initial_margin_threshold=config.initial_margin_threshold,
            min_threshold=config.min_threshold,
            max_threshold=config.max_threshold,
            min_margin_threshold=config.min_margin_threshold,
            max_margin_threshold=config.max_margin_threshold,
            budget_upload_rate=config.budget_upload_rate,
            budget_embedding_rate=config.budget_embedding_rate,
            adaptation_rate=config.adaptation_rate,
            sliding_window=config.sliding_window,
            target_edge_accuracy=config.target_edge_accuracy,
            priority_weights=priority_weights,
            false_positive_costs=config.false_positive_costs or {},
        )

        final_preds: List[int] = []
        transmitted_bytes: List[int] = []
        routes: List[str] = []
        thresholds: List[float] = []
        margin_thresholds: List[float] = []
        cloud_used: List[bool] = []
        embedding_used: List[bool] = []

        for i in range(len(labels)):
            label_name = IDX_TO_CLASS[int(edge_preds[i])]
            edge_correct = int(edge_preds[i]) == int(labels[i])

            if policy_name == "always_cloud":
                pred = _simulate_cloud_prediction(
                    int(labels[i]),
                    int(edge_preds[i]),
                    rng,
                    config.cloud_accuracy_boost,
                    num_classes,
                )
                bytes_sent = int(config.raw_audio_bytes_per_second * clip_seconds)
                route = "raw_audio"
                threshold_used = config.initial_threshold
                margin_threshold_used = config.initial_margin_threshold
                used_cloud = True
                sent_embedding = False

            elif policy_name == "edge_only":
                pred = int(edge_preds[i])
                bytes_sent = config.alert_payload_bytes
                route = "alert_only"
                threshold_used = config.initial_threshold
                margin_threshold_used = config.initial_margin_threshold
                used_cloud = False
                sent_embedding = False

            elif policy_name == "fixed_threshold":
                use_cloud = float(confidences[i]) < config.initial_threshold
                if use_cloud:
                    pred = _simulate_cloud_prediction(
                        int(labels[i]),
                        int(edge_preds[i]),
                        rng,
                        config.cloud_accuracy_boost,
                        num_classes,
                    )
                    bytes_sent = int(config.raw_audio_bytes_per_second * clip_seconds)
                    route = "raw_audio"
                else:
                    pred = int(edge_preds[i])
                    bytes_sent = config.alert_payload_bytes
                    route = "alert_only"

                threshold_used = config.initial_threshold
                margin_threshold_used = config.initial_margin_threshold
                used_cloud = use_cloud
                sent_embedding = False

            elif policy_name == "fixed_triage":
                if float(confidences[i]) < config.initial_threshold:
                    pred = _simulate_cloud_prediction(
                        int(labels[i]),
                        int(edge_preds[i]),
                        rng,
                        config.cloud_accuracy_boost,
                        num_classes,
                    )
                    bytes_sent = int(config.raw_audio_bytes_per_second * clip_seconds)
                    route = "raw_audio"
                    used_cloud = True
                    sent_embedding = False
                elif float(margins[i]) < config.initial_margin_threshold:
                    pred = _simulate_embedding_prediction(
                        int(labels[i]),
                        int(edge_preds[i]),
                        rng,
                        config.embedding_accuracy_boost,
                        num_classes,
                    )
                    bytes_sent = int(config.alert_payload_bytes + config.embedding_bytes)
                    route = "embedding"
                    used_cloud = False
                    sent_embedding = True
                else:
                    pred = int(edge_preds[i])
                    bytes_sent = config.alert_payload_bytes
                    route = "alert_only"
                    used_cloud = False
                    sent_embedding = False

                threshold_used = config.initial_threshold
                margin_threshold_used = config.initial_margin_threshold

            else:
                decision = router.decide(float(confidences[i]), float(margins[i]), label_name)

                if decision.used_cloud:
                    pred = _simulate_cloud_prediction(
                        int(labels[i]),
                        int(edge_preds[i]),
                        rng,
                        config.cloud_accuracy_boost,
                        num_classes,
                    )
                elif decision.sent_embedding:
                    pred = _simulate_embedding_prediction(
                        int(labels[i]),
                        int(edge_preds[i]),
                        rng,
                        config.embedding_accuracy_boost,
                        num_classes,
                    )
                else:
                    pred = int(edge_preds[i])

                bytes_sent = decision.transmitted_bytes
                route = decision.route
                threshold_used = decision.threshold_used
                margin_threshold_used = decision.margin_threshold_used
                used_cloud = decision.used_cloud
                sent_embedding = decision.sent_embedding

                router.update(route=decision.route, edge_correct=edge_correct)

            final_preds.append(pred)
            transmitted_bytes.append(bytes_sent)
            routes.append(route)
            thresholds.append(threshold_used)
            margin_thresholds.append(margin_threshold_used)
            cloud_used.append(used_cloud)
            embedding_used.append(sent_embedding)

        metrics = classification_metrics(labels, final_preds, labels=list(range(len(URBAN_SOUND_CLASSES))))
        metrics["weighted_event_score"] = weighted_event_score(labels, final_preds, class_weights)
        metrics.update(
            weighted_precision_recall_f1(
                labels,
                final_preds,
                class_weights,
                labels=list(range(len(URBAN_SOUND_CLASSES))),
            )
        )
        metrics["avg_bytes_per_clip"] = float(np.mean(transmitted_bytes))

        raw_bytes = float(config.raw_audio_bytes_per_second * clip_seconds)
        metrics["bandwidth_reduction_vs_always_cloud"] = float(1.0 - (np.mean(transmitted_bytes) / raw_bytes))
        metrics["cloud_upload_rate"] = float(np.mean(cloud_used))
        metrics["embedding_route_rate"] = float(np.mean(embedding_used))
        metrics["priority_false_positive_cost"] = float(
            np.mean(
                [
                    false_positive_costs[int(pred)]
                    for truth, pred in zip(labels, final_preds)
                    if int(truth) != int(pred)
                ]
            )
            if any(int(truth) != int(pred) for truth, pred in zip(labels, final_preds))
            else 0.0
        )

        df = pd.DataFrame(
            {
                "true_label": [IDX_TO_CLASS[int(y)] for y in labels],
                "edge_pred": [IDX_TO_CLASS[int(y)] for y in edge_preds],
                "final_pred": [IDX_TO_CLASS[int(y)] for y in final_preds],
                "confidence": confidences,
                "margin": margins,
                "route": routes,
                "threshold_used": thresholds,
                "margin_threshold_used": margin_thresholds,
                "transmitted_bytes": transmitted_bytes,
                "used_cloud": cloud_used,
                "sent_embedding": embedding_used,
                "embedding_l2": np.linalg.norm(embeddings, axis=1),
            }
        )

        df.to_csv(output_dir / f"events_{policy_name}.csv", index=False)
        results[policy_name] = SimulationResult(summary=metrics, events=df)

    summary_df = pd.DataFrame([{"policy": k, **v.summary} for k, v in results.items()])
    summary_df.to_csv(output_dir / "policy_summary.csv", index=False)

    return results


@torch.no_grad()
def collect_from_model(model, loader: DataLoader, device: torch.device):
    return collect_logits(model, loader, device)
