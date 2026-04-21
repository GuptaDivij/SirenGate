from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .calibration import TemperatureScaler
from .metrics import classification_metrics, weighted_event_score
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
    num_classes = probs.shape[1]

    class_weights = {idx: float(config.priority_weights.get(name, 1.0)) for idx, name in IDX_TO_CLASS.items()}
    results: Dict[str, SimulationResult] = {}

    baselines = ["always_cloud", "edge_only", "fixed_threshold", "adaptive_threshold"]

    for policy_name in baselines:
        router = AdaptiveRouter(
            alert_payload_bytes=config.alert_payload_bytes,
            raw_audio_bytes_per_second=config.raw_audio_bytes_per_second,
            clip_seconds=clip_seconds,
            initial_threshold=config.initial_threshold,
            min_threshold=config.min_threshold,
            max_threshold=config.max_threshold,
            budget_upload_rate=config.budget_upload_rate,
            adaptation_rate=config.adaptation_rate,
            sliding_window=config.sliding_window,
            priority_weights=config.priority_weights,
        )

        final_preds: List[int] = []
        transmitted_bytes: List[int] = []
        routes: List[str] = []
        thresholds: List[float] = []
        cloud_used: List[bool] = []

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
                used_cloud = True

            elif policy_name == "edge_only":
                pred = int(edge_preds[i])
                bytes_sent = config.alert_payload_bytes
                route = "alert_only"
                threshold_used = config.initial_threshold
                used_cloud = False

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
                used_cloud = use_cloud

            else:
                decision = router.decide(float(confidences[i]), label_name)

                if decision.used_cloud:
                    pred = _simulate_cloud_prediction(
                        int(labels[i]),
                        int(edge_preds[i]),
                        rng,
                        config.cloud_accuracy_boost,
                        num_classes,
                    )
                else:
                    pred = int(edge_preds[i])

                bytes_sent = decision.transmitted_bytes
                route = decision.route
                threshold_used = decision.threshold_used
                used_cloud = decision.used_cloud

                router.update(uploaded=decision.used_cloud, edge_correct=edge_correct)

            final_preds.append(pred)
            transmitted_bytes.append(bytes_sent)
            routes.append(route)
            thresholds.append(threshold_used)
            cloud_used.append(used_cloud)

        metrics = classification_metrics(labels, final_preds, labels=list(range(len(URBAN_SOUND_CLASSES))))
        metrics["weighted_event_score"] = weighted_event_score(labels, final_preds, class_weights)
        metrics["avg_bytes_per_clip"] = float(np.mean(transmitted_bytes))

        raw_bytes = float(config.raw_audio_bytes_per_second * clip_seconds)
        metrics["bandwidth_reduction_vs_always_cloud"] = float(1.0 - (np.mean(transmitted_bytes) / raw_bytes))
        metrics["cloud_upload_rate"] = float(np.mean(cloud_used))

        df = pd.DataFrame(
            {
                "true_label": [IDX_TO_CLASS[int(y)] for y in labels],
                "edge_pred": [IDX_TO_CLASS[int(y)] for y in edge_preds],
                "final_pred": [IDX_TO_CLASS[int(y)] for y in final_preds],
                "confidence": confidences,
                "route": routes,
                "threshold_used": thresholds,
                "transmitted_bytes": transmitted_bytes,
                "used_cloud": cloud_used,
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