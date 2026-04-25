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


def _top2_margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def _simulate_cloud_prediction(
    true_label: int,
    edge_pred: int,
    rng: np.random.Generator,
    boost: float,
    num_classes: int,
) -> int:
    """
    Simulates a stronger cloud verifier.

    If the edge model is already correct, cloud keeps the correct label.
    If the edge model is wrong, cloud has a route-dependent chance to recover.
    Raw audio receives a larger boost than compact embedding.
    """
    if edge_pred == true_label:
        return true_label

    p_correct = min(0.98, 0.55 + float(boost))
    if rng.random() < p_correct:
        return true_label

    wrong_choices = [i for i in range(num_classes) if i != true_label]
    return int(rng.choice(wrong_choices))


def _collect_metrics(
    labels: np.ndarray,
    final_preds: List[int],
    transmitted_bytes: List[int],
    cloud_used: List[bool],
    routes: List[str],
    class_weights: Dict[int, float],
    false_positive_costs: Dict[int, float],
    clip_seconds: float,
    config,
) -> Dict[str, float]:
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

    avg_bytes = float(np.mean(transmitted_bytes))
    raw_bytes = float(config.raw_audio_bytes_per_second * clip_seconds)

    metrics["avg_bytes_per_clip"] = avg_bytes
    metrics["raw_bytes_per_clip"] = raw_bytes
    metrics["bandwidth_reduction_vs_always_cloud"] = float(1.0 - (avg_bytes / max(raw_bytes, 1.0)))

    metrics["cloud_verification_rate"] = float(np.mean(cloud_used))
    metrics["alert_only_rate"] = float(np.mean([r == "alert_only" for r in routes]))
    metrics["embedding_rate"] = float(np.mean([r == "embedding" for r in routes]))
    metrics["raw_audio_rate"] = float(np.mean([r == "raw_audio" for r in routes]))

    mistakes = [(truth, pred) for truth, pred in zip(labels, final_preds) if int(truth) != int(pred)]
    metrics["priority_false_positive_cost"] = float(
        np.mean([false_positive_costs.get(int(pred), 1.0) for _, pred in mistakes])
        if mistakes
        else 0.0
    )

    return metrics


def _write_class_report(
    output_dir: Path,
    policy_name: str,
    labels: np.ndarray,
    final_preds: List[int],
    class_weights: Dict[int, float],
) -> None:
    rows = []

    for class_idx, class_name in IDX_TO_CLASS.items():
        y_true = np.asarray(labels) == int(class_idx)
        y_pred = np.asarray(final_preds) == int(class_idx)

        tp = int(np.sum(y_true & y_pred))
        fp = int(np.sum(~y_true & y_pred))
        fn = int(np.sum(y_true & ~y_pred))
        support = int(np.sum(y_true))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        rows.append(
            {
                "policy": policy_name,
                "class_idx": class_idx,
                "class_name": class_name,
                "priority_weight": float(class_weights.get(class_idx, 1.0)),
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / f"class_report_{policy_name}.csv", index=False)


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
    margins = _top2_margin(probs)
    edge_preds = probs.argmax(axis=1)
    num_classes = probs.shape[1]

    priority_weights = config.priority_weights or {}
    class_weights = {idx: float(priority_weights.get(name, 1.0)) for idx, name in IDX_TO_CLASS.items()}
    false_positive_costs = {
        idx: float((config.false_positive_costs or priority_weights).get(name, 1.0))
        for idx, name in IDX_TO_CLASS.items()
    }

    policies = [
        "always_cloud",
        "edge_only",
        "fixed_threshold",
        "fixed_triage",
        "adaptive_triage",
    ]

    results: Dict[str, SimulationResult] = {}

    for policy_name in policies:
        router = AdaptiveRouter(
            alert_payload_bytes=config.alert_payload_bytes,
            embedding_bytes=config.embedding_bytes,
            raw_audio_bytes_per_second=config.raw_audio_bytes_per_second,
            clip_seconds=clip_seconds,
            initial_threshold=config.initial_threshold,
            min_threshold=config.min_threshold,
            max_threshold=config.max_threshold,
            budget_upload_rate=config.budget_upload_rate,
            adaptation_rate=config.adaptation_rate,
            sliding_window=config.sliding_window,
            target_edge_accuracy=config.target_edge_accuracy,
            embedding_band_width=config.embedding_band_width,
            min_top2_margin_for_alert=config.min_top2_margin_for_alert,
            margin_penalty_strength=config.margin_penalty_strength,
            priority_weights=config.priority_weights or {},
            false_positive_costs=config.false_positive_costs or {},
        )

        final_preds: List[int] = []
        transmitted_bytes: List[int] = []
        routes: List[str] = []
        thresholds: List[float] = []
        raw_thresholds: List[float] = []
        cloud_used: List[bool] = []

        for i in range(len(labels)):
            true_label = int(labels[i])
            edge_pred = int(edge_preds[i])
            edge_label_name = IDX_TO_CLASS[edge_pred]
            confidence = float(confidences[i])
            margin = float(margins[i])
            edge_correct = edge_pred == true_label

            if policy_name == "always_cloud":
                pred = _simulate_cloud_prediction(
                    true_label=true_label,
                    edge_pred=edge_pred,
                    rng=rng,
                    boost=config.raw_cloud_accuracy_boost,
                    num_classes=num_classes,
                )
                route = "raw_audio"
                bytes_sent = int(config.raw_audio_bytes_per_second * clip_seconds)
                threshold_used = config.initial_threshold
                raw_threshold_used = config.initial_threshold
                used_cloud = True

            elif policy_name == "edge_only":
                pred = edge_pred
                route = "alert_only"
                bytes_sent = int(config.alert_payload_bytes)
                threshold_used = config.initial_threshold
                raw_threshold_used = config.initial_threshold
                used_cloud = False

            elif policy_name == "fixed_threshold":
                used_cloud = confidence < config.initial_threshold

                if used_cloud:
                    pred = _simulate_cloud_prediction(
                        true_label=true_label,
                        edge_pred=edge_pred,
                        rng=rng,
                        boost=config.raw_cloud_accuracy_boost,
                        num_classes=num_classes,
                    )
                    route = "raw_audio"
                    bytes_sent = int(config.raw_audio_bytes_per_second * clip_seconds)
                else:
                    pred = edge_pred
                    route = "alert_only"
                    bytes_sent = int(config.alert_payload_bytes)

                threshold_used = config.initial_threshold
                raw_threshold_used = config.initial_threshold

            elif policy_name == "fixed_triage":
                alert_threshold = config.initial_threshold
                raw_threshold = max(config.min_threshold, alert_threshold - config.embedding_band_width)

                if confidence >= alert_threshold and margin >= config.min_top2_margin_for_alert:
                    pred = edge_pred
                    route = "alert_only"
                    bytes_sent = int(config.alert_payload_bytes)
                    used_cloud = False
                elif confidence >= raw_threshold:
                    pred = _simulate_cloud_prediction(
                        true_label=true_label,
                        edge_pred=edge_pred,
                        rng=rng,
                        boost=config.embedding_cloud_accuracy_boost,
                        num_classes=num_classes,
                    )
                    route = "embedding"
                    bytes_sent = int(config.embedding_bytes)
                    used_cloud = True
                else:
                    pred = _simulate_cloud_prediction(
                        true_label=true_label,
                        edge_pred=edge_pred,
                        rng=rng,
                        boost=config.raw_cloud_accuracy_boost,
                        num_classes=num_classes,
                    )
                    route = "raw_audio"
                    bytes_sent = int(config.raw_audio_bytes_per_second * clip_seconds)
                    used_cloud = True

                threshold_used = alert_threshold
                raw_threshold_used = raw_threshold

            else:
                decision = router.decide(
                    confidence=confidence,
                    top2_margin=margin,
                    predicted_label=edge_label_name,
                )

                if decision.route == "alert_only":
                    pred = edge_pred
                elif decision.route == "embedding":
                    pred = _simulate_cloud_prediction(
                        true_label=true_label,
                        edge_pred=edge_pred,
                        rng=rng,
                        boost=config.embedding_cloud_accuracy_boost,
                        num_classes=num_classes,
                    )
                else:
                    pred = _simulate_cloud_prediction(
                        true_label=true_label,
                        edge_pred=edge_pred,
                        rng=rng,
                        boost=config.raw_cloud_accuracy_boost,
                        num_classes=num_classes,
                    )

                route = decision.route
                bytes_sent = decision.transmitted_bytes
                threshold_used = decision.threshold_used
                raw_threshold_used = decision.raw_threshold_used
                used_cloud = decision.used_cloud

                router.update(transmitted_bytes=bytes_sent, edge_correct=edge_correct)

            final_preds.append(int(pred))
            transmitted_bytes.append(int(bytes_sent))
            routes.append(route)
            thresholds.append(float(threshold_used))
            raw_thresholds.append(float(raw_threshold_used))
            cloud_used.append(bool(used_cloud))

        metrics = _collect_metrics(
            labels=labels,
            final_preds=final_preds,
            transmitted_bytes=transmitted_bytes,
            cloud_used=cloud_used,
            routes=routes,
            class_weights=class_weights,
            false_positive_costs=false_positive_costs,
            clip_seconds=clip_seconds,
            config=config,
        )

        df = pd.DataFrame(
            {
                "true_label": [IDX_TO_CLASS[int(y)] for y in labels],
                "edge_pred": [IDX_TO_CLASS[int(y)] for y in edge_preds],
                "final_pred": [IDX_TO_CLASS[int(y)] for y in final_preds],
                "confidence": confidences,
                "top2_margin": margins,
                "route": routes,
                "threshold_used": thresholds,
                "raw_threshold_used": raw_thresholds,
                "transmitted_bytes": transmitted_bytes,
                "used_cloud": cloud_used,
                "priority_weight": [class_weights[int(y)] for y in edge_preds],
                "false_positive_cost": [false_positive_costs[int(y)] for y in edge_preds],
                "embedding_l2": np.linalg.norm(embeddings, axis=1),
                "embedding_dim": embeddings.shape[1],
            }
        )

        df.to_csv(output_dir / f"events_{policy_name}.csv", index=False)
        _write_class_report(output_dir, policy_name, labels, final_preds, class_weights)

        results[policy_name] = SimulationResult(summary=metrics, events=df)

    summary_df = pd.DataFrame([{"policy": name, **result.summary} for name, result in results.items()])
    summary_df.to_csv(output_dir / "policy_summary.csv", index=False)

    return results


def sweep_thresholds_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    clip_seconds: float,
    scaler: TemperatureScaler,
    config,
    thresholds: list[float],
    output_csv: str | Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)

    probs = scaler.transform_probs(logits)
    confidences = probs.max(axis=1)
    edge_preds = probs.argmax(axis=1)
    num_classes = probs.shape[1]

    priority_weights = config.priority_weights or {}
    class_weights = {idx: float(priority_weights.get(name, 1.0)) for idx, name in IDX_TO_CLASS.items()}
    false_positive_costs = {
        idx: float((config.false_positive_costs or priority_weights).get(name, 1.0))
        for idx, name in IDX_TO_CLASS.items()
    }

    rows: list[dict] = []

    for threshold in thresholds:
        final_preds: List[int] = []
        transmitted_bytes: List[int] = []
        cloud_used: List[bool] = []
        routes: List[str] = []

        for i in range(len(labels)):
            true_label = int(labels[i])
            edge_pred = int(edge_preds[i])
            used_cloud = float(confidences[i]) < float(threshold)

            if used_cloud:
                pred = _simulate_cloud_prediction(
                    true_label=true_label,
                    edge_pred=edge_pred,
                    rng=rng,
                    boost=config.raw_cloud_accuracy_boost,
                    num_classes=num_classes,
                )
                bytes_sent = int(config.raw_audio_bytes_per_second * clip_seconds)
                route = "raw_audio"
            else:
                pred = edge_pred
                bytes_sent = int(config.alert_payload_bytes)
                route = "alert_only"

            final_preds.append(int(pred))
            transmitted_bytes.append(int(bytes_sent))
            cloud_used.append(bool(used_cloud))
            routes.append(route)

        metrics = _collect_metrics(
            labels=labels,
            final_preds=final_preds,
            transmitted_bytes=transmitted_bytes,
            cloud_used=cloud_used,
            routes=routes,
            class_weights=class_weights,
            false_positive_costs=false_positive_costs,
            clip_seconds=clip_seconds,
            config=config,
        )

        rows.append({"threshold": float(threshold), **metrics})

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(output_csv, index=False)

    return sweep_df


@torch.no_grad()
def collect_from_model(model, loader: DataLoader, device: torch.device):
    return collect_logits(model, loader, device)