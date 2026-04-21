from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], labels: List[int]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
    }


def weighted_event_score(y_true: Iterable[int], y_pred: Iterable[int], class_weights: Dict[int, float]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)

    total_weight = 0.0
    score = 0.0

    for yt, yp in zip(y_true, y_pred):
        w = float(class_weights.get(int(yt), 1.0))
        total_weight += w
        score += w * (1.0 if int(yt) == int(yp) else 0.0)

    return float(score / max(total_weight, 1e-8))


def weighted_precision_recall_f1(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_weights: Dict[int, float],
    labels: List[int],
) -> Dict[str, float]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    weights = np.asarray([float(class_weights.get(int(label), 1.0)) for label in labels], dtype=np.float64)
    denom = float(weights.sum()) if float(weights.sum()) > 0 else 1.0

    return {
        "priority_weighted_precision": float(np.dot(precisions, weights) / denom),
        "priority_weighted_recall": float(np.dot(recalls, weights) / denom),
        "priority_weighted_f1": float(np.dot(f1s, weights) / denom),
    }
