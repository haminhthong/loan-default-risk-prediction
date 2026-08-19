"""Hàm đánh giá và chọn ngưỡng theo chi phí nghiệp vụ."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(
    y_true,
    probabilities,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> tuple[float, pd.DataFrame]:
    """Chọn ngưỡng có tổng chi phí FN/FP thấp nhất trên tập validation."""
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        prediction = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "cost": fn * false_negative_cost + fp * false_positive_cost,
                "false_negative": fn,
                "false_positive": fp,
                "recall": recall_score(y_true, prediction, zero_division=0),
                "precision": precision_score(y_true, prediction, zero_division=0),
                "f1": f1_score(y_true, prediction, zero_division=0),
            }
        )
    table = pd.DataFrame(rows).sort_values(["cost", "threshold"])
    return float(table.iloc[0]["threshold"]), table


def classification_metrics(y_true, probabilities, threshold: float) -> dict[str, float]:
    """Tính các chỉ số phù hợp cho bài toán mất cân bằng lớp."""
    prediction = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "f1": f1_score(y_true, prediction),
        "recall": recall_score(y_true, prediction),
        "precision": precision_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
    }

