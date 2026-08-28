"""Hàm đánh giá và chọn ngưỡng theo chi phí nghiệp vụ."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
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
        _, false_positive, false_negative, _ = confusion_matrix(
            y_true, prediction, labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "threshold": threshold,
                "cost": (
                    false_negative * false_negative_cost
                    + false_positive * false_positive_cost
                ),
                "false_negative": false_negative,
                "false_positive": false_positive,
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
        "brier_score": brier_score_loss(y_true, probabilities),
    }


def slice_metrics(
    data: pd.DataFrame,
    y_true,
    probabilities,
    threshold: float,
    column: str,
) -> pd.DataFrame:
    """Đánh giá theo nhóm; chỉ báo cáo nhóm có ít nhất 30 quan sát."""
    frame = pd.DataFrame(
        {
            column: data[column].astype("string"),
            "y": y_true,
            "p": probabilities,
        }
    )
    rows = []
    for value, group in frame.groupby(column, dropna=False):
        if len(group) < 30 or group["y"].nunique() < 2:
            continue
        metrics = classification_metrics(group["y"], group["p"], threshold)
        rows.append(
            {
                column: str(value),
                "n": len(group),
                "default_rate": group["y"].mean(),
                **metrics,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("n", ascending=False)
