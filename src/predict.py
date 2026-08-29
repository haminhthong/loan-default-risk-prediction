"""Nạp artifact và dự báo xác suất vỡ nợ cho dữ liệu mới."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features import build_features


def load_artifact(
    path: str | Path = "artifacts/loan_default_enhanced.joblib",
) -> dict[str, Any]:
    """Nạp artifact gồm pipeline, ngưỡng và metadata huấn luyện."""
    return joblib.load(path)


def predict(data: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Trả về xác suất và quyết định theo ngưỡng đã chọn trên validation."""
    features = build_features(data)
    features = features.reindex(columns=artifact["feature_columns"])
    probability = artifact["pipeline"].predict_proba(features)[:, 1]
    return pd.DataFrame(
        {
            "default_probability": probability,
            "default_prediction": (probability >= artifact["threshold"]).astype(int),
        },
        index=data.index,
    )
