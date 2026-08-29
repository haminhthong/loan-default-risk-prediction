"""Các báo cáo giải thích, hiệu chỉnh và dịch chuyển dữ liệu."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_table(
    target,
    probabilities,
    bins: int = 10,
) -> pd.DataFrame:
    """Tổng hợp xác suất dự báo và tỷ lệ vỡ nợ theo các khoảng xác suất."""
    frame = pd.DataFrame({"target": target, "probability": probabilities})
    frame["bin"] = pd.cut(
        frame["probability"],
        bins=np.linspace(0, 1, bins + 1),
        include_lowest=True,
    )
    return (
        frame.groupby("bin", observed=True)
        .agg(
            observations=("target", "size"),
            mean_probability=("probability", "mean"),
            default_rate=("target", "mean"),
        )
        .reset_index()
        .assign(bin=lambda data: data["bin"].astype("string"))
    )


def population_stability_index(
    train: pd.Series,
    test: pd.Series,
    bins: int = 10,
) -> float:
    """Tính PSI cho một biến số hoặc biến phân loại."""
    epsilon = 1e-6
    if pd.api.types.is_numeric_dtype(train):
        edges = np.unique(train.dropna().quantile(np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        train_group = pd.cut(train, edges, include_lowest=True).astype("string")
        test_group = pd.cut(test, edges, include_lowest=True).astype("string")
    else:
        train_group = train.astype("string").fillna("<missing>")
        test_group = test.astype("string").fillna("<missing>")

    categories = pd.Index(train_group.unique()).union(pd.Index(test_group.unique()))
    train_share = train_group.value_counts(normalize=True).reindex(categories, fill_value=0)
    test_share = test_group.value_counts(normalize=True).reindex(categories, fill_value=0)
    train_share = train_share.clip(lower=epsilon)
    test_share = test_share.clip(lower=epsilon)
    return float(((test_share - train_share) * np.log(test_share / train_share)).sum())


def drift_report(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Xếp hạng mức dịch chuyển phân phối giữa train và test."""
    rows = [
        {
            "feature": column,
            "psi": population_stability_index(train[column], test[column]),
            "train_missing_rate": train[column].isna().mean(),
            "test_missing_rate": test[column].isna().mean(),
        }
        for column in train.columns
    ]
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def logistic_odds_ratios(pipeline, top_n: int = 30) -> pd.DataFrame:
    """Trích hệ số và odds ratio từ pipeline Logistic Regression đã fit."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]
    report = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "absolute_coefficient": np.abs(coefficients),
        }
    )
    return report.nlargest(top_n, "absolute_coefficient").drop(
        columns="absolute_coefficient"
    )
