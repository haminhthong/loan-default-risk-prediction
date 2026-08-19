"""Huấn luyện pipeline và lưu cả mô hình lẫn ngưỡng quyết định."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluate import choose_threshold, classification_metrics
from src.features import TARGET, build_features, create_target


def make_pipeline(features: pd.DataFrame) -> Pipeline:
    """Tạo pipeline tự nhận diện cột số và cột phân loại."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [("preprocess", preprocessing), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]
    )


def train(data_path: Path, output_path: Path, random_state: int = 42) -> dict:
    """Chia train/validation/test, chọn ngưỡng trên validation và đánh giá test."""
    labeled = create_target(pd.read_csv(data_path))
    y = labeled[TARGET]
    X = build_features(labeled)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1765,
        stratify=y_train_val,
        random_state=random_state,
    )

    pipeline = make_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    validation_probability = pipeline.predict_proba(X_validation)[:, 1]
    threshold, _ = choose_threshold(y_validation, validation_probability)
    test_probability = pipeline.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, test_probability, threshold)

    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "feature_columns": X.columns.tolist(),
        "metrics": metrics,
        "label_definition": {"Fully Paid": 0, "Charged Off": 1},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return artifact


def main() -> None:
    # Tránh lỗi Unicode trên Windows khi tên chỉ số và thông báo có dấu.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Bank Loan Dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/loan_default.joblib"))
    args = parser.parse_args()
    artifact = train(args.data, args.output)
    print(f"Đã lưu mô hình tại {args.output}")
    print(f"Ngưỡng quyết định: {artifact['threshold']:.2f}")
    print(pd.Series(artifact["metrics"]).round(4).to_string())


if __name__ == "__main__":
    main()
