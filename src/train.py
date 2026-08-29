"""Huấn luyện pipeline và lưu cả mô hình lẫn ngưỡng quyết định."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_data, temporal_split
from src.analysis import calibration_table, drift_report, logistic_odds_ratios
from src.evaluate import choose_threshold, classification_metrics, slice_metrics
from src.features import TARGET, build_features, create_target

RANDOM_STATE = 42
SLICE_COLUMNS = ("grade", "home_ownership", "addr_state")


def make_pipeline(features: pd.DataFrame, estimator: Any = None) -> Pipeline:
    """Tạo pipeline tự nhận diện cột số và cột phân loại."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
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
    if estimator is None:
        estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def candidate_estimators(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Tạo các mô hình ứng viên dùng cho cả so sánh và huấn luyện."""
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="liblinear"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def compare_models(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """So sánh baseline và ứng viên trên tập train bằng CV phân tầng."""
    candidates = candidate_estimators(random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
    }
    rows = []
    for name, estimator in candidates.items():
        scores = cross_validate(
            make_pipeline(features, estimator),
            features,
            target,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("pr_auc_mean", ascending=False)
        .reset_index(drop=True)
    )


def save_reports(
    report_dir: Path,
    comparison: pd.DataFrame,
    metrics: dict[str, float],
    slices: dict[str, pd.DataFrame],
    extra_reports: dict[str, pd.DataFrame],
) -> None:
    """Lưu các báo cáo đánh giá dưới định dạng dễ kiểm tra phiên bản."""
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    for column, table in slices.items():
        table.to_csv(report_dir / f"slice_{column}.csv", index=False)
    for name, table in extra_reports.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)
    (report_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def compare_feature_sets(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """So sánh mô hình có và không có đặc trưng định giá tín dụng."""
    rows = []
    for include_pricing in (True, False):
        X_train = build_features(train_data, include_pricing)
        X_validation = build_features(validation_data, include_pricing)
        X_test = build_features(test_data, include_pricing)
        model = CalibratedClassifierCV(
            make_pipeline(X_train), method="sigmoid", cv=3, n_jobs=-1
        )
        model.fit(X_train, train_data[TARGET])
        validation_probability = model.predict_proba(X_validation)[:, 1]
        threshold, _ = choose_threshold(
            validation_data[TARGET], validation_probability
        )
        test_probability = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "feature_set": "with_pricing" if include_pricing else "without_pricing",
                "threshold": threshold,
                **classification_metrics(test_data[TARGET], test_probability, threshold),
            }
        )
    return pd.DataFrame(rows)


def train(
    data_path: Path,
    output_path: Path,
    report_dir: Path = Path("reports"),
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Chia train/validation/test, chọn ngưỡng trên validation và đánh giá test."""
    labeled = create_target(load_data(data_path))
    train_data, validation_data, test_data = temporal_split(labeled)

    X_train = build_features(train_data)
    X_validation = build_features(validation_data)
    X_test = build_features(test_data)
    y_train = train_data[TARGET]
    y_validation = validation_data[TARGET]
    y_test = test_data[TARGET]

    comparison = compare_models(X_train, y_train, random_state)
    eligible = comparison.loc[comparison["model"] != "dummy"]
    champion_name = str(eligible.iloc[0]["model"])
    champion = candidate_estimators(random_state)[champion_name]
    base_pipeline = make_pipeline(X_train, champion)

    # Hiệu chỉnh xác suất chỉ trên train; validation vẫn độc lập để chọn ngưỡng.
    pipeline = CalibratedClassifierCV(
        base_pipeline,
        method="sigmoid",
        cv=3,
        n_jobs=-1,
    )
    pipeline.fit(X_train, y_train)
    validation_probability = pipeline.predict_proba(X_validation)[:, 1]
    threshold, _ = choose_threshold(y_validation, validation_probability)
    test_probability = pipeline.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, test_probability, threshold)
    threshold_sensitivity = []
    for false_negative_cost in (2.0, 5.0, 10.0):
        selected_threshold, table = choose_threshold(
            y_validation,
            validation_probability,
            false_negative_cost=false_negative_cost,
        )
        selected = table.iloc[0]
        threshold_sensitivity.append(
            {
                "false_negative_cost": false_negative_cost,
                "false_positive_cost": 1.0,
                "threshold": selected_threshold,
                "validation_cost": selected["cost"],
                "validation_recall": selected["recall"],
                "validation_precision": selected["precision"],
            }
        )
    slices = {
        column: slice_metrics(
            X_test,
            y_test,
            test_probability,
            threshold,
            column,
        )
        for column in SLICE_COLUMNS
        if column in X_test
    }
    explanation_pipeline = make_pipeline(X_train)
    explanation_pipeline.fit(X_train, y_train)
    extra_reports = {
        "feature_set_comparison": compare_feature_sets(
            train_data, validation_data, test_data
        ),
        "threshold_sensitivity": pd.DataFrame(threshold_sensitivity),
        "calibration_test": calibration_table(y_test, test_probability),
        "drift_psi": drift_report(X_train, X_test),
        "logistic_odds_ratios": logistic_odds_ratios(explanation_pipeline),
    }
    save_reports(report_dir, comparison, metrics, slices, extra_reports)

    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "feature_columns": X_train.columns.tolist(),
        "metrics": metrics,
        "label_definition": {"Fully Paid": 0, "Charged Off": 1},
        "model_name": f"calibrated_{champion_name}",
        "random_state": random_state,
        "data_rows": len(labeled),
        "split_rows": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        "split_definition": {
            "train": "issue_date < 2011-01-01",
            "validation": "2011-01-01 <= issue_date <= 2011-06-30",
            "test": "issue_date >= 2011-07-01",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return artifact


def main() -> None:
    # Tránh lỗi Unicode trên Windows khi tên chỉ số và thông báo có dấu.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/lendingclub_2007_2011.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/loan_default_enhanced.joblib"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    artifact = train(args.data, args.output, args.report_dir)
    print(f"Đã lưu mô hình tại {args.output}")
    print(f"Ngưỡng quyết định: {artifact['threshold']:.2f}")
    print(pd.Series(artifact["metrics"]).round(4).to_string())


if __name__ == "__main__":
    main()
