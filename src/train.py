"""
Mô-đun Huấn Luyện Pipeline Machine Learning Dự Báo Rủi Ro Vỡ Nợ (Model Training Pipeline).

Quy trình thực hiện:
1. Nạp và kiểm tra schema guard từ dữ liệu CSV.
2. Phân chia dữ liệu theo mốc thời gian (Temporal Split) thành Train / Validation / Test.
3. Tạo ma trận đặc trưng an toàn chống rò rỉ (Point-in-Time Features).
4. So sánh các mô hình ứng viên (Dummy, Logistic Regression, Random Forest) trên tập Train với 5-fold Stratified CV.
5. Lựa chọn mô hình tốt nhất (Champion Model) dựa trên chỉ số PR-AUC CV.
6. Thực hiện hiệu chỉnh xác suất (Sigmoid Calibration / CalibratedClassifierCV) trên tập Train.
7. Tối ưu ngưỡng phân loại (Decision Threshold) dựa trên tổng chi phí tổn thất tài chính trên tập Validation.
8. Mở tập Test một lần duy nhất để đánh giá hiệu năng out-of-time và phân tích phân khúc (Slice Metrics).
9. Xuất báo cáo đánh giá dạng CSV/JSON và lưu artifact hoàn chỉnh dạng joblib phục vụ Deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

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

from src.analysis import calibration_table, drift_report, logistic_odds_ratios
from src.data import load_data, temporal_split
from src.evaluate import choose_threshold, classification_metrics, slice_metrics
from src.features import TARGET, build_features, create_target

# Cấu hình giá trị ngẫu nhiên cố định để đảm bảo tính tái lập (Reproducibility)
RANDOM_STATE = 42

# Các thuộc tính dùng để phân tích hiệu năng theo nhóm (Slice Analysis)
SLICE_COLUMNS = ("grade", "home_ownership", "addr_state")

# Cấu hình mặc định: loại bỏ biến định giá (interest_rate, sub_grade) ở champion model
CHAMPION_INCLUDE_PRICING = False


def make_pipeline(features: pd.DataFrame, estimator: Any = None) -> Pipeline:
    """
    Tự động xây dựng Pipeline tiền xử lý dữ liệu và mô hình hóa.

    Bước tiền xử lý:
    - Biến định lượng (Numeric): Điền giá trị khuyết bằng Median -> Chuẩn hóa z-score (StandardScaler).
    - Biến định danh (Categorical): Điền giá trị khuyết bằng Most Frequent -> Mã hóa One-Hot (OneHotEncoder).

    Args:
        features (pd.DataFrame): Ma trận đặc trưng đầu vào.
        estimator (Any, optional): Mô hình phân loại scikit-learn. Mặc định là LogisticRegression.

    Returns:
        Pipeline: Pipeline scikit-learn đã đóng gói hoàn chỉnh.
    """
    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    categorical_cols = features.select_dtypes(exclude="number").columns.tolist()

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    if estimator is None:
        estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )

    return Pipeline(steps=[("preprocess", preprocessing), ("model", estimator)])


def candidate_estimators(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Khởi tạo danh sách các thuật toán ứng viên phục vụ thử nghiệm so sánh.

    Args:
        random_state (int): Hạt giống ngẫu nhiên.

    Returns:
        Dict[str, Any]: Từ điển chứa tên và đối tượng thuật toán.
    """
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
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
    """
    Thực hiện so sánh các mô hình ứng viên bằng 5-fold Stratified Cross-Validation trên tập Train.

    Args:
        features (pd.DataFrame): Ma trận đặc trưng tập Train.
        target (pd.Series): Nhãn tập Train.
        random_state (int): Hạt giống ngẫu nhiên.

    Returns:
        pd.DataFrame: Bảng kết quả trung bình và độ lệch chuẩn của các chỉ số qua các folds.
    """
    candidates = candidate_estimators(random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
    }

    rows = []
    for name, estimator in candidates.items():
        pipeline = make_pipeline(features, estimator)
        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
        rows.append(row)

    # Sắp xếp danh sách mô hình theo chỉ số PR-AUC giảm dần
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
    """
    Lưu trữ tất cả báo cáo kết quả thực nghiệm ra thư mục `reports/`.

    Args:
        report_dir (Path): Thư mục đích để lưu file báo cáo.
        comparison (pd.DataFrame): Kết quả so sánh mô hình qua CV.
        metrics (dict[str, float]): Metrics đánh giá trên tập Test.
        slices (dict[str, pd.DataFrame]): Kết quả phân tích theo nhóm thuộc tính.
        extra_reports (dict[str, pd.DataFrame]): Các báo cáo phụ (Drift, Calibration, Odds Ratio...).
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Báo cáo so sánh mô hình
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)

    # 2. Báo cáo đánh giá theo nhóm (Slice Analysis)
    for column, table in slices.items():
        table.to_csv(report_dir / f"slice_{column}.csv", index=False)

    # 3. Các báo cáo phụ (Calibration, Drift, Odds Ratios...)
    for name, table in extra_reports.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)

    # 4. Metrics tập Test định dạng JSON
    (report_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def compare_feature_sets(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    So sánh hiệu năng giữa hai tập đặc trưng: Có và Không có biến định giá (int_rate, sub_grade).

    Thử nghiệm này (Ablation Study) giúp đánh giá xem mô hình có thực sự học được tín hiệu rủi ro
    từ thông tin cá nhân của người vay hay chỉ đơn thuần học lại chính sách giá lãi suất của ngân hàng.

    Args:
        train_data (pd.DataFrame): Tập dữ liệu Train.
        validation_data (pd.DataFrame): Tập dữ liệu Validation.
        test_data (pd.DataFrame): Tập dữ liệu Test.

    Returns:
        pd.DataFrame: Bảng so sánh các chỉ số trên tập Test của 2 tập đặc trưng.
    """
    rows = []
    for include_pricing in (True, False):
        X_train = build_features(train_data, include_pricing)
        X_validation = build_features(validation_data, include_pricing)
        X_test = build_features(test_data, include_pricing)

        model = CalibratedClassifierCV(
            make_pipeline(X_train), method="sigmoid", cv=3, n_jobs=-1
        )
        model.fit(X_train, train_data[TARGET])

        # Chọn ngưỡng tối ưu trên tập Validation
        validation_prob = model.predict_proba(X_validation)[:, 1]
        threshold, _ = choose_threshold(validation_data[TARGET], validation_prob)

        # Đánh giá trên tập Test
        test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = classification_metrics(test_data[TARGET], test_prob, threshold)

        rows.append(
            {
                "feature_set": "with_pricing" if include_pricing else "without_pricing",
                "threshold": threshold,
                **test_metrics,
            }
        )
    return pd.DataFrame(rows)


def train(
    data_path: Path,
    output_path: Path,
    report_dir: Path = Path("reports"),
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Quy trình huấn luyện mô hình chính (End-to-End Training Pipeline).

    Args:
        data_path (Path): Đường dẫn tệp CSV dữ liệu gốc.
        output_path (Path): Đường dẫn lưu file mô hình artifact (.joblib).
        report_dir (Path): Thư mục xuất báo cáo.
        random_state (int): Hạt giống ngẫu nhiên.

    Returns:
        dict[str, Any]: Từ điển chứa thông tin artifact đã được lưu.
    """
    # 1. Nạp dữ liệu và kiểm tra Schema
    labeled = create_target(load_data(data_path))

    # 2. Chia tập dữ liệu theo mốc thời gian
    train_data, validation_data, test_data = temporal_split(labeled)

    # 3. Trích xuất đặc trưng
    X_train = build_features(train_data, CHAMPION_INCLUDE_PRICING)
    X_validation = build_features(validation_data, CHAMPION_INCLUDE_PRICING)
    X_test = build_features(test_data, CHAMPION_INCLUDE_PRICING)

    y_train = train_data[TARGET]
    y_validation = validation_data[TARGET]
    y_test = test_data[TARGET]

    # 4. So sánh các mô hình trên tập Train (5-fold CV)
    comparison = compare_models(X_train, y_train, random_state)

    # Chọn mô hình vô địch (loại bỏ DummyClassifier)
    eligible = comparison.loc[comparison["model"] != "dummy"]
    champion_name = str(eligible.iloc[0]["model"])
    champion_estimator = candidate_estimators(random_state)[champion_name]
    base_pipeline = make_pipeline(X_train, champion_estimator)

    # 5. Hiệu chỉnh xác suất bằng Sigmoid Calibration (Platt Scaling) chỉ trên tập Train
    calibrated_pipeline = CalibratedClassifierCV(
        base_pipeline,
        method="sigmoid",
        cv=3,
        n_jobs=-1,
    )
    calibrated_pipeline.fit(X_train, y_train)

    # 6. Chọn ngưỡng quyết định tối ưu trên tập Validation dựa theo chi phí nghiệp vụ (FN:FP = 5:1)
    val_prob = calibrated_pipeline.predict_proba(X_validation)[:, 1]
    optimal_threshold, _ = choose_threshold(y_validation, val_prob)

    # 7. Mở tập Test để đánh giá kết quả cuối cùng
    test_prob = calibrated_pipeline.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, test_prob, optimal_threshold)

    # Phân tích độ nhạy của ngưỡng theo các mức chi phí tổn thất khác nhau (2:1, 5:1, 10:1)
    threshold_sensitivity = []
    for fn_cost in (2.0, 5.0, 10.0):
        thresh, table = choose_threshold(
            y_validation,
            val_prob,
            false_negative_cost=fn_cost,
        )
        selected_row = table.iloc[0]
        threshold_sensitivity.append(
            {
                "false_negative_cost": fn_cost,
                "false_positive_cost": 1.0,
                "threshold": thresh,
                "validation_cost": float(selected_row["cost"]),
                "validation_recall": float(selected_row["recall"]),
                "validation_precision": float(selected_row["precision"]),
            }
        )

    # Phân tích theo nhóm thuộc tính (Slice Metrics) trên tập Test
    slices = {
        col: slice_metrics(X_test, y_test, test_prob, optimal_threshold, col)
        for col in SLICE_COLUMNS
        if col in X_test.columns
    }

    # Trích xuất hệ số Odds Ratio cho mục đích giải thích mô hình
    explanation_pipeline = make_pipeline(X_train)
    explanation_pipeline.fit(X_train, y_train)

    # Tổng hợp các báo cáo phụ
    extra_reports = {
        "feature_set_comparison": compare_feature_sets(
            train_data, validation_data, test_data
        ),
        "threshold_sensitivity": pd.DataFrame(threshold_sensitivity),
        "calibration_test": calibration_table(y_test, test_prob),
        "drift_psi": drift_report(X_train, X_test),
        "logistic_odds_ratios": logistic_odds_ratios(explanation_pipeline),
    }

    # Lưu tất cả báo cáo
    save_reports(report_dir, comparison, metrics, slices, extra_reports)

    # 8. Đóng gói mô hình và metadata vào một Artifact duy nhất
    artifact = {
        "pipeline": calibrated_pipeline,
        "threshold": optimal_threshold,
        "feature_columns": X_train.columns.tolist(),
        "metrics": metrics,
        "label_definition": {"Fully Paid": 0, "Charged Off": 1},
        "model_name": f"calibrated_{champion_name}",
        "include_pricing_features": CHAMPION_INCLUDE_PRICING,
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
    """Hàm thực thi CLI chính khi chạy lệnh `python -m src.train`."""
    # Đảm bảo stdout hỗ trợ UTF-8 để hiển thị tiếng Việt trên Windows PowerShell/CMD
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Quy trình huấn luyện mô hình dự báo rủi ro vỡ nợ khoản vay."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/lendingclub_2007_2011.csv"),
        help="Đường dẫn file CSV chứa dữ liệu khoản vay.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/loan_default_cv.joblib"),
        help="Đường dẫn xuất file mô hình artifact joblib.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Thư mục xuất báo cáo đánh giá.",
    )

    args = parser.parse_args()
    print(">>> Đang khởi tạo quy trình huấn luyện pipeline...")
    artifact = train(args.data, args.output, args.report_dir)

    print("\n==================================================")
    print(f"✅ Hoàn tất! Đã lưu mô hình tại: {args.output}")
    print(f"🎯 Mô hình Champion: {artifact['model_name']}")
    print(f"⚖️ Ngưỡng tối ưu theo chi phí (Validation): {artifact['threshold']:.2f}")
    print("--------------------------------------------------")
    print("📊 Hiệu năng đánh giá trên tập Out-of-Time Test:")
    for metric_name, val in artifact["metrics"].items():
        print(f"  - {metric_name:20s}: {val:.4f}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
