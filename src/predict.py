"""
Mô-đun Nạp Artifact và Suy Luận Dự Báo Rủi Ro (Model Inference).

Nhiệm vụ:
1. Nạp file mô hình artifact (.joblib) đã đóng gói.
2. Trích xuất đặc trưng phù hợp từ dữ liệu mới.
3. Tái sắp xếp cột dữ liệu đúng chuẩn ma trận đã dùng lúc huấn luyện.
4. Dự báo xác suất vỡ nợ và phân loại nhãn nhị phân theo ngưỡng quyết định đã chọn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features import build_features


def load_artifact(
    path: str | Path = "artifacts/loan_default_cv.joblib",
) -> dict[str, Any]:
    """
    Nạp tệp artifact chứa pipeline mô hình, ngưỡng quyết định và danh sách cột đặc trưng.

    Args:
        path (str | Path): Đường dẫn file joblib. Mặc định 'artifacts/loan_default_cv.joblib'.

    Returns:
        dict[str, Any]: Từ điển chứa các đối tượng đã được lưu trong quá trình huấn luyện.

    Raises:
        FileNotFoundError: Nếu tệp mô hình không tồn tại.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp artifact tại đường dẫn: {artifact_path}")
    return joblib.load(artifact_path)


def predict(data: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """
    Thực hiện dự báo xác suất vỡ nợ và đưa ra quyết định tín dụng cho dữ liệu khoản vay mới.

    Args:
        data (pd.DataFrame): DataFrame chứa dữ liệu hồ sơ khoản vay cần chấm điểm.
        artifact (dict[str, Any]): Artifact mô hình được nạp từ `load_artifact`.

    Returns:
        pd.DataFrame: DataFrame gồm 2 cột:
            - `default_probability`: Xác suất rủi ro vỡ nợ (từ 0.0 đến 1.0).
            - `default_prediction`: Nhãn dự báo (1: Rủi ro vỡ nợ / Cảnh báo, 0: Khả năng cao thanh toán đủ).
    """
    include_pricing = artifact.get("include_pricing_features", False)

    # 1. Trích xuất đặc trưng theo cấu hình của mô hình
    features = build_features(data, include_pricing=include_pricing)

    # 2. Tái sắp xếp ma trận đặc trưng theo đúng thứ tự các cột đã huấn luyện
    features = features.reindex(columns=artifact["feature_columns"])

    # 3. Dự báo xác suất thuộc lớp 1 (Charged Off / Vỡ nợ)
    probability = artifact["pipeline"].predict_proba(features)[:, 1]

    # 4. Đưa ra nhãn quyết định theo ngưỡng đã chọn từ tập Validation
    threshold = artifact["threshold"]
    prediction = (probability >= threshold).astype(int)

    return pd.DataFrame(
        {
            "default_probability": probability,
            "default_prediction": prediction,
        },
        index=data.index,
    )
