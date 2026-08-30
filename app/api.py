"""
Dịch Vụ RESTful API Chấm Điểm Rủi Ro Vỡ Nợ Khoản Vay (FastAPI Service).

Cung cấp các API endpoints phục vụ tích hợp hệ thống:
1. `/health`  - Kiểm tra trạng thái hoạt động của dịch vụ và sự tồn tại của mô hình.
2. `/info`    - Truy vấn thông tin metadata mô hình, ngưỡng quyết định và các chỉ số hiệu năng.
3. `/score`   - Chấm điểm danh sách hồ sơ vay (Batch Prediction Service).
4. `/explain` - Trích xuất các đặc trưng và hệ số Odds Ratio phục vụ giải thích mô hình.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_artifact, predict

# Xác định đường dẫn gốc dự án và đường dẫn file mô hình artifact
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "loan_default_cv.joblib"

# Khởi tạo ứng dụng FastAPI với tiêu đề và mô tả đầy đủ
app = FastAPI(
    title="Loan Default Risk Prediction API",
    description="Dịch vụ RESTful API chấm điểm rủi ro vỡ nợ tín dụng chống rò rỉ dữ liệu.",
    version="1.0.0",
)


class LoanApplication(BaseModel):
    """
    Schema định nghĩa các trường dữ liệu đầu vào của một hồ sơ đăng ký khoản vay.
    Chỉ bao gồm các thuộc tính sẵn có TẠI THỜI ĐIỂM CẤP VAY để tránh Data Leakage.
    """

    model_config = ConfigDict(extra="ignore")

    loan_amnt: float = Field(..., gt=0, description="Số tiền xin vay (USD)", examples=[10000.0])
    term: str = Field(..., description="Kỳ hạn vay (ví dụ: '36 months' hoặc '60 months')", examples=["36 months"])
    int_rate: str = Field(..., description="Lãi suất khoản vay", examples=["12.50%"])
    installment: float = Field(..., gt=0, description="Số tiền phải trả góp hàng tháng (USD)", examples=[334.54])
    grade: str = Field(..., description="Xếp hạng tín dụng gốc (A, B, C, D, E, F, G)", examples=["B"])
    sub_grade: str = Field(..., description="Phân hạng tín dụng chi tiết (B1, B2, ...)", examples=["B3"])
    emp_length: str | None = Field(default=None, description="Thời gian làm việc (ví dụ: '10+ years', '2 years')", examples=["5 years"])
    home_ownership: str = Field(..., description="Hình thức sở hữu nhà (RENT, OWN, MORTGAGE)", examples=["RENT"])
    annual_inc: float = Field(..., gt=0, description="Tổng thu nhập hàng năm (USD)", examples=[60000.0])
    verification_status: str = Field(..., description="Trạng thái xác minh thu nhập (Verified, Source Verified, Not Verified)", examples=["Verified"])
    purpose: str = Field(..., description="Mục đích sử dụng khoản vay", examples=["debt_consolidation"])
    addr_state: str = Field(..., min_length=2, max_length=2, description="Mã bang cư trú (ví dụ: CA, NY, TX)", examples=["CA"])
    dti: float | None = Field(default=None, ge=0, description="Tỷ lệ nợ trên thu nhập (Debt-to-Income ratio %)", examples=[15.2])
    delinq_2yrs: float | None = Field(default=None, ge=0, description="Số lần nợ quá hạn trong 2 năm qua", examples=[0.0])
    inq_last_6mths: float | None = Field(default=None, ge=0, description="Số lần truy vấn tín dụng trong 6 tháng gần nhất", examples=[1.0])
    open_acc: float | None = Field(default=None, ge=0, description="Số tài khoản tín dụng đang mở", examples=[10.0])
    pub_rec: float | None = Field(default=None, ge=0, description="Số kỷ luật/hồ sơ công khai tiêu cực", examples=[0.0])
    revol_bal: float | None = Field(default=None, ge=0, description="Dư nợ tín dụng quay vòng", examples=[5000.0])
    revol_util: str | None = Field(default=None, description="Tỷ lệ sử dụng hạn mức quay vòng", examples=["45.20%"])
    total_acc: float | None = Field(default=None, ge=0, description="Tổng số tài khoản tín dụng từng có", examples=[20.0])
    earliest_cr_line: str | None = Field(default=None, description="Tháng/Năm mở tài khoản tín dụng đầu tiên", examples=["Jan-00"])
    issue_d: str = Field(..., description="Tháng/Năm phát hành khoản vay", examples=["Dec-11"])


class ScoreRequest(BaseModel):
    """Schema danh sách các hồ sơ vay trong một yêu cầu chấm điểm theo lô (Batch)."""

    records: List[LoanApplication] = Field(..., min_length=1, max_length=1000, description="Danh sách các hồ sơ cần chấm điểm.")


class ScorePredictionResult(BaseModel):
    """Schema kết quả chấm điểm cho từng hồ sơ."""

    default_probability: float = Field(..., description="Xác suất rủi ro vỡ nợ (0.0 đến 1.0)")
    default_prediction: int = Field(..., description="Nhãn quyết định (1: Cảnh báo vỡ nợ, 0: Khả năng tốt)")


class ScoreResponse(BaseModel):
    """Schema phản hồi kết quả API /score."""

    model_name: str = Field(..., description="Tên mô hình Champion đang sử dụng")
    threshold: float = Field(..., description="Ngưỡng quyết định rủi ro được áp dụng")
    predictions: List[ScorePredictionResult] = Field(..., description="Danh sách kết quả dự báo tương ứng từng hồ sơ")


@lru_cache(maxsize=1)
def get_artifact() -> dict[str, Any]:
    """
    Nạp và lưu bộ nhớ đệm (Cache) tệp artifact để tái sử dụng tối ưu tốc độ giữa các API requests.

    Returns:
        dict[str, Any]: Artifact chứa mô hình pipeline và metadata.
    """
    return load_artifact(MODEL_PATH)


@app.get("/health", summary="Kiểm tra trạng thái hệ thống")
def health() -> dict[str, str]:
    """Trả về trạng thái hoạt động của dịch vụ API và file mô hình."""
    return {
        "status": "ok" if MODEL_PATH.exists() else "model_missing",
        "model_path": str(MODEL_PATH),
    }


@app.get("/info", summary="Truy vấn thông tin mô hình")
def model_info() -> dict[str, Any]:
    """Trả về metadata mô hình, ngưỡng quyết định và các chỉ số hiệu năng trên tập Test."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Chưa tìm thấy mô hình artifact. Vui lòng chạy huấn luyện trước.")
    artifact = get_artifact()
    return {
        "model_name": artifact.get("model_name"),
        "threshold": artifact.get("threshold"),
        "metrics": artifact.get("metrics"),
        "data_rows": artifact.get("data_rows"),
        "split_rows": artifact.get("split_rows"),
        "feature_columns_count": len(artifact.get("feature_columns", [])),
    }


@app.post("/score", response_model=ScoreResponse, summary="Chấm điểm rủi ro cho danh sách hồ sơ vay")
def score(payload: ScoreRequest) -> ScoreResponse:
    """
    Thực hiện chấm điểm danh sách hồ sơ tín dụng đầu vào và trả về xác suất rủi ro cùng nhãn quyết định.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Chưa có mô hình artifact. Hãy chạy 'python -m src.train' trước khi chấm điểm.",
        )
    try:
        artifact = get_artifact()
        records_data = [record.model_dump() for record in payload.records]
        input_df = pd.DataFrame(records_data)
        predictions_df = predict(input_df, artifact)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Lỗi cấu trúc hoặc dữ liệu không hợp lệ: {exc}",
        ) from exc

    return ScoreResponse(
        model_name=artifact["model_name"],
        threshold=artifact["threshold"],
        predictions=predictions_df.to_dict(orient="records"),
    )


@app.get("/explain", summary="Giải thích mô hình qua Odds Ratios")
def explain() -> dict[str, Any]:
    """
    Trả về danh sách các đặc trưng ảnh hưởng mạnh nhất đến rủi ro vỡ nợ dưới dạng Odds Ratios.
    """
    reports_dir = ROOT / "reports"
    odds_path = reports_dir / "logistic_odds_ratios.csv"
    if not odds_path.exists():
        raise HTTPException(status_code=404, detail="Chưa có báo cáo logistic_odds_ratios.csv trong thư mục reports/.")

    odds_df = pd.read_csv(odds_path)
    return {
        "description": "Odds Ratio > 1.0 làm tăng nguy cơ vỡ nợ; Odds Ratio < 1.0 làm giảm nguy cơ vỡ nợ.",
        "top_features": odds_df.to_dict(orient="records"),
    }
