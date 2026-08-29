"""API chấm điểm theo lô, dùng đúng pipeline đã lưu khi huấn luyện."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_artifact, predict

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "loan_default_enhanced.joblib"
app = FastAPI(title="Loan Default Risk API", version="1.0.0")


class LoanApplication(BaseModel):
    """Các trường đầu vào có tại thời điểm cấp khoản vay."""

    model_config = ConfigDict(extra="ignore")

    loan_amnt: float = Field(gt=0, examples=[10000])
    term: str = Field(examples=["36 months"])
    int_rate: str = Field(examples=["12.50%"])
    installment: float = Field(gt=0, examples=[334.54])
    grade: str = Field(examples=["B"])
    sub_grade: str = Field(examples=["B3"])
    emp_length: str | None = Field(default=None, examples=["5 years"])
    home_ownership: str = Field(examples=["RENT"])
    annual_inc: float = Field(gt=0, examples=[60000])
    verification_status: str = Field(examples=["Verified"])
    purpose: str = Field(examples=["debt_consolidation"])
    addr_state: str = Field(min_length=2, max_length=2, examples=["CA"])
    dti: float | None = Field(default=None, ge=0, examples=[15.2])
    delinq_2yrs: float | None = Field(default=None, ge=0)
    inq_last_6mths: float | None = Field(default=None, ge=0)
    open_acc: float | None = Field(default=None, ge=0)
    pub_rec: float | None = Field(default=None, ge=0)
    revol_bal: float | None = Field(default=None, ge=0)
    revol_util: str | None = Field(default=None, examples=["45.20%"])
    total_acc: float | None = Field(default=None, ge=0)
    earliest_cr_line: str | None = Field(default=None, examples=["Jan-00"])
    issue_d: str = Field(examples=["Dec-11"])


class ScoreRequest(BaseModel):
    """Danh sách hồ sơ cần chấm điểm trong một yêu cầu."""

    records: list[LoanApplication] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def get_artifact() -> dict[str, Any]:
    """Nạp artifact một lần và tái sử dụng giữa các yêu cầu."""
    return load_artifact(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    """Kiểm tra nhanh trạng thái sẵn sàng của artifact."""
    return {"status": "ok" if MODEL_PATH.exists() else "model_missing"}


@app.post("/score")
def score(payload: ScoreRequest) -> dict[str, Any]:
    """Trả về xác suất và nhãn dự báo cho từng bản ghi đầu vào."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Chưa có model artifact.")
    try:
        artifact = get_artifact()
        records = [record.model_dump() for record in payload.records]
        result = predict(pd.DataFrame(records), artifact)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Dữ liệu đầu vào không hợp lệ: {exc}",
        ) from exc
    return {
        "model_name": artifact["model_name"],
        "threshold": artifact["threshold"],
        "predictions": result.to_dict(orient="records"),
    }
