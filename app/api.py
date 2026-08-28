"""API chấm điểm theo lô, dùng đúng pipeline đã lưu khi huấn luyện."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_artifact, predict

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "loan_default.joblib"
app = FastAPI(title="Loan Default Risk API", version="1.0.0")


class ScoreRequest(BaseModel):
    """Danh sách bản ghi cần chấm điểm trong một yêu cầu."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


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
        result = predict(pd.DataFrame(payload.records), get_artifact())
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Dữ liệu đầu vào không hợp lệ: {exc}",
        ) from exc
    return {"predictions": result.to_dict(orient="records")}
