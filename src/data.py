"""Nạp và kiểm định schema dữ liệu trước khi mô hình hóa."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "id",
    "member_id",
    "loan_status",
    "annual_income",
    "dti",
    "installment",
    "int_rate",
    "loan_amount",
    "total_acc",
    "emp_length",
    "term",
    "issue_date",
}
ALLOWED_STATUSES = {"Fully Paid", "Charged Off", "Current"}


def load_and_validate(path: str | Path) -> pd.DataFrame:
    """Đọc CSV và dừng sớm khi schema hoặc khóa định danh không hợp lệ."""
    data = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")
    if data["id"].isna().any() or data["id"].duplicated().any():
        raise ValueError("Cột id phải đầy đủ và duy nhất.")

    unknown = set(data["loan_status"].dropna().unique()) - ALLOWED_STATUSES
    if unknown:
        raise ValueError(f"loan_status có giá trị chưa được định nghĩa: {sorted(unknown)}")
    return data
