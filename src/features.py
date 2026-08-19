"""Tạo đặc trưng chỉ từ thông tin có tại thời điểm cấp khoản vay."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "default_flag"
FINAL_STATUSES = {"Fully Paid": 0, "Charged Off": 1}

# Các cột này có sau giải ngân hoặc là định danh, không được đưa vào mô hình.
LEAKAGE_COLUMNS = [
    "loan_status",
    "total_payment",
    "last_payment_date",
    "next_payment_date",
    "last_credit_pull_date",
    "id",
    "member_id",
]

EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}


def create_target(data: pd.DataFrame) -> pd.DataFrame:
    """Giữ các khoản vay đã có kết quả cuối cùng và tạo nhãn nhị phân."""
    if "loan_status" not in data.columns:
        raise ValueError("Thiếu cột bắt buộc 'loan_status'.")
    result = data.loc[data["loan_status"].isin(FINAL_STATUSES)].copy()
    result[TARGET] = result["loan_status"].map(FINAL_STATUSES).astype("int8")
    return result


def build_features(data: pd.DataFrame, include_pricing: bool = True) -> pd.DataFrame:
    """Tạo ma trận đặc trưng; không thay đổi DataFrame đầu vào."""
    result = data.copy()
    result["emp_length_years"] = result["emp_length"].map(EMP_LENGTH_MAP)
    result["term_months"] = pd.to_numeric(
        result["term"].astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    result["log_annual_income"] = np.log1p(result["annual_income"].clip(lower=0))
    result["installment_income_ratio"] = (
        12 * result["installment"] / result["annual_income"].replace(0, np.nan)
    )
    result["issue_month"] = pd.to_datetime(
        result["issue_date"], dayfirst=True, errors="coerce"
    ).dt.month

    result = result.drop(
        columns=LEAKAGE_COLUMNS + [TARGET, "emp_length", "term", "issue_date", "emp_title"],
        errors="ignore",
    )
    if not include_pricing:
        result = result.drop(columns=["int_rate", "sub_grade"], errors="ignore")
    return result
