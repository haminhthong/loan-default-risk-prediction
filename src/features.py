"""Tạo nhãn và đặc trưng chỉ từ thông tin có tại thời điểm cấp vay."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "default_flag"
FINAL_STATUS_MAP = {"Fully Paid": 0, "Charged Off": 1}

MODEL_COLUMNS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "addr_state", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "earliest_cr_line", "issue_date",
]


def create_target(data: pd.DataFrame) -> pd.DataFrame:
    """Loại khoản vay chưa kết thúc và ánh xạ trạng thái thành nhãn nhị phân."""
    final_loans = data.loc[data["loan_status"].isin(FINAL_STATUS_MAP)].copy()
    final_loans[TARGET] = (
        final_loans["loan_status"].map(FINAL_STATUS_MAP).astype("int8")
    )
    return final_loans


def _parse_percentage(series: pd.Series) -> pd.Series:
    """Đổi chuỗi phần trăm như `10.65%` thành tỷ lệ số thực `0.1065`."""
    return pd.to_numeric(
        series.astype("string").str.rstrip("%"), errors="coerce"
    ).div(100)


def build_features(
    data: pd.DataFrame,
    include_pricing: bool = True,
) -> pd.DataFrame:
    """Tạo ma trận đặc trưng mà không thay đổi DataFrame đầu vào."""
    source = data.copy()
    if "issue_date" not in source and "issue_d" in source:
        source["issue_date"] = pd.to_datetime(
            source["issue_d"], format="%b-%y", errors="coerce"
        )

    columns = [column for column in MODEL_COLUMNS if column in source.columns]
    features = source.loc[:, columns].copy()

    features["term_months"] = pd.to_numeric(
        features.pop("term").astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    features["interest_rate"] = _parse_percentage(features.pop("int_rate"))
    features["revolving_utilization"] = _parse_percentage(features.pop("revol_util"))
    annual_income = features.pop("annual_inc")
    features["log_annual_income"] = np.log1p(annual_income.clip(lower=0))
    features["installment_income_ratio"] = (
        12 * features["installment"] / annual_income.replace(0, np.nan)
    )

    earliest_credit = pd.to_datetime(
        features.pop("earliest_cr_line"), format="%b-%y", errors="coerce"
    )
    issue_date = features.pop("issue_date")
    # `%y` có thể hiểu 1968 thành 2068; lùi một thế kỷ nếu ngày mở tín dụng
    # nằm sau ngày cấp khoản vay.
    earliest_credit = earliest_credit.where(
        earliest_credit <= issue_date,
        earliest_credit - pd.DateOffset(years=100),
    )
    features["credit_history_years"] = (
        (issue_date - earliest_credit).dt.days / 365.25
    )
    features["issue_month"] = issue_date.dt.month

    if not include_pricing:
        features = features.drop(
            columns=["interest_rate", "sub_grade"], errors="ignore"
        )
    return features
