"""Nạp dữ liệu LendingClub và chia tập theo thời gian phát hành."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "id", "loan_status", "issue_d", "loan_amnt", "term", "int_rate",
    "installment", "grade", "sub_grade", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "purpose", "addr_state", "dti",
    "total_acc",
}
ALLOWED_STATUSES = {"Fully Paid", "Charged Off", "Current"}


def load_data(path: str | Path) -> pd.DataFrame:
    """Đọc CSV và dừng sớm nếu schema, ID hoặc trạng thái không hợp lệ."""
    data = pd.read_csv(path, low_memory=False)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing_columns)}")
    if data["id"].isna().any() or data["id"].duplicated().any():
        raise ValueError("Cột id phải đầy đủ và duy nhất.")

    unknown_statuses = set(data["loan_status"].dropna().unique()) - ALLOWED_STATUSES
    if unknown_statuses:
        raise ValueError(
            f"loan_status có giá trị chưa được định nghĩa: {sorted(unknown_statuses)}"
        )

    data["issue_date"] = pd.to_datetime(
        data["issue_d"], format="%b-%y", errors="coerce"
    )
    if data["issue_date"].isna().any():
        raise ValueError("Có issue_d không chuyển đổi được sang ngày.")
    return data


def temporal_split(data: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Chia train/validation/test theo thời gian, không xáo trộn tương lai."""
    train = data.loc[data["issue_date"] < "2011-01-01"].copy()
    validation = data.loc[
        data["issue_date"].between("2011-01-01", "2011-06-30")
    ].copy()
    test = data.loc[data["issue_date"] >= "2011-07-01"].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Không đủ dữ liệu cho cách chia theo thời gian đã định nghĩa.")
    return train, validation, test
