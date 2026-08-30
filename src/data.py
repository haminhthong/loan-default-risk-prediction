"""
Mô-đun Nạp và Kiểm Tra Schema Dữ Liệu Tín Dụng (Data Ingestion & Validation).

Tác dụng:
- Đọc tệp CSV dữ liệu khoản vay LendingClub.
- Thực hiện Kiểm tra Schema Guard (Validation): đảm bảo đầy đủ các cột bắt buộc,
  ID duy nhất không rỗng và trạng thái khoản vay hợp lệ.
- Thực hiện Phân chia Dữ liệu theo Mốc Thời gian (Temporal Out-of-Time Split):
  ngăn chặn hiện tượng rò rỉ dữ liệu tương lai (Data Leakage).
"""

from pathlib import Path
from typing import Tuple

import pandas as pd

# Tập hợp các cột bắt buộc phải có trong dữ liệu đầu vào để đảm bảo pipeline hoạt động
REQUIRED_COLUMNS = {
    "id",
    "loan_status",
    "issue_d",
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "total_acc",
}

# Các trạng thái khoản vay được phép xử lý trong bài toán
ALLOWED_STATUSES = {"Fully Paid", "Charged Off", "Current"}


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Đọc dữ liệu CSV từ đường dẫn và kiểm tra tính hợp lệ của Schema Guard.

    Args:
        path (str | Path): Đường dẫn tới tệp CSV chứa dữ liệu khoản vay.

    Returns:
        pd.DataFrame: DataFrame đã được nạp và chuyển đổi cột issue_date sang kiểu datetime.

    Raises:
        ValueError: Nếu thiếu cột bắt buộc, trùng lặp/rỗng ID, trạng thái không hợp lệ,
                   hoặc không thể chuyển đổi ngày phát hành khoản vay.
    """
    # Đọc tệp CSV với low_memory=False để tránh cảnh báo mixed types từ pandas
    data = pd.read_csv(path, low_memory=False)

    # 1. Kiểm tra các cột bắt buộc (Schema Guard)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dữ liệu đầu vào thiếu các cột bắt buộc: {sorted(missing_columns)}")

    # 2. Kiểm tra cột định danh id (Phải đầy đủ và duy nhất)
    if data["id"].isna().any():
        raise ValueError("Cột 'id' chứa giá trị rỗng (NaN/Null).")
    if data["id"].duplicated().any():
        raise ValueError("Cột 'id' chứa các giá trị trùng lặp. Yêu cầu ID phải duy nhất.")

    # 3. Kiểm tra các giá trị trạng thái khoản vay (loan_status)
    unknown_statuses = set(data["loan_status"].dropna().unique()) - ALLOWED_STATUSES
    if unknown_statuses:
        raise ValueError(
            f"Trạng thái 'loan_status' chứa giá trị không hợp lệ: {sorted(unknown_statuses)}"
        )

    # 4. Chuyển đổi định dạng ngày phát hành khoản vay (issue_d: ví dụ 'Dec-11' -> datetime)
    data["issue_date"] = pd.to_datetime(
        data["issue_d"], format="%b-%y", errors="coerce"
    )
    if data["issue_date"].isna().any():
        raise ValueError("Tồn tại giá trị 'issue_d' không chuyển đổi được sang kiểu ngày tháng.")

    return data


def temporal_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia dữ liệu thành 3 tập Train, Validation, Test theo mốc thời gian phát hành (issue_date).

    Nguyên tắc chống rò rỉ dữ liệu (Anti-Leakage):
    - Tập Train: Các khoản vay phát hành trước 2011-01-01.
    - Tập Validation: Các khoản vay phát hành nửa đầu năm 2011 (2011-01-01 đến 2011-06-30).
    - Tập Test: Các khoản vay phát hành nửa cuối năm 2011 (từ 2011-07-01 trở đi).

    Args:
        data (pd.DataFrame): DataFrame đã có cột `issue_date`.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, validation_df, test_df)

    Raises:
        ValueError: Nếu một trong các tập bị rỗng do không đủ khoảng thời gian dữ liệu.
    """
    train = data.loc[data["issue_date"] < "2011-01-01"].copy()
    validation = data.loc[
        data["issue_date"].between("2011-01-01", "2011-06-30")
    ].copy()
    test = data.loc[data["issue_date"] >= "2011-07-01"].copy()

    # Đảm bảo cả 3 tập đều có dữ liệu
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            f"Không đủ dữ liệu cho phân chia theo thời gian. "
            f"Số lượng bản ghi: Train={len(train)}, Val={len(validation)}, Test={len(test)}"
        )

    return train, validation, test
