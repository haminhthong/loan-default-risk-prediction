"""
Mô-đun Tạo Nhãn và Trích Xuất Đặc Trưng Chống Rò Rỉ Dữ Liệu (Feature Engineering).

Nguyên tắc quan trọng:
- Chỉ sử dụng các thuộc tính có sẵn TẠI THỜI ĐIỂM XEM XÉT CẤP VAY (Point-in-Time Features).
- Loại bỏ hoàn toàn các trường thông tin phát sinh sau khi giải ngân (như tổng tiền đã trả,
  ngày trả gần nhất, v.v.) để tránh Rò rỉ Dữ liệu (Data Leakage).
- Tùy chọn loại bỏ các thuộc tính định giá sẵn có (interest rate, sub-grade) để kiểm tra
  xem mô hình có học được các yếu tố rủi ro độc lập thay vì chỉ học lại chính sách giá cũ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Tên cột nhãn mục tiêu trong mô hình
TARGET = "default_flag"

# Ánh xạ trạng thái khoản vay kết thúc sang nhãn nhị phân (0: Thanh toán đủ, 1: Vỡ nợ)
FINAL_STATUS_MAP = {"Fully Paid": 0, "Charged Off": 1}

# Danh sách các cột đầu vào hợp lệ tại thời điểm phê duyệt khoản vay
MODEL_COLUMNS = [
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
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "earliest_cr_line",
    "issue_date",
]


def create_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Lọc bỏ các khoản vay chưa kết thúc ('Current') và tạo cột nhãn nhị phân `default_flag`.

    Args:
        data (pd.DataFrame): Dữ liệu gốc chứa cột `loan_status`.

    Returns:
        pd.DataFrame: Dữ liệu đã lọc chỉ gồm khoản vay đã hoàn tất, bổ sung cột `default_flag`.
    """
    # Chỉ giữ các khoản vay đã kết thúc kết quả: Fully Paid hoặc Charged Off
    final_loans = data.loc[data["loan_status"].isin(FINAL_STATUS_MAP)].copy()

    # Ánh xạ nhãn: Fully Paid -> 0 (Âm tính), Charged Off -> 1 (Dương tính - Rủi ro)
    final_loans[TARGET] = (
        final_loans["loan_status"].map(FINAL_STATUS_MAP).astype("int8")
    )
    return final_loans


def _parse_percentage(series: pd.Series) -> pd.Series:
    """
    Chuyển đổi chuỗi phần trăm (ví dụ '12.50%') thành số thực tương ứng (0.1250).

    Args:
        series (pd.Series): Cột chứa dữ liệu kiểu chuỗi phần trăm.

    Returns:
        pd.Series: Cột dạng float biểu diễn tỷ lệ thực.
    """
    return pd.to_numeric(
        series.astype("string").str.rstrip("%"), errors="coerce"
    ).div(100)


def build_features(
    data: pd.DataFrame,
    include_pricing: bool = True,
) -> pd.DataFrame:
    """
    Trích xuất và tính toán các đặc trưng (Feature Engineering) từ dữ liệu đầu vào.

    Các đặc trưng được tạo mới:
    - `term_months`: Số tháng vay (chuyển từ chuỗi '36 months' -> 36).
    - `interest_rate`: Lãi suất dạng số thực (0.125 thay vì '12.5%').
    - `revolving_utilization`: Tỷ lệ sử dụng hạn mức tín dụng dạng số thực.
    - `log_annual_income`: Logarithm tự nhiên của thu nhập hàng năm (biến đổi giảm lệch phải).
    - `installment_income_ratio`: Tỷ lệ tổng tiền trả góp hàng năm trên tổng thu nhập.
    - `credit_history_years`: Thâm niên tín dụng (tính từ mở khoản tín dụng đầu tiên tới khi cấp vay).
    - `issue_month`: Tháng phát hành khoản vay.

    Args:
        data (pd.DataFrame): Dữ liệu khoản vay đầu vào.
        include_pricing (bool): Có giữ lại biến lãi suất và sub-grade hay không (Default: True).

    Returns:
        pd.DataFrame: Ma trận đặc trưng sẵn sàng cho tiền xử lý và huấn luyện mô hình.
    """
    source = data.copy()

    # Tự động tạo cột issue_date nếu chưa có sẵn từ cột issue_d
    if "issue_date" not in source and "issue_d" in source:
        source["issue_date"] = pd.to_datetime(
            source["issue_d"], format="%b-%y", errors="coerce"
        )

    # Lọc lấy danh sách các cột thuộc MODEL_COLUMNS có mặt trong DataFrame
    columns = [column for column in MODEL_COLUMNS if column in source.columns]
    features = source.loc[:, columns].copy()

    # 1. Trích xuất số tháng vay từ chuỗi '36 months' hoặc '60 months'
    if "term" in features.columns:
        features["term_months"] = pd.to_numeric(
            features.pop("term").astype("string").str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )

    # 2. Xử lý định dạng phần trăm cho lãi suất và tỷ lệ sử dụng tín dụng
    if "int_rate" in features.columns:
        features["interest_rate"] = _parse_percentage(features.pop("int_rate"))
    if "revol_util" in features.columns:
        features["revolving_utilization"] = _parse_percentage(features.pop("revol_util"))

    # 3. Biến đổi thu nhập hàng năm (log scale & tỷ lệ trả góp / thu nhập)
    if "annual_inc" in features.columns:
        annual_income = features.pop("annual_inc")
        features["log_annual_income"] = np.log1p(annual_income.clip(lower=0))
        if "installment" in features.columns:
            features["installment_income_ratio"] = (
                12 * features["installment"] / annual_income.replace(0, np.nan)
            )

    # 4. Tính toán thâm niên tài khoản tín dụng (credit_history_years)
    if "earliest_cr_line" in features.columns and "issue_date" in features.columns:
        earliest_credit = pd.to_datetime(
            features.pop("earliest_cr_line"), format="%b-%y", errors="coerce"
        )
        issue_date = features.pop("issue_date")

        # Xử lý vấn đề mốc năm 2 chữ số (ví dụ: '%y' có thể nhận diện năm 1968 thành 2068)
        # Nếu ngày mở tài khoản tín dụng lớn hơn ngày cấp vay, lùi lại 100 năm
        earliest_credit = earliest_credit.where(
            earliest_credit <= issue_date,
            earliest_credit - pd.DateOffset(years=100),
        )
        features["credit_history_years"] = (
            (issue_date - earliest_credit).dt.days / 365.25
        )
        features["issue_month"] = issue_date.dt.month
    elif "issue_date" in features.columns:
        issue_date = features.pop("issue_date")
        features["issue_month"] = issue_date.dt.month

    # 5. Loại bỏ thuộc tính định giá nếu include_pricing=False (để thử nghiệm ablation study)
    if not include_pricing:
        features = features.drop(
            columns=["interest_rate", "sub_grade"], errors="ignore"
        )

    return features
