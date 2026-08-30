"""
Các bài kiểm thử tự động (Unit Tests) cho mô-đun tạo nhãn và trích xuất đặc trưng `src/features.py`.
"""

import pandas as pd

from src.features import TARGET, build_features, create_target


def sample_data() -> pd.DataFrame:
    """Tạo mẫu dữ liệu chứa cả trạng thái đã kết thúc và trạng thái đang chạy ('Current')."""
    return pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Current"],
            "loan_amnt": [5000, 8000, 6000],
            "term": ["36 months", "60 months", "36 months"],
            "int_rate": ["10.00%", "15.50%", "12.00%"],
            "installment": [200, 250, 220],
            "grade": ["B", "C", "B"],
            "sub_grade": ["B2", "C3", "B4"],
            "emp_length": ["2 years", "< 1 year", None],
            "home_ownership": ["RENT"] * 3,
            "annual_inc": [50000, 30000, 40000],
            "verification_status": ["Verified"] * 3,
            "purpose": ["car"] * 3,
            "addr_state": ["CA"] * 3,
            "dti": [0.1, 0.2, 0.15],
            "revol_util": ["20.00%", "50.00%", None],
            "total_acc": [4, 6, 5],
            "earliest_cr_line": ["Jan-00"] * 3,
            "issue_date": pd.to_datetime(["2010-01-01"] * 3),
        }
    )


def test_target_excludes_current():
    """Kiểm tra việc loại bỏ khoản vay chưa kết thúc ('Current') và ánh xạ nhãn đúng (0 và 1)."""
    assert create_target(sample_data())[TARGET].tolist() == [0, 1]


def test_feature_engineering_is_point_in_time_safe():
    """Kiểm tra đặc trưng được trích xuất an toàn, chuyển đổi phần trăm đúng và loại bỏ nhãn/trạng thái."""
    features = build_features(create_target(sample_data()))
    assert TARGET not in features
    assert "loan_status" not in features
    assert features["interest_rate"].tolist() == [0.10, 0.155]
