"""
Các bài kiểm thử tự động (Unit Tests) cho mô-đun nạp và kiểm tra dữ liệu `src/data.py`.
"""

import pandas as pd
import pytest

from src.data import load_data, temporal_split


def valid_frame() -> pd.DataFrame:
    """Tạo mẫu dữ liệu khoản vay hợp lệ tối thiểu thuộc 3 mốc thời gian."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "loan_status": ["Fully Paid", "Charged Off", "Fully Paid"],
            "issue_d": ["Dec-10", "Mar-11", "Sep-11"],
            "loan_amnt": [5000] * 3,
            "term": ["36 months"] * 3,
            "int_rate": ["10.00%"] * 3,
            "installment": [200] * 3,
            "grade": ["B"] * 3,
            "sub_grade": ["B2"] * 3,
            "emp_length": ["2 years"] * 3,
            "home_ownership": ["RENT"] * 3,
            "annual_inc": [50000] * 3,
            "verification_status": ["Verified"] * 3,
            "purpose": ["debt_consolidation"] * 3,
            "addr_state": ["CA"] * 3,
            "dti": [0.2] * 3,
            "total_acc": [4] * 3,
        }
    )


def test_load_data_and_temporal_split(tmp_path):
    """Kiểm tra nạp dữ liệu từ CSV và chia tập theo mốc thời gian (Train, Validation, Test)."""
    path = tmp_path / "data.csv"
    valid_frame().to_csv(path, index=False)
    train, validation, test = temporal_split(load_data(path))
    assert (len(train), len(validation), len(test)) == (1, 1, 1)


def test_load_data_rejects_duplicate_id(tmp_path):
    """Kiểm tra tính năng Schema Guard phát hiện và từ chối cột ID chứa giá trị trùng lặp."""
    data = valid_frame()
    data.loc[1, "id"] = 1
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duy nhất"):
        load_data(path)
