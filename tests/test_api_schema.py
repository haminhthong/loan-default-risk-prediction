"""
Các bài kiểm thử tự động (Unit Tests) cho Pydantic Schema kiểm tra dữ liệu đầu vào API `app/api.py`.
"""

import pytest
from pydantic import ValidationError

from app.api import LoanApplication


def valid_application() -> dict:
    """Tạo mẫu dữ liệu hồ sơ xin vay hợp lệ tối thiểu cho Pydantic Schema."""
    return {
        "loan_amnt": 10000,
        "term": "36 months",
        "int_rate": "12.50%",
        "installment": 334.54,
        "grade": "B",
        "sub_grade": "B3",
        "home_ownership": "RENT",
        "annual_inc": 60000,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "addr_state": "CA",
        "issue_d": "Dec-11",
    }


def test_api_schema_accepts_valid_application():
    """Kiểm tra Pydantic Schema chấp nhận hồ sơ đầy đủ các trường hợp lệ."""
    assert LoanApplication(**valid_application()).loan_amnt == 10000


def test_api_schema_rejects_negative_loan_amount():
    """Kiểm tra Pydantic Schema từ chối khi số tiền xin vay nhận giá trị âm (loan_amnt < 0)."""
    application = valid_application()
    application["loan_amnt"] = -1
    with pytest.raises(ValidationError):
        LoanApplication(**application)
