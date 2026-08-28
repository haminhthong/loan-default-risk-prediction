import pandas as pd
import pytest

from src.data import load_and_validate


def valid_frame() -> pd.DataFrame:
    """Tạo một bản ghi tối thiểu thỏa schema đầu vào."""
    return pd.DataFrame(
        {
            "id": [1],
            "member_id": [2],
            "loan_status": ["Fully Paid"],
            "annual_income": [50000],
            "dti": [0.2],
            "installment": [200],
            "int_rate": [0.1],
            "loan_amount": [5000],
            "total_acc": [4],
            "emp_length": ["2 years"],
            "term": ["36 months"],
            "issue_date": ["01-01-2021"],
        }
    )


def test_schema_validation_accepts_valid_csv(tmp_path):
    path = tmp_path / "data.csv"
    valid_frame().to_csv(path, index=False)
    assert len(load_and_validate(path)) == 1


def test_schema_validation_rejects_duplicate_id(tmp_path):
    data = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duy nhất"):
        load_and_validate(path)
