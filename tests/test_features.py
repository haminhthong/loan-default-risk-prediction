import pandas as pd

from src.features import TARGET, build_features, create_target


def sample_data() -> pd.DataFrame:
    """Tạo dữ liệu nhỏ chứa đủ ba trạng thái khoản vay."""
    return pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Current"],
            "emp_length": ["2 years", "< 1 year", None],
            "term": ["36 months"] * 3,
            "annual_income": [60000, 30000, 50000],
            "installment": [200, 150, 180],
            "issue_date": ["01-01-2021"] * 3,
            "total_payment": [100, 50, 20],
            "id": [1, 2, 3],
            "member_id": [11, 12, 13],
            "emp_title": ["A", "B", "C"],
        }
    )


def test_target_excludes_current():
    assert create_target(sample_data())[TARGET].tolist() == [0, 1]


def test_features_block_label_and_post_outcome_columns():
    features = build_features(create_target(sample_data()))
    forbidden_columns = {
        TARGET,
        "loan_status",
        "total_payment",
        "id",
        "member_id",
    }
    assert not forbidden_columns & set(features.columns)
