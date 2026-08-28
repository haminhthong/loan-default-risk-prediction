import pandas as pd
from sklearn.dummy import DummyClassifier

from src.features import build_features
from src.predict import predict


def test_prediction_schema():
    """Kết quả dự báo phải giữ đúng số dòng và schema công bố."""
    raw = pd.DataFrame(
        {
            "emp_length": ["2 years", "3 years"],
            "term": ["36 months"] * 2,
            "annual_income": [50000, 60000],
            "installment": [200, 210],
            "issue_date": ["01-01-2021"] * 2,
        }
    )
    features = build_features(raw)
    model = DummyClassifier(strategy="prior").fit(features, [0, 1])
    artifact = {
        "pipeline": model,
        "threshold": 0.5,
        "feature_columns": features.columns.tolist(),
    }
    result = predict(raw, artifact)
    assert result.columns.tolist() == ["default_probability", "default_prediction"]
    assert len(result) == 2
