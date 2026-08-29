from sklearn.dummy import DummyClassifier

from src.features import build_features
from src.predict import predict
from tests.test_features import sample_data


def test_prediction_schema():
    """Kết quả dự báo phải giữ đúng số dòng và schema công bố."""
    raw = sample_data().iloc[:2]
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
