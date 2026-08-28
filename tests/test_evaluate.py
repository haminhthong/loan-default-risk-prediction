import numpy as np

from src.evaluate import choose_threshold, classification_metrics


def test_high_fn_cost_prefers_recall():
    """Chi phí bỏ sót cao phải ưu tiên ngưỡng có recall tối đa."""
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.9])
    threshold, _ = choose_threshold(
        target,
        probabilities,
        false_negative_cost=10,
        false_positive_cost=1,
    )
    assert threshold <= 0.45
    assert classification_metrics(target, probabilities, threshold)["recall"] == 1.0
