"""
Các bài kiểm thử tự động (Unit Tests) cho mô-đun đánh giá hiệu năng và chọn ngưỡng `src/evaluate.py`.
"""

import numpy as np

from src.evaluate import choose_threshold, classification_metrics


def test_high_fn_cost_prefers_recall():
    """Kiểm tra chi phí bỏ sót Nợ xấu (FN) cao sẽ tự động ưu tiên ngưỡng có Recall cao hơn."""
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.9])
    threshold, _ = choose_threshold(
        target,
        probabilities,
        false_negative_cost=10.0,
        false_positive_cost=1.0,
    )
    assert threshold <= 0.45
    assert classification_metrics(target, probabilities, threshold)["recall"] == 1.0
