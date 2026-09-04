"""
Mô-đun Đánh Giá Hiệu Năng Mô Hình và Tối Ưu Ngưỡng Quyết Định (Cost-Sensitive Evaluation).

Ý nghĩa nghiệp vụ:
- Bài toán dự báo rủi ro vỡ nợ là bài toán phân loại nhị phân mất cân bằng lớp.
- Chi phí cho một ca Bỏ sót nợ xấu (False Negative - FN) thường lớn hơn rất nhiều so với
  chi phí Báo động nhầm một khách hàng tốt (False Positive - FP).
- Ngưỡng tối ưu được chọn dựa trên bài toán tối thiểu hóa Tổng chi phí tổn thất tài chính
  trên tập Validation, thay vì cố định ở mức 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(
    y_true,
    probabilities,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> tuple[float, pd.DataFrame]:
    """
    Tìm ngưỡng phân loại (Decision Threshold) tối thiểu hóa tổng chi phí nghiệp vụ trên tập Validation.

    Công thức chi phí:
        Total Cost = (False Negatives * FN_Cost) + (False Positives * FP_Cost)

    Args:
        y_true: Nhãn thực tế (0 hoặc 1).
        probabilities: Xác suất rủi ro vỡ nợ dự báo bởi mô hình.
        false_negative_cost (float): Hệ số chi phí khi bỏ sót nợ xấu (Mặc định: 5.0).
        false_positive_cost (float): Hệ số chi phí khi từ chối nhầm nhãn tốt (Mặc định: 1.0).

    Returns:
        tuple[float, pd.DataFrame]: (ngưỡng_tối_ưu, bảng_thống_kê_chi_phí_theo_ngưỡng)
    """
    rows = []
    # Duyệt qua các ngưỡng quyết định từ 0.01 đến 0.99 với bước nhảy 0.01
    for threshold in np.linspace(0.01, 0.99, 99):
        prediction = (probabilities >= threshold).astype(int)
        # Tính toán ma trận nhầm lẫn (Confusion Matrix)
        _, false_positive, false_negative, _ = confusion_matrix(
            y_true, prediction, labels=[0, 1]
        ).ravel()

        # Tính tổng chi phí nghiệp vụ
        total_cost = (
            false_negative * false_negative_cost
            + false_positive * false_positive_cost
        )

        rows.append(
            {
                "threshold": threshold,
                "cost": total_cost,
                "false_negative": false_negative,
                "false_positive": false_positive,
                "recall": recall_score(y_true, prediction, zero_division=0),
                "precision": precision_score(y_true, prediction, zero_division=0),
                "f1": f1_score(y_true, prediction, zero_division=0),
            }
        )

    # Sắp xếp bảng kết quả theo tổng chi phí tăng dần
    table = pd.DataFrame(rows).sort_values(["cost", "threshold"])
    best_threshold = float(table.iloc[0]["threshold"])
    return best_threshold, table


def classification_metrics(y_true, probabilities, threshold: float) -> dict[str, float]:
    """
    Tính toán tập hợp các chỉ số đánh giá toàn diện cho bài toán mất cân bằng lớp.

    Các chỉ số được tính:
    - `pr_auc`: Diện tích dưới đường Precision-Recall Curve (Metric chính bài toán mất cân bằng).
    - `roc_auc`: Diện tích dưới đường ROC Curve.
    - `f1`: F1-Score tại ngưỡng chọn.
    - `recall`: Tỷ lệ phát hiện đúng khoản nợ xấu (Sensitivity / True Positive Rate).
    - `precision`: Độ chính xác khi cảnh báo rủi ro vỡ nợ.
    - `balanced_accuracy`: Độ chính xác trung bình giữa hai lớp.
    - `brier_score`: Độ chính xác của hiệu chỉnh xác suất (Brier Score càng thấp càng tốt).

    Args:
        y_true: Nhãn thực tế.
        probabilities: Xác suất dự báo rủi ro.
        threshold (float): Ngưỡng phân loại được chọn.

    Returns:
        dict[str, float]: Từ điển chứa giá trị các chỉ số đo lường.
    """
    prediction = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def slice_metrics(
    data: pd.DataFrame,
    y_true,
    probabilities,
    threshold: float,
    column: str,
) -> pd.DataFrame:
    """
    Đánh giá hiệu năng mô hình chia theo từng nhóm/phân khúc khách hàng (Subgroup/Slice Analysis).

    Giúp phát hiện sự thiên vị (bias) hoặc hiệu năng kém ở một số thuộc tính đặc thù
    (như hạng tín dụng grade, tình trạng nhà ở home_ownership, bang residence).

    Args:
        data (pd.DataFrame): DataFrame chứa thuộc tính phân nhóm.
        y_true: Nhãn thực tế.
        probabilities: Xác suất rủi ro.
        threshold (float): Ngưỡng phân loại.
        column (str): Tên cột dùng để phân nhóm (ví dụ: 'grade').

    Returns:
        pd.DataFrame: Bảng kết quả chỉ số cho các nhóm có đủ quy mô mẫu (>= 30 mẫu).
    """
    frame = pd.DataFrame(
        {
            column: data[column].astype("string"),
            "y": y_true,
            "p": probabilities,
        }
    )
    rows = []
    for value, group in frame.groupby(column, dropna=False):
        # Bỏ qua phân khúc có ít hơn 30 mẫu hoặc chỉ có 1 lớp duy nhất
        if len(group) < 30 or group["y"].nunique() < 2:
            continue

        metrics = classification_metrics(group["y"], group["p"], threshold)
        rows.append(
            {
                column: str(value),
                "n": len(group),
                "default_rate": float(group["y"].mean()),
                **metrics,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def bootstrap_metric_ci(
    y_true,
    probabilities,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Tính toán khoảng tin cậy Bootstrap (Bootstrap Confidence Interval) cho một chỉ số đánh giá.

    Args:
        y_true: Nhãn thực tế.
        probabilities: Xác suất dự báo.
        metric_fn: Hàm tính toán chỉ số (nhận y_true, probabilities và trả về float).
        n_bootstrap (int): Số lượng mẫu bootstrap ngẫu nhiên (Mặc định: 1000).
        ci (float): Mức độ tin cậy (Mặc định: 0.95 tức 95%).
        random_state (int): Hạt giống ngẫu nhiên tái lập kết quả.

    Returns:
        tuple[float, float, float]: (chỉ_số_gốc, giới_hạn_dưới_ci, giới_hạn_trên_ci)
    """
    y_true_arr = np.asarray(y_true)
    prob_arr = np.asarray(probabilities)
    n_samples = len(y_true_arr)

    original_val = float(metric_fn(y_true_arr, prob_arr))
    if n_samples == 0:
        return original_val, original_val, original_val

    rng = np.random.default_rng(random_state)
    boot_scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_true_arr[idx]
        p_boot = prob_arr[idx]

        # Đảm bảo mẫu bootstrap có cả 2 lớp 0 và 1
        if len(np.unique(y_boot)) < 2:
            continue
        score = metric_fn(y_boot, p_boot)
        boot_scores.append(score)

    if not boot_scores:
        return original_val, original_val, original_val

    alpha = (1.0 - ci) / 2.0
    lower_pct = alpha * 100
    upper_pct = (1.0 - alpha) * 100

    lower = float(np.percentile(boot_scores, lower_pct))
    upper = float(np.percentile(boot_scores, upper_pct))
    return original_val, lower, upper


def compute_calibration_diagnostics(
    y_true,
    probabilities,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Tính toán các chỉ số chẩn đoán hiệu chỉnh xác suất (Calibration Diagnostics):
    - Expected Calibration Error (ECE)
    - Brier Skill Score (BSS) so với baseline tỷ lệ nợ xấu tự nhiên
    - Calibration Intercept (Độ lệch trung bình logit)
    - Calibration Slope (Độ dốc logit)

    Args:
        y_true: Nhãn thực tế (0 hoặc 1).
        probabilities: Xác suất rủi ro dự báo.
        n_bins (int): Số lượng khoảng chia xác suất.

    Returns:
        dict[str, float]: Từ điển chứa ECE, BSS, brier_score, intercept, slope.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    prob_arr = np.asarray(probabilities, dtype=float)

    # 1. Brier Score & Brier Skill Score (BSS)
    brier = float(brier_score_loss(y_true_arr, prob_arr))
    p_baseline = float(np.mean(y_true_arr))
    brier_baseline = float(np.mean((y_true_arr - p_baseline) ** 2))
    bss = float(1.0 - (brier / brier_baseline)) if brier_baseline > 0 else 0.0

    # 2. Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true_arr)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (prob_arr >= bin_lower) & (prob_arr <= bin_upper)
        else:
            in_bin = (prob_arr >= bin_lower) & (prob_arr < bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            actual_in_bin = np.mean(y_true_arr[in_bin])
            pred_in_bin = np.mean(prob_arr[in_bin])
            ece += np.abs(actual_in_bin - pred_in_bin) * (np.sum(in_bin) / n_total)

    # 3. Calibration Intercept & Slope via Logistic Regression on Logits
    # Kẹp xác suất tránh log(0)
    eps = 1e-15
    p_clipped = np.clip(prob_arr, eps, 1 - eps)
    logits = np.log(p_clipped / (1 - p_clipped))

    # Hồi quy Logistic tuyến tính giữa logit và y_true
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(solver="lbfgs", C=1e5)
        lr.fit(logits.reshape(-1, 1), y_true_arr)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception:
        slope = 1.0
        intercept = 0.0

    return {
        "brier_score": brier,
        "brier_skill_score": bss,
        "expected_calibration_error": float(ece),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }

