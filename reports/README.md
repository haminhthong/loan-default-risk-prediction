# Báo cáo có thể tái tạo

Các tệp trong thư mục này được sinh bởi `python -m src.train`:

- `model_comparison.csv`: PR-AUC, ROC-AUC và balanced accuracy từ 5-fold CV trên train.
- `test_metrics.json`: đánh giá một lần trên test độc lập.
- `slice_*.csv`: metric theo grade, sở hữu nhà và `addr_state` cho nhóm đủ cỡ mẫu.
- `feature_set_comparison.csv`: so sánh có/không có `int_rate`, `sub_grade`.
- `threshold_sensitivity.csv`: ngưỡng dưới ba giả định chi phí bỏ sót.
- `calibration_test.csv`: xác suất dự báo và default rate theo bin.
- `drift_psi.csv`: PSI và tỷ lệ thiếu giữa train và test.
- `logistic_odds_ratios.csv`: hệ số và odds ratio phục vụ giải thích mô hình.

Các slice chỉ hỗ trợ phát hiện chênh lệch hiệu năng, không chứng minh fairness vì
dữ liệu thiếu thuộc tính nhạy cảm và nguồn gốc chưa được xác minh.
