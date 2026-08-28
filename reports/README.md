# Báo cáo có thể tái tạo

Các tệp trong thư mục này được sinh bởi `python -m src.train`:

- `model_comparison.csv`: PR-AUC, ROC-AUC và balanced accuracy từ 5-fold CV trên train.
- `test_metrics.json`: đánh giá một lần trên test độc lập.
- `slice_*.csv`: metric theo grade, sở hữu nhà và bang cho nhóm đủ cỡ mẫu.

Các slice chỉ hỗ trợ phát hiện chênh lệch hiệu năng, không chứng minh fairness vì
dữ liệu thiếu thuộc tính nhạy cảm và nguồn gốc chưa được xác minh.
