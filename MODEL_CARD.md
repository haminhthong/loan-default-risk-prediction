# Model card

## Intended use

Minh họa pipeline dự báo `Charged Off` tại thời điểm cấp khoản vay cho mục đích
học tập và portfolio. Không dùng để tự động phê duyệt, từ chối hoặc định giá tín dụng.

## Model và đánh giá

Champion là Logistic Regression sau sigmoid calibration. Model được chọn theo
PR-AUC trung bình của 5-fold CV trên train. Threshold được chọn riêng trên
validation với giả định chi phí false negative:false positive là 5:1; test chỉ
được dùng để báo cáo cuối.

## Hạn chế

- Nguồn, giấy phép và phép biến đổi ngày của CSV chưa được xác minh đầy đủ.
- Không có out-of-time validation hợp lệ.
- Cost ratio 5:1 chỉ là sensitivity scenario, không phải expected loss thực tế.
- Slice theo địa lý/grade không thay thế kiểm định fairness với thuộc tính nhạy cảm.
- Xác suất và threshold không được dùng ngoài phân phối dữ liệu này nếu chưa tái kiểm định.
