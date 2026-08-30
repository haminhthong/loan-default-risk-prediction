# Model card

## Intended use

Minh họa pipeline dự báo `Charged Off` tại thời điểm cấp khoản vay cho mục đích
học tập và portfolio. Không dùng để tự động phê duyệt, từ chối hoặc định giá tín dụng.

## Model và đánh giá

Champion là Logistic Regression sau sigmoid calibration. Model được chọn theo
PR-AUC trung bình của 5-fold CV trên dữ liệu trước năm 2011. Threshold được chọn
trên nửa đầu năm 2011 với giả định chi phí false negative:false positive là 5:1;
nửa cuối năm 2011 là test out-of-time và chỉ dùng để báo cáo cuối.

Artifact triển khai không dùng `int_rate` và `sub_grade`. Phiên bản này đạt
PR-AUC 0,3384, nhỉnh hơn phiên bản đầy đủ 0,3366, đồng thời giảm nguy cơ học lại
hệ thống định giá sẵn có. Quyết định vẫn cần xác nhận trên dữ liệu ngoài mẫu.

## Hạn chế

- Nguồn, giấy phép và phép biến đổi ngày của CSV chưa được xác minh đầy đủ.
- Out-of-time test chỉ bao phủ một giai đoạn lịch sử và một nguồn dữ liệu.
- Cost ratio 5:1 chỉ là sensitivity scenario, không phải expected loss thực tế.
- Slice theo địa lý/grade không thay thế kiểm định fairness với thuộc tính nhạy cảm.
- Xác suất và threshold không được dùng ngoài phân phối dữ liệu này nếu chưa tái kiểm định.
- PSI cho thấy drift đáng kể ở một số trường; cần giám sát và đặt ngưỡng tái huấn luyện.
- Odds ratio chỉ mô tả liên hệ trong mô hình, không được diễn giải là tác động nhân quả.
