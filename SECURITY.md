# Security & Privacy Policy (Chính Sách Bảo Mật & Quyền Riêng Tư)

## 1. Nguyên Tắc Bảo Mật Dữ Liệu Tín Dụng
- **Không Ghi Nhật Ký Dữ Liệu Nhạy Cảm (No PII Logging)**: Dịch vụ API và ứng dụng web không lưu vết thông tin nhận dạng cá nhân (PII), thu nhập, địa chỉ cư trú hoặc số dư tài khoản tín dụng vào tệp log hệ thống.
- **Xử Lý Trong Bộ Nhớ (In-Memory Processing)**: Dữ liệu hồ sơ gửi tới API `/score` hoặc upload qua giao diện CSV Streamlit chỉ được xử lý tạm thời trong bộ nhớ RAM phục vụ suy luận dự báo và lập tức bị giải phóng.
- **Giới Hạn Tải Tệp (File Limits)**: Tệp tải lên giao diện batch qua CSV bị giới hạn dung lượng **10 MB** và tối đa **10.000 dòng** cho mỗi yêu cầu để phòng chống tấn công từ chối dịch vụ (DoS/OOM).
- **Mã Hóa Dữ Liệu (Data Encryption)**: Mọi giao tiếp sản xuất bắt buộc sử dụng mã hóa HTTPS/TLS 1.3 qua Reverse Proxy (Nginx/Traefik).

## 2. Quản Lý Quyền Truy Cập & API Key
- API hỗ trợ xác thực bằng header `X-API-Key`.
- Khóa truy cập được cấu hình thông qua biến môi trường `LOAN_API_KEY`. Tuyệt đối không commit bí mật hoặc mã thông báo vào mã nguồn Git.

## 3. Báo Cáo Lỗ Hổng Bảo Mật (Vulnerability Reporting)
Nếu phát hiện bất kỳ lỗ hổng bảo mật nào trong hệ thống, vui lòng liên hệ trực tiếp với nhóm phát triển qua email hoặc tạo một private security advisory. Không công bố công khai lỗ hổng trước khi có bản vá chính thức.
