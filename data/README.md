# Dữ liệu

## Tệp sử dụng

Notebook và mã nguồn kỳ vọng tệp `Bank Loan Dataset.csv` nằm trong thư mục `data/`.
Tệp làm việc hiện có 38.576 dòng, 24 cột và 3 trạng thái khoản vay:

| Trạng thái | Số dòng | Vai trò |
|---|---:|---|
| `Fully Paid` | 32.145 | Nhãn 0: đã trả đủ |
| `Charged Off` | 5.333 | Nhãn 1: ghi nhận mất vốn/nợ xấu |
| `Current` | 1.098 | Loại khỏi huấn luyện vì kết quả cuối cùng chưa quan sát được |

Khoảng ngày `issue_date` trong tệp là 01/01/2021–12/12/2021. Tuy nhiên, đây không
phải thời gian phát hành gốc đáng tin cậy. Ví dụ, khoản vay `id=1077430` tương ứng
với một bản ghi LendingClub được công khai ở nơi khác với ngày phát hành tháng
12/2011. Điều này cho thấy bản dữ liệu hiện tại đã được xử lý và thay đổi ngày.

## Nguồn và giấy phép

- Trang có cấu trúc 24 cột gần như trùng khớp: [Financial Loan Dataset của Aryan Singh trên Kaggle](https://www.kaggle.com/datasets/datawitharyan/financial-loan-dataset).
- Nguồn dữ liệu thượng nguồn có khả năng là dữ liệu khoản vay LendingClub lịch sử.
- Không có data card, tệp giấy phép, lịch sử tải xuống hoặc checksum từ nguồn đi
  kèm bản CSV hiện tại. Vì vậy chưa thể chứng minh bản này được tải từ trang trên,
  ai đã thực hiện biến đổi, hoặc giấy phép nào áp dụng cho bản đã biến đổi.

> Dataset được sử dụng cho mục đích học tập; nguồn gốc và quy trình thu thập ban đầu chưa được xác minh đầy đủ.

Không phân phối lại CSV công khai trước khi xác minh quyền sử dụng. Khi tìm được
nguồn chính xác, hãy bổ sung tên tác giả/tổ chức, URL phiên bản, ngày truy cập,
giấy phép và checksum SHA-256.

## Ý nghĩa dữ liệu và cách tạo nhãn

Dữ liệu mô tả hồ sơ khoản vay, đặc điểm người vay, điều khoản cấp tín dụng và kết
quả thanh toán. Dự án dự đoán rủi ro tại thời điểm cấp vay, nên loại các trường chỉ
phát sinh sau giải ngân như `total_payment`, `last_payment_date`,
`next_payment_date` và `last_credit_pull_date`.

`loan_status` là cột trạng thái có sẵn trong CSV, không phải nhãn do dự án thu
thập. Nhãn mô hình `default_flag` được ánh xạ như sau:

```text
Charged Off -> 1
Fully Paid  -> 0
Current     -> loại khỏi tập mô hình
```

Cách ánh xạ này chỉ tạo bài toán nhị phân từ trạng thái cuối kỳ; nó không chứng
minh quy tắc nghiệp vụ ban đầu mà bên cung cấp dùng để tạo `loan_status`.

## Đặt dữ liệu

```text
data/
└── Bank Loan Dataset.csv
```

CSV không được đưa vào Git mặc định nếu điều khoản phân phối lại chưa rõ.

