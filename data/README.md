# Danh mục dữ liệu

## Dữ liệu dùng cho mô hình chính

`raw/lendingclub_2007_2011.csv` là bản LendingClub đầy đủ gồm 39.717 dòng và
111 cột. Dữ liệu có `issue_d` từ tháng 06/2007 đến 12/2011, nhờ đó dự án có thể
đánh giá out-of-time thay vì dùng ngày 2021 đã bị biến đổi.

| Trạng thái | Số dòng | Cách xử lý |
|---|---:|---|
| `Fully Paid` | 32.950 | Nhãn 0 |
| `Charged Off` | 5.627 | Nhãn 1 |
| `Current` | 1.140 | Loại vì chưa có kết quả cuối cùng |

Pipeline chỉ dùng các trường có tại thời điểm cấp vay. Các trường hậu nghiệm như
`total_pymnt`, `recoveries`, `last_pymnt_d`, `out_prncp` và
`collection_recovery_fee` không nằm trong danh sách đặc trưng.

Split theo thời gian:

- Train: trước 01/01/2011 — 18.061 khoản vay đã kết thúc.
- Validation: 01/01/2011–30/06/2011 — 9.015 khoản vay.
- Test: từ 01/07/2011 — 11.501 khoản vay.

## Các tệp bổ sung

| Tệp | Kích thước | Vai trò | Có gộp vào model chính? |
|---|---:|---|---|
| `external/credit_train.csv` | 100.514 × 19 | Bộ phân loại tín dụng khác; có nhãn nhưng 18.514 Loan ID trùng | Không |
| `external/credit_test.csv` | 10.353 × 18 | Test không nhãn của cùng bộ trên | Không |
| `external/bank_personal_loan.csv` | 5.000 × 14 | Dự đoán chấp nhận personal loan, không phải default | Không |
| `Bank Loan Dataset.csv` | 38.576 × 24 | Bản LendingClub rút gọn, ngày đã chuyển sang 2021 | Chỉ giữ để đối chiếu legacy |

Không gộp các tệp chỉ vì cùng nói về “loan”. Chúng khác đơn vị quan sát, schema,
nguồn và định nghĩa nhãn; gộp sẽ tạo target không nhất quán và metric khó diễn giải.

## Nguồn và giấy phép

Các tệp do người dùng cung cấp không đi kèm URL tải xuống, data card, checksum
nguồn hoặc giấy phép. Dấu vết cột và ID cho thấy dữ liệu chính bắt nguồn từ
LendingClub lịch sử, nhưng chưa đủ bằng chứng để xác nhận quyền phân phối lại.

> Dataset được sử dụng cho mục đích học tập; nguồn gốc và quy trình thu thập ban đầu chưa được xác minh đầy đủ.

Toàn bộ CSV được loại khỏi Git bằng `.gitignore`. Trước khi công khai repository,
cần bổ sung URL phiên bản chính xác, tác giả/tổ chức, ngày truy cập, giấy phép và
SHA-256; nếu không, chỉ cung cấp hướng dẫn để người dùng tự đặt dữ liệu vào máy.

Checksum SHA-256 của các tệp đang dùng:

| Tệp | SHA-256 |
|---|---|
| `lendingclub_2007_2011.csv` | `a57286c2a5f329930c875366790c8f5291be7525b7b4e2355dcbfb2e73af6f04` |
| `credit_train.csv` | `40b2f45ba4bacebb641bdcb0db290cd66c54fbeb334514ee9ba93f6825494894` |
| `credit_test.csv` | `ca992a7fae41f734973eab26482fe47d79d29752a821aa070cf38aaa3ee7b4b5` |
| `bank_personal_loan.csv` | `a095bd860adb83bb426ce3c54cdc41bdd7a7cc43be20d7ecaf93d87a2d2f57ee` |
| `Bank Loan Dataset.csv` | `29f86ecd2a11e9eac5a97f28692d2437a6aa79b9f07fbffbd94beac2f5fdf578` |

## Định nghĩa nhãn

`loan_status` là trạng thái có sẵn trong dữ liệu. Dự án chỉ ánh xạ:

```text
Charged Off -> default_flag = 1
Fully Paid  -> default_flag = 0
Current     -> loại khỏi mô hình
```

Cách ánh xạ này không thay thế định nghĩa nghiệp vụ ban đầu của nhà cung cấp.
