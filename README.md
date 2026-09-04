# Dự Báo Rủi Ro Vỡ Nợ Khoản Vay (Loan Default Risk Prediction Pipeline)

Hệ thống Machine Learning phân loại rủi ro tín dụng cá nhân được thiết kế theo hướng **Production-oriented & Leakage-Safe**, tập trung vào quy trình kiểm thử trung thực: phân chia dữ liệu theo mốc thời gian phát hành (Temporal Split), bảo vệ kép chống rò rỉ dữ liệu (Point-in-Time Features & Denylist Leakage Guard), hiệu chỉnh xác suất (Probability Calibration), phân tích độ nhạy chi phí kinh doanh (Cost-Sensitive Thresholding) và đánh giá độ ổn định phân khúc (Segment Performance Stability).

Dự án được triển khai với **RESTful API (FastAPI)**, **Giao diện Web tương tác (Streamlit)**, **Suite kiểm thử tự động (Pytest với ~30 unit tests)**, **Kiểm thử tải (Locust 100 users)**, **Đóng gói Container (Docker)** và **Tự động hóa CI (GitHub Actions)**.

---

## 📌 1. Tầng Problem (Định Nghĩa Bài Toán & Data Contract)

### 1.1. Định Nghĩa Bài Toán Tín Dụng

- **Mục tiêu**: Ước lượng xác suất một khoản vay sẽ rơi vào trạng thái `Charged Off` (không có khả năng thu hồi) dựa trên thông tin sẵn có tại thời điểm khoản vay được phát hành.
- **Đơn vị dự báo**: Một khoản vay cá nhân (`id`).
- **Thời điểm dự báo**: Ngay trước hoặc tại thời điểm phát hành/giải ngân khoản vay (Point-in-Time).
- **Nhãn mục tiêu (`default_flag`)**:
  - `Charged Off = 1` (Dương tính - Nợ xấu / Rủi ro cao).
  - `Fully Paid = 0` (Âm tính - Đã hoàn tất thanh toán).
  - `Current` (Khoản vay đang chạy chưa hoàn tất) -> **Loại khỏi tập huấn luyện** để tránh bài toán nhãn chưa chín (Target Censoring).
- **Đầu ra hệ thống**:
  - Xác suất rủi ro `Charged Off` (từ `0.0` đến `1.0`).
  - Cờ cảnh báo rủi ro (`default_prediction = 1 hoặc 0`) theo ngưỡng đã được chọn trên tập Validation.
- **Không thuộc phạm vi bài toán (Out of Scope)**:
  - Tự động phê duyệt hoặc từ chối khoản vay.
  - Định giá lãi suất (Interest Pricing).
  - Dự báo tổn thất tài chính bằng tiền cụ thể.
  - Thay thế chuyên viên hoặc hội đồng thẩm định tín dụng.

> [!NOTE]
> Cột nhãn `default_flag` trong dự án là một nhãn phân loại kỹ thuật Machine Learning dựa trên dữ liệu lịch sử LendingClub, **không phải là định nghĩa pháp lý về vỡ nợ** theo quy định ngân hàng thương mại.

---

### 1.2. Hợp Đồng Dữ Liệu (Data Contract)

Mọi yêu cầu dự báo đầu vào đều tuân thủ nghiêm ngặt hợp đồng dữ liệu sau (được kiểm soát bởi Pydantic Schema với `ConfigDict(extra="forbid")`):

| Cột | Kiểu Dữ Liệu | Đơn Vị | Thời Điểm Có | Miền Giá Trị Hợp Lệ |
|---|---|---|---|---|
| `loan_amnt` | `float` | USD | Trước giải ngân | $> 0$ |
| `term` | `Literal` | Tháng | Khi đăng ký | `"36 months"`, `"60 months"` |
| `installment` | `float` | USD/tháng | Khi đăng ký | $> 0$ |
| `grade` | `Literal` | Xếp hạng | Khi đăng ký | `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`, `"G"` |
| `emp_length` | `str \| None` | Thâm niên | Khi đăng ký | Chuỗi thâm niên (ví dụ: `"5 years"`) |
| `home_ownership` | `Literal` | Sở hữu nhà | Khi đăng ký | `"RENT"`, `"OWN"`, `"MORTGAGE"`, `"OTHER"` |
| `annual_inc` | `float` | USD/năm | Khi đăng ký | $> 0$ |
| `verification_status` | `Literal` | Trạng thái | Khi đăng ký | `"Verified"`, `"Source Verified"`, `"Not Verified"` |
| `purpose` | `str` | Mục đích | Khi đăng ký | Chuỗi mục đích vay hợp lệ |
| `addr_state` | `str` | Bang | Khi đăng ký | Mã 2 chữ cái in hoa (ví dụ: `"CA"`, `"NY"`) |
| `dti` | `float \| None` | % | Khi đăng ký | $0.0 \le \text{DTI} \le 100.0$ |
| `delinq_2yrs` | `float \| None` | Số lần | Lịch sử 2 năm | $\ge 0$ |
| `inq_last_6mths` | `float \| None` | Số lần | Lịch sử 6 tháng | $\ge 0$ |
| `open_acc` | `float \| None` | Tài khoản | Khi đăng ký | $\ge 0$ |
| `pub_rec` | `float \| None` | Hồ sơ | Lịch sử | $\ge 0$ |
| `revol_bal` | `float \| None` | USD | Khi đăng ký | $\ge 0$ |
| `revol_util` | `str \| None` | % | Khi đăng ký | Chuỗi phần trăm (ví dụ: `"45.20%"`) |
| `total_acc` | `float \| None` | Tài khoản | Lịch sử | $\ge 0$ |
| `earliest_cr_line` | `str \| None` | Tháng-Năm | Lịch sử | Định dạng `MMM-YY` (ví dụ: `"Jan-00"`) |
| `issue_d` | `str` | Tháng-Năm | Khi cấp vay | Định dạng `MMM-YY` (ví dụ: `"Dec-11"`) |

---

## 🎯 2. Tầng AI/ML Correctness & Đánh Giá Thực Nghiệm

### 2.1. Bảo Vệ Lớp Đôi Chống Rò Rỉ Dữ Liệu (Dual-Layer Anti-Leakage Guard)

Hệ thống kết hợp cả phương pháp **Allowlist** (chỉ giữ thuộc tính trước giải ngân) và **Denylist** (`LEAKAGE_COLUMNS`):
Tất cả các trường hậu nghiệm phát sinh sau giải ngân (`total_pymnt`, `last_pymnt_amnt`, `recoveries`, `collection_recovery_fee`, `out_prncp`, v.v.) sẽ bị kiểm tra và kích hoạt `ValueError` lập tức nếu xuất hiện trong ma trận đặc trưng.

### 2.2. Temporal Cross-Validation (Expanding-Window)

Thay vì chia K-Fold ngẫu nhiên gây rò rỉ dữ liệu tương lai vào quá khứ, quá trình huấn luyện áp dụng **Expanding-Window Temporal CV** theo tháng phát hành:
- **Fold 1**: Train 2007–2008 $\rightarrow$ Validate 2009
- **Fold 2**: Train 2007–2009 $\rightarrow$ Validate H1/2010
- **Fold 3**: Train 2007–H1/2010 $\rightarrow$ Validate H2/2010

### 2.3. Thử Nghiệm Ablation Study Biến Định Giá & Chính Sách Cũ

Để kiểm tra xem mô hình có thực sự học được tín hiệu rủi ro nội tại hay chỉ học lại xếp hạng tín dụng cũ (`grade`, `int_rate`), thực nghiệm Ablation được tiến hành trên 4 cấp độ:

| Cấp Độ Ablation | Số Đặc Trưng | PR-AUC (Test) | ROC-AUC (Test) | Recall (Test) | Precision (Test) |
|---|---:|---:|---:|---:|---:|
| `all` (Tất cả đặc trưng) | 20 | 0.3384 | 0.7195 | 62.83% | 29.24% |
| `no_int_sub` (Không int_rate, sub_grade) | 18 | 0.3366 | 0.7180 | 62.50% | 29.10% |
| `no_int_sub_grade` (Không int_rate, sub_grade, grade) | 17 | 0.3290 | 0.7110 | 61.80% | 28.50% |
| `no_pricing_all` (Không int_rate, sub_grade, grade, installment) | 15 | 0.3150 | 0.6980 | 59.40% | 27.20% |

### 2.4. Khái Niệm Expected Loss & Ngưỡng Chi Phí 5:1

Chi phí tổn thất nghiệp vụ tuân theo mô hình **Expected Loss (Tổn thất kỳ vọng)**:
$$\text{Expected Loss} = \text{Probability of Default (PD)} \times \text{Exposure at Default (EAD)} \times \text{Loss Given Default (LGD)}$$

Trong thực nghiệm, việc đặt tỷ lệ chi phí FN:FP = 5:1 (chi phí bỏ sót khoản nợ xấu gấp 5 lần chi phí báo động nhầm khách hàng tốt) dẫn đến ngưỡng phân loại **0.14**.
> [!IMPORTANT]
> Ngưỡng **0.14** được định vị là **"ngưỡng minh họa theo giả định chi phí 5:1"**, không phải là "ngưỡng kinh doanh tối ưu tuyệt đối". Trong vận hành thực tế, ngưỡng này phải được điều chỉnh linh hoạt theo năng lực thẩm định thủ công của chuyên viên tín dụng và chi phí vốn ròng.

### 2.5. Kết Quả Thực Nghiệm & Khoảng Tin Cậy 95% Bootstrap CIs

| Metric Đánh Giá | Giá Trị Point Estimate | Khoảng Tin Cậy 95% Bootstrap CI | Ý Nghĩa Nghiệp Vụ |
|---|---:|---|---|
| **PR-AUC** | **0.3384** | `[0.3120 – 0.3650]` | Metric chính cho dữ liệu mất cân bằng lớp |
| **ROC-AUC** | **0.7195** | `[0.6980 – 0.7410]` | Khả năng xếp hạng rủi ro out-of-time |
| **Recall (Nợ xấu)** | **62.83%** | `[59.10% – 66.50%]` | Tỷ lệ phát hiện thành công khoản vay charged off |
| **Precision** | **29.24%** | `[27.10% – 31.40%]` | Độ chính xác của các cảnh báo phát ra |
| **Brier Score** | **0.1286** | `[0.1210 – 0.1360]` | Độ chính xác hiệu chỉnh xác suất |

### 2.6. Độ Ôn Định Theo Phân Khúc (Segment Performance Stability)

Dự án thực hiện phân tích hiệu năng theo các lát cắt dữ liệu (`grade`, `home_ownership`, `addr_state`) và định danh chuẩn là **Performance Stability by Segment**.
> [!NOTE]
> Dự án chưa thực hiện fairness audit do không có hoặc không sử dụng thuộc tính nhạy cảm đã được quản trị phù hợp.

---

## 🛠️ 3. Tầng Software Engineering & Quy Trình Tái Lập (Reproducibility)

### 3.1. Hướng Dẫn Clone – Prepare – Train – Test

Quy trình tái lập kết quả thực nghiệm hoàn chỉnh từ repository:

```bash
# 1. Clone repository
git clone https://github.com/haminhthong/loan-default-risk-prediction.git
cd loan-default-risk-prediction

# 2. Tạo và kích hoạt môi trường ảo Python
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Cài đặt thư viện phát triển
python -m pip install -r requirements-dev.txt

# 4. Người dùng tải dữ liệu thô LendingClub 2007-2011 và đặt vào data/raw/
# tệp: data/raw/lendingclub_2007_2011.csv

# 5. Kiểm tra tính toàn vẹn và hash SHA-256 của tệp dữ liệu
python scripts/verify_data.py

# 6. Chạy quy trình huấn luyện pipeline Machine Learning
python -m src.train

# 7. Thực thi suite kiểm thử tự động (Pytest)
python -m pytest -q
```

---

## 🚀 4. RESTful API, Streamlit Dashboard & Load Test

### 4.1. Khởi Chạy API Backend (FastAPI)

```bash
uvicorn app.api:app --reload --port 8000
```
- Tài liệu swagger OpenAPI tương tác: `http://localhost:8000/docs`
- Security note: Endpoint `/health` trả về kết quả trạng thái an toàn `{"status": "ok", "model_loaded": true}` và **không làm lộ đường dẫn file hệ thống**.

### 4.2. Khởi Chạy Web App (Streamlit)

```bash
streamlit run app/streamlit_app.py
```
- Giao diện tích hợp kiểm soát kích thước file upload (tối đa 10 MB và 10.000 hồ sơ) cùng cơ chế xử lý lỗi an toàn không rò rỉ stack trace.

### 4.3. Benchmark Kiểm Thử Tải (Locust Load Test)

```bash
locust -f load_tests/locustfile.py --headless -u 100 -r 10 --run-time 2m --host http://localhost:8000
```
**Acceptance Criteria Target**:
- Concurrency: 100 users đồng thời
- Latency p95: $< 500 \text{ ms}$
- Error rate: $< 1.0\%$
- Throughput: $\ge 50 \text{ req/sec}$

---

## 🔗 Tài Liệu & Báo Cáo Liên Quan

- 📜 **Báo cáo Phân tích Chi tiết (DOCX Report)**: [`reports/Loan_Default_Risk_Project_Report.docx`](file:///d:/hoc/can%20lam/Loan%20Default%20Risk%20Prediction%20with%20a%20Leakage-Safe%20Machine%20Learning%20Pipeline/loan-default-risk-prediction/reports/Loan_Default_Risk_Project_Report.docx)
- 📓 **Google Colab Notebook (Tái lập metric)**: [`notebooks/02_google_colab_reproducible.ipynb`](file:///d:/hoc/can%20lam/Loan%20Default%20Risk%20Prediction%20with%20a%20Leakage-Safe%20Machine%20Learning%20Pipeline/loan-default-risk-prediction/notebooks/02_google_colab_reproducible.ipynb)
- 🔒 **Chính Sách Bảo Mật & Privacy**: [`SECURITY.md`](file:///d:/hoc/can%20lam/Loan%20Default%20Risk%20Prediction%20with%20a%20Leakage-Safe%20Machine%20Learning%20Pipeline/loan-default-risk-prediction/SECURITY.md)

---
*Dự án phục vụ mục đích học tập, nghiên cứu và thiết kế Machine Learning Pipeline chuẩn nghiệp vụ tín dụng.*
