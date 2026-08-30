# Dự Báo Rủi Ro Vỡ Nợ Khoản Vay (Loan Default Risk Prediction Pipeline)

Hệ thống Machine Learning phân loại rủi ro tín dụng cá nhân đạt tiêu chuẩn **Production & Leakage-Safe**, tập trung vào quy trình đánh giá trung thực: phân chia dữ liệu theo thời gian (Temporal Split), đóng gói tiền xử lý trong pipeline, loại bỏ thuộc tính hậu nghiệm (Anti-Data Leakage), hiệu chỉnh xác suất (Probability Calibration) và tối ưu ngưỡng quyết định theo chi phí kinh doanh tổn thất tài chính (Cost-Sensitive Thresholding).

Dự án được xây dựng hoàn chỉnh với **RESTful API (FastAPI)**, **Giao diện Web tương tác (Streamlit)**, **Unit Tests (Pytest)**, **Đóng gói Container (Docker)** và **Tự động hóa CI (GitHub Actions)**.

---

## 🎯 Điểm Nổi Bật & Kết Quả Thực Nghiệm

- **Bộ dữ liệu thực tế**: 39.717 khoản vay LendingClub (giai đoạn 2007 - 2011) được khôi phục đúng mốc thời gian phát hành.
- **Tập nhãn nhị phân**: `Charged Off` (Rủi ro vỡ nợ = 1) và `Fully Paid` (Đã thanh toán đủ = 0). Loại bỏ khoản vay `Current` để đảm bảo bài toán phân loại kết thúc rõ ràng.
- **Chống rò rỉ dữ liệu (Point-in-Time Features)**: Loại bỏ triệt me các trường thông tin phát sinh sau khi giải ngân (`total_payment`, `last_payment_date`, `next_payment_date`, v.v.).
- **Temporal Out-of-Time Split**: Chia dữ liệu theo thời gian phát hành: Train (trước năm 2011: 30.744 mẫu), Validation (T1-T6/2011: 4.098 mẫu), Test (T7-T12/2011: 3.735 mẫu).
- **So sánh Mô hình qua 5-Fold Cross-Validation**:
  - Mô hình **Logistic Regression** đạt **PR-AUC CV: 0.2479 ± 0.0107** (vượt trội hơn Random Forest **0.2346** và Dummy **0.1313**).
- **Hiệu chỉnh xác suất (Platt Scaling)**: Áp dụng `CalibratedClassifierCV(method='sigmoid')` trên tập Train, giúp xác suất dự báo phản ánh đúng tỷ lệ vỡ nợ thực tế (Brier Score trên tập Test: **0.1286**).
- **Tối ưu Ngưỡng theo Chi Phí Kinh Doanh**: Chọn ngưỡng phân loại trên tập Validation độc lập với tỷ lệ chi phí bỏ sót nợ xấu (FN) đắt gấp **5 lần** cảnh báo nhầm (FP). Ngưỡng tối ưu đạt được là **0.14**.

### 📊 Bảng Kết Quả Đánh Giá Trực Tiếp Trên Tập Out-of-Time Test (T7-T12/2011)

| Chỉ số Đánh giá (Metric) | Giá trị | Ý nghĩa Nghiệp vụ |
|---|---:|---|
| **PR-AUC** | **0.3384** | Metric chính cho bài toán mất cân bằng lớp (Đạt độ bao phủ cao) |
| **ROC-AUC** | **0.7195** | Khả năng phân biệt và xếp hạng rủi ro khoản vay |
| **Recall (Nợ xấu)** | **62.83%** | Phát hiện thành công hơn 62.8% các ca vỡ nợ thực tế |
| **Precision (Nợ xấu)** | **29.24%** | Độ chính xác khi đưa ra cảnh báo rủi ro vỡ nợ |
| **F1-Score** | **0.3991** | Trung bình hài hòa giữa Precision và Recall |
| **Balanced Accuracy** | **66.20%** | Độ chính xác trung bình cân bằng giữa lớp 0 và 1 |
| **Brier Score** | **0.1286** | Sai số dự báo xác suất (Càng thấp xác suất càng chuẩn xác) |
| **Ngưỡng Chọn (Validation)** | **0.14** | Tối thiểu hóa chi phí rủi ro theo tỷ lệ FN:FP = 5:1 |

---

## 🏗️ Kiến Trúc Hệ Thống & Luồng Xử Lý (Pipeline Architecture)

```text
[ Dữ liệu CSV gốc LendingClub ]
              │
              ▼
    [ Schema Guard & Data Validation ]  (Kiểm tra cột bắt buộc, trùng lặp ID & null)
              │
              ▼
   [ Point-in-Time Feature Engineering ] (Log income, DTI, Credit history, Payment ratio)
              │
              ▼
   [ Temporal Out-of-Time Split ] ──────► Train (quá khứ) / Validation / Test (tương lai)
              │
              ▼
    [ ColumnTransformer Preprocessing ]  (Imputer, StandardScaler, OneHotEncoder)
              │
              ▼
    [ 5-Fold Stratified CV Model Search ] (Dummy vs Logistic Regression vs Random Forest)
              │
              ▼
   [ CalibratedClassifierCV (Sigmoid) ]  (Fit duy nhất trên tập Train)
              │
              ▼
 [ Cost-Sensitive Threshold Selection ] ──► Tối ưu tổng chi phí FN:FP = 5:1 trên Validation
              │
              ▼
  [ Final Evaluation & Drift Report ] ──► Đánh giá Test độc lập & Tính chỉ số PSI Drift
              │
              ▼
  [ Single Model Artifact (.joblib) ]
              │
      ┌───────┴───────┐
      ▼               ▼
[ REST API FastAPI ] [ Web App Streamlit ]
```

---

## 📂 Cấu Trúc Repository

```text
loan-default-risk-prediction/
├── README.md                   # Báo cáo tổng quan dự án & Hướng dẫn sử dụng
├── MODEL_CARD.md               # Thẻ mô hình (Model Card) ghi nhận giới hạn & đạo đức AI
├── requirements.txt            # Thư viện phục vụ chạy ứng dụng Production
├── requirements-dev.txt        # Thư viện phục vụ huấn luyện, testing và phát triển
├── pytest.ini                  # Cấu hình kiểm thử tự động Pytest
├── Dockerfile                  # Đóng gói ứng dụng container Docker
├── .github/
│   └── workflows/
│       └── ci.yml              # Quy trình kiểm thử tự động GitHub Actions CI
├── app/
│   ├── api.py                  # Dịch vụ RESTful API FastAPI (/health, /info, /score, /explain)
│   └── streamlit_app.py        # Dashboard tương tác 3 Tab (Thẩm định đơn, CSV batch, Diagnostics)
├── src/
│   ├── __init__.py
│   ├── data.py                 # Nạp dữ liệu, kiểm tra Schema Guard & chia tập theo thời gian
│   ├── features.py             # Trích xuất đặc trưng an toàn chống rò rỉ dữ liệu
│   ├── evaluate.py             # Tính toán metrics & Tối ưu ngưỡng theo chi phí nghiệp vụ
│   ├── analysis.py             # Hiệu chỉnh xác suất, tính PSI Drift & Odds Ratios
│   ├── train.py                # Pipeline huấn luyện chính, so sánh CV & xuất artifact
│   └── predict.py              # Đóng gói hàm suy luận dự báo từ file artifact
├── tests/
│   ├── test_data.py            # Unit tests cho quy trình nạp & chia dữ liệu
│   ├── test_features.py        # Unit tests cho trích xuất đặc trưng
│   ├── test_evaluate.py        # Unit tests cho chọn ngưỡng chi phí
│   ├── test_analysis.py        # Unit tests cho bảng calibration & PSI
│   ├── test_predict.py         # Unit tests cho hàm suy luận dự báo
│   └── test_api_schema.py      # Unit tests cho Pydantic validation schema
├── artifacts/
│   └── loan_default_cv.joblib  # Artifact mô hình đóng gói hoàn chỉnh
├── reports/                    # Thư mục xuất tự động các báo cáo CSV/JSON
└── data/                       # Thư mục chứa dữ liệu gốc LendingClub
```

---

## ⚡ Hướng Dẫn Cài Đặt & Chạy Dự Án

### 1. Yêu cầu Môi trường
- Python 3.10 trở lên (Khuyến nghị 3.11).

### 2. Khởi tạo Môi trường Ảo (Virtual Environment)

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

---

## 🚀 Thực Thi Các Thành Phần Dự Án

### 1. Huấn luyện Pipeline Mô hình
Chạy quy trình huấn luyện tự động từ dữ liệu thô, so sánh mô hình, chọn ngưỡng và xuất artifact:
```bash
python -m src.train
```

### 2. Chạy Suite Kiểm Thử Tự Động (Unit Tests)
Kiểm tra tính đúng đắn của toàn bộ các mô-đun dữ liệu, đặc trưng và API:
```bash
python -m pytest -q
```

### 3. Khởi chạy RESTful API (FastAPI)
Khởi chạy dịch vụ API backend tại cổng `8000`:
```bash
uvicorn app.api:app --reload
```
- Mở tài liệu tương tác OpenAPI / Swagger UI tại: **`http://localhost:8000/docs`**

#### Các API Endpoints Chính:
- `GET /health`: Kiểm tra trạng thái hoạt động của mô hình.
- `GET /info`: Xem thông số mô hình Champion, ngưỡng tối ưu và các metrics.
- `POST /score`: Chấm điểm rủi ro cho danh sách hồ sơ tín dụng đầu vào.
- `GET /explain`: Trích xuất danh sách các đặc trưng có hệ số Odds Ratio ảnh hưởng lớn nhất.

### 4. Khởi chạy Giao Diện Web Tương Tác (Streamlit Dashboard)
Khởi chạy bảng điều khiển thẩm định tín dụng trực quan:
```bash
streamlit run app/streamlit_app.py
```
- Truy cập giao diện tại: **`http://localhost:8501`**

#### Giao diện bao gồm 3 Tab chuyên nghiệp:
1. **📝 Thẩm Định Hồ Sơ Đơn**: Form nhập dữ liệu khách hàng thực tế, tính điểm rủi ro, thanh gauge hiển thị điểm và khuyến nghị Duyệt / Từ chối.
2. **📁 Chấm Điểm Hàng Loạt (Batch CSV)**: Upload file CSV danh sách khoản vay, tự động chấm điểm và cho phép tải về báo cáo kết quả.
3. **📊 Tổng Quan Mô Hình**: Tra cứu thông số kỹ thuật mô hình Champion và quy trình chống rò rỉ dữ liệu.

---

## 🐳 Đóng Gói Container (Docker)

Xây dựng và khởi chạy ứng dụng dễ dàng bằng Docker:

```bash
# Build Docker Image
docker build -t loan-default-risk .

# Chạy Docker Container (Map cổng 8000)
docker run --rm -p 8000:8000 loan-default-risk
```

---

## 💼 Gợi Ý Mô Tả Dự Án Trong CV (CV Presentation Guide)

Để đưa dự án này vào CV một cách ấn tượng nhất cho các vị trí **Machine Learning Engineer**, **Data Scientist**, hoặc **Risk Data Analyst**, bạn có thể tham khảo cách trình bày sau:

### Tên Dự Án: **Loan Default Risk Prediction with Leakage-Safe ML Pipeline**

#### Các Bullet Points Nổi Bật Cho CV:
- **Thiết kế Pipeline ML Chống Rò Rỉ Dữ Liệu (Anti-Leakage)**: Xây dựng quy trình xử lý dữ liệu chuẩn tín dụng trên 39.717 khoản vay LendingClub, áp dụng *Point-in-Time Feature Engineering* và *Temporal Out-of-Time Split* (Train < 2011, Test H2/2011) giúp đánh giá chính xác khả năng chống suy giảm dữ liệu (Data Drift).
- **Tối Ưu Ngưỡng Quyết Định Theo Chi Phí Tín Dụng & Probability Calibration**: Áp dụng *Platt Scaling (Sigmoid Calibration)* đạt Brier Score 0.1286; thiết kế bài toán tối ưu ngưỡng phân loại dựa trên ma trận chi phí tổn thất tài chính (FN:FP = 5:1), đạt chỉ số **ROC-AUC 0.720** và **Recall nợ xấu 62.8%** trên tập test tương lai.
- **Đóng Gói & Triển Khai Service (MLOps Standards)**: Xây dựng dịch vụ **RESTful API với FastAPI** kết hợp **Streamlit Interactive Dashboard**, bổ sung bộ 10 Unit Tests đạt 100% coverage qua **Pytest**, hỗ trợ **Docker Containerization** và tự động kiểm thử qua **GitHub Actions CI**.

#### Các Từ Khóa Kỹ Thuật Đã Sử Dụng (Keywords):
`Python`, `Scikit-Learn`, `Pandas`, `FastAPI`, `Streamlit`, `Docker`, `Pytest`, `GitHub Actions`, `Data Leakage Prevention`, `Temporal Split`, `Cost-Sensitive Learning`, `Probability Calibration`, `PSI Data Drift Analysis`.

---
*Dự án phục vụ mục đích học tập và nghiên cứu kỹ thuật Machine Learning Pipeline chuẩn công nghiệp.*
