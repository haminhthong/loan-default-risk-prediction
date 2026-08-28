# Dự báo rủi ro vỡ nợ khoản vay

Dự án xây dựng pipeline machine learning dự báo một khoản vay sẽ bị
`Charged Off` hay được thanh toán đầy đủ. Trọng tâm của dự án là quy trình đánh
giá trung thực: tách dữ liệu trước khi phân tích, đóng gói tiền xử lý trong
pipeline, loại biến hậu nghiệm và chỉ chọn ngưỡng trên tập validation.

> Đây là dự án học tập, không phải hệ thống chấm điểm tín dụng dùng trong thực tế.

## Kết quả chính

- Bài toán phân loại nhị phân mất cân bằng lớp.
- Nhãn dương: `Charged Off`; nhãn âm: `Fully Paid`.
- Khoản vay `Current` bị loại vì chưa có kết quả cuối cùng.
- Tiền xử lý được fit riêng trong từng tập train/fold bằng `ColumnTransformer`.
- So sánh Dummy, Logistic Regression và Random Forest bằng 5-fold CV trên train.
- Hiệu chỉnh xác suất bằng sigmoid calibration chỉ từ train.
- Đánh giá bằng PR-AUC, ROC-AUC, F1, recall, precision, balanced accuracy và Brier score.
- Ngưỡng quyết định được chọn theo chi phí: một ca bỏ sót nợ xấu mặc định đắt gấp
  5 lần một cảnh báo nhầm.
- Mô hình và ngưỡng được lưu chung bằng `joblib`.

Lần kiểm thử với `random_state=42`, calibrated Logistic Regression và cách chia 70/15/15 cho
kết quả trên test độc lập:

| Chỉ số | Giá trị |
|---|---:|
| PR-AUC | 0,2796 |
| ROC-AUC | 0,7059 |
| Recall nợ xấu | 0,6575 |
| Precision nợ xấu | 0,2441 |
| F1 | 0,3560 |
| Balanced accuracy | 0,6598 |
| Brier score | 0,1138 |
| Ngưỡng chọn trên validation | 0,15 |

Các con số phụ thuộc phiên bản dữ liệu và môi trường. Artifact đi kèm lưu cả
metric và ngưỡng để có thể đối chiếu; nên chạy lại lệnh huấn luyện khi dữ liệu
thay đổi.

## Nguồn dữ liệu và giới hạn

Tệp hiện có gồm 38.576 khoản vay và 24 cột. Cấu trúc của nó gần như trùng với
[Financial Loan Dataset của Aryan Singh trên Kaggle](https://www.kaggle.com/datasets/datawitharyan/financial-loan-dataset),
và một số định danh khớp dữ liệu LendingClub lịch sử. Tuy nhiên, bản CSV này đã
được đổi tên/rút gọn cột và thay đổi ngày; không có data card hay tệp giấy phép đi
kèm để chứng minh chính xác chuỗi nguồn gốc.

> Dataset được sử dụng cho mục đích học tập; nguồn gốc và quy trình thu thập ban đầu chưa được xác minh đầy đủ.

Do chưa xác định được giấy phép áp dụng cho đúng bản đã biến đổi, không nên phân
phối lại CSV trong repository công khai. Chi tiết về thời gian quan sát, ý nghĩa
cột và cách tạo nhãn nằm trong [data/README.md](data/README.md).

## Tránh rò rỉ dữ liệu

Mục tiêu là dự báo tại thời điểm cấp vay. Các trường phát sinh sau thời điểm đó sẽ
làm kết quả đẹp giả tạo và bị loại khỏi mô hình:

| Nhóm | Cột | Lý do loại |
|---|---|---|
| Kết quả | `loan_status` | Chính là nguồn tạo nhãn |
| Thanh toán | `total_payment` | Chỉ biết sau khi người vay bắt đầu trả nợ |
| Mốc hậu nghiệm | `last_payment_date`, `next_payment_date`, `last_credit_pull_date` | Chứa thông tin sau giải ngân |
| Định danh | `id`, `member_id` | Không có ý nghĩa dự báo ổn định |

`int_rate`, `grade` và `sub_grade` có thể chứa đánh giá rủi ro sẵn có của bên cho
vay. Chúng không nhất thiết là rò rỉ nếu đã có tại thời điểm quyết định, nhưng có
thể khiến mô hình học lại chính sách giá hiện hữu. Notebook vì vậy nên báo cáo
thêm phép so sánh có và không có `int_rate`, `sub_grade` trước khi diễn giải.

## Cấu trúc repository

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── Bank Loan Dataset.csv
├── notebooks/
│   └── Bank_Loan.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
├── reports/
├── artifacts/
├── configs/model.json
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── Dockerfile
└── .github/workflows/ci.yml
```

## Kiến trúc

```text
CSV -> schema guard -> point-in-time features -> split 70/15/15
                                      |
             train -> 5-fold model comparison -> calibration CV
                                      |
             validation -> cost-based threshold
                                      |
             test -> metrics + slice reports
                                      |
             joblib artifact -> FastAPI / Streamlit
```

## Cài đặt

Yêu cầu Python 3.11. Tạo môi trường riêng và cài đúng các phiên bản đã cố định:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Chạy dự án

Khám phá và tái tạo phân tích đầy đủ:

```bash
jupyter notebook notebooks/Bank_Loan.ipynb
```

Huấn luyện pipeline gọn từ dòng lệnh:

```bash
python -m src.train --data "data/Bank Loan Dataset.csv" --output artifacts/loan_default.joblib
```

Chạy test:

```bash
pytest -q
```

Chạy API và mở tài liệu tương tác tại `http://localhost:8000/docs`:

```bash
uvicorn app.api:app --reload
```

Chạy giao diện minh họa sau khi đã huấn luyện:

```bash
streamlit run app/streamlit_app.py
```

Chạy bằng Docker sau khi artifact đã được tạo:

```bash
docker build -t loan-default-risk .
docker run --rm -p 8000:8000 loan-default-risk
```

## Quy trình mô hình hóa

1. Chỉ giữ `Fully Paid` và `Charged Off`, sau đó tạo `default_flag`.
2. Chia train/validation/test có stratify trước khi dùng dữ liệu để ra quyết định.
3. Tạo đặc trưng từ thông tin có trước hoặc tại lúc giải ngân.
4. Điền thiếu, chuẩn hóa và one-hot encoding bên trong pipeline.
5. So sánh model bằng CV trên train, chọn champion theo PR-AUC.
6. Calibrate champion bằng CV nội bộ trên train.
7. Chọn ngưỡng trên validation theo chi phí false negative/false positive.
8. Mở test đúng một lần để báo cáo cuối cùng và phân tích slice.
9. Lưu pipeline, danh sách cột, định nghĩa nhãn, ngưỡng và metric trong một artifact.

## Model comparison trên train

| Model | PR-AUC CV | ROC-AUC CV | Balanced accuracy CV |
|---|---:|---:|---:|
| Logistic Regression | 0,2809 ± 0,0201 | 0,7033 ± 0,0116 | 0,6425 ± 0,0127 |
| Random Forest | 0,2778 ± 0,0090 | 0,6966 ± 0,0067 | 0,5935 ± 0,0109 |
| Dummy | 0,1423 ± 0,0001 | 0,5000 | 0,5000 |

Logistic Regression được chọn vì PR-AUC cao nhất, không phải vì phức tạp hơn.

Tách ngẫu nhiên phù hợp với bản hiện tại vì trường ngày đã bị biến đổi và chỉ phủ
năm 2021. Không nên gọi đây là out-of-time validation. Khi có ngày phát hành gốc
qua nhiều giai đoạn, cần chia theo thời gian và dành giai đoạn mới nhất làm test.

## Hướng phát triển

- So sánh Logistic Regression với CatBoost/XGBoost bằng cùng các fold.
- Tối ưu siêu tham số bằng `RandomizedSearchCV` trên train, không dùng test.
- Báo cáo hai kịch bản có/không có `int_rate` và `sub_grade`.
- Hiệu chỉnh chi phí FN/FP dựa trên tổn thất thực tế thay vì hệ số minh họa 5:1.
- Thêm out-of-time validation khi khôi phục được ngày gốc.
- Thêm SHAP cho mô hình cây hoặc odds ratio cho Logistic Regression.
- Theo dõi calibration và drift sau triển khai.

## Hạn chế và sử dụng có trách nhiệm

Dữ liệu không có tài liệu đầy đủ về quy trình thu thập, đại diện mẫu, định nghĩa
nợ xấu hay giấy phép. Các trường như địa lý và thông tin việc làm có thể tạo chênh
lệch giữa các nhóm. Trước mọi ứng dụng thực tế cần có kiểm định fairness, đánh giá
pháp lý, giám sát drift, quy trình giải trình và quyền khiếu nại của người vay.

Xem thêm [MODEL_CARD.md](MODEL_CARD.md) và các báo cáo sinh tự động trong
[reports/](reports/README.md).

## Gợi ý mô tả trong CV

- Xây dựng pipeline dự báo nợ xấu chống leakage trên 37.478 khoản vay có nhãn,
  dùng 5-fold CV và test độc lập; đạt ROC-AUC 0,706 và recall 65,8%.
- Thiết kế threshold theo chi phí FN:FP 5:1, sigmoid calibration và lưu chung
  preprocessing/model/threshold trong một artifact phục vụ FastAPI và Streamlit.
- Bổ sung schema guard, 6 unit test, slice analysis, Docker và GitHub Actions để
  chuyển notebook thành repository có thể cài đặt, kiểm thử và chạy lại.
