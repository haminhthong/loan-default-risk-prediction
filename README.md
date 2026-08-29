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

Lần kiểm thử với `random_state=42`, calibrated Logistic Regression và test
out-of-time nửa cuối năm 2011 cho kết quả:

| Chỉ số | Giá trị |
|---|---:|
| PR-AUC | 0,3366 |
| ROC-AUC | 0,7196 |
| Recall nợ xấu | 0,5907 |
| Precision nợ xấu | 0,3036 |
| F1 | 0,4011 |
| Balanced accuracy | 0,6598 |
| Brier score | 0,1287 |
| Ngưỡng chọn trên validation | 0,15 |

Các con số phụ thuộc phiên bản dữ liệu và môi trường. Artifact đi kèm lưu cả
metric và ngưỡng để có thể đối chiếu; nên chạy lại lệnh huấn luyện khi dữ liệu
thay đổi.

## Nguồn dữ liệu và giới hạn

Mô hình chính dùng 39.717 khoản vay LendingClub với 111 cột và ngày phát hành từ
06/2007 đến 12/2011. Bộ dữ liệu này khôi phục ngày gốc và thay thế bản 24 cột có
ngày bị đổi sang năm 2021. Ba CSV bổ sung được lưu trong data catalog nhưng không
gộp vào model vì khác schema hoặc khác định nghĩa nhãn.

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
├── requirements-dev.txt
├── data/
│   ├── README.md
│   ├── raw/lendingclub_2007_2011.csv
│   └── external/
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
CSV -> schema guard -> point-in-time features -> split theo issue_date
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

Notebook trên được giữ làm lịch sử phân tích của bản dữ liệu 24 cột. Pipeline và
kết quả chính thức hiện nằm trong `src/` và dùng dữ liệu LendingClub gốc; không
dùng metric trong notebook legacy để mô tả phiên bản hiện tại.

Huấn luyện pipeline gọn từ dòng lệnh:

```bash
python -m src.train
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
2. Chia train/validation/test theo `issue_date`, giữ giai đoạn mới nhất làm test.
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
| Logistic Regression | 0,2485 ± 0,0172 | 0,6898 ± 0,0072 | 0,6385 ± 0,0111 |
| Random Forest | 0,2360 ± 0,0153 | 0,6821 ± 0,0114 | 0,5352 ± 0,0051 |
| Dummy | 0,1313 ± 0,0001 | 0,5000 | 0,5000 |

Logistic Regression được chọn vì PR-AUC cao nhất, không phải vì phức tạp hơn.

## Pricing-feature ablation

| Feature set | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Có `int_rate`, `sub_grade` | 0,3366 | 0,7196 | 0,5907 | 0,3036 | 0,4011 |
| Không có pricing features | 0,3384 | 0,7195 | 0,6283 | 0,2924 | 0,3991 |

Loại pricing features không làm giảm khả năng xếp hạng và còn tăng nhẹ PR-AUC.
Điều này cho thấy mô hình không chỉ học lại mức lãi suất hoặc sub-grade sẵn có.
Artifact triển khai vẫn giữ phiên bản có pricing để tương thích với protocol đã
chốt; báo cáo ablation là bằng chứng cần cân nhắc trước phiên bản tiếp theo.

## Threshold theo chi phí

| Chi phí FN:FP | Threshold | Recall validation | Precision validation |
|---|---:|---:|---:|
| 2:1 | 0,37 | 0,0486 | 0,4063 |
| 5:1 | 0,15 | 0,6114 | 0,2515 |
| 10:1 | 0,08 | 0,9028 | 0,1895 |

Các kết quả thể hiện trade-off rất lớn. Tỷ lệ 5:1 chỉ là kịch bản minh họa; một
hệ thống thật phải thay bằng EAD, LGD và lợi nhuận cơ hội của từng khoản vay.

## Calibration, giải thích và drift

- Brier score trên test: `0,1287`.
- Ở bin xác suất `0,2–0,3`, xác suất trung bình là `0,239` nhưng default rate thực
  tế là `0,333`; mô hình vẫn đánh giá thấp rủi ro ở vùng này.
- Các biến drift mạnh ngoài mùa phát hành gồm `verification_status` (PSI `0,221`),
  `interest_rate` (`0,220`) và `purpose` (`0,203`).
- Odds ratio được xuất cho Logistic Regression. Ví dụ `small_business` có odds
  dự báo cao hơn khoảng `1,80×`; đây là quan hệ dự báo, không phải quan hệ nhân quả.

Train gồm các khoản vay trước năm 2011, validation là nửa đầu năm 2011 và test là
nửa cuối năm 2011. Cách chia này đo suy giảm qua thời gian tốt hơn random split,
nhưng vẫn chưa thay thế external validation trên một tổ chức hoặc giai đoạn khác.

## Hướng phát triển

- So sánh Logistic Regression với CatBoost/XGBoost bằng cùng các fold.
- Tối ưu siêu tham số bằng `RandomizedSearchCV` trên train, không dùng test.
- Thay cost ratio minh họa bằng expected loss dựa trên EAD/LGD thật.
- Thêm external validation trên nguồn dữ liệu có định nghĩa nhãn tương thích.
- Thêm SHAP nếu champion tương lai là mô hình cây.
- Định nghĩa ngưỡng cảnh báo drift và lịch tái huấn luyện.

## Hạn chế và sử dụng có trách nhiệm

Dữ liệu không có tài liệu đầy đủ về quy trình thu thập, đại diện mẫu, định nghĩa
nợ xấu hay giấy phép. Các trường như địa lý và thông tin việc làm có thể tạo chênh
lệch giữa các nhóm. Trước mọi ứng dụng thực tế cần có kiểm định fairness, đánh giá
pháp lý, giám sát drift, quy trình giải trình và quyền khiếu nại của người vay.

Xem thêm [MODEL_CARD.md](MODEL_CARD.md) và các báo cáo sinh tự động trong
[reports/](reports/README.md).

## Gợi ý mô tả trong CV

- Xây dựng pipeline dự báo nợ xấu chống leakage trên 38.577 khoản vay đã kết thúc,
  dùng 5-fold CV và test out-of-time; đạt ROC-AUC 0,720 và PR-AUC 0,337.
- Thiết kế threshold theo chi phí FN:FP 5:1, sigmoid calibration và lưu chung
  preprocessing/model/threshold trong một artifact phục vụ FastAPI và Streamlit.
- Bổ sung schema guard, 8 unit test, slice/drift/calibration analysis, Docker và GitHub Actions để
  chuyển notebook thành repository có thể cài đặt, kiểm thử và chạy lại.
