"""Giao diện dự báo thử từ một tệp CSV."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import load_artifact, predict  # noqa: E402

st.set_page_config(page_title="Rủi ro vỡ nợ", page_icon="🏦", layout="wide")
st.title("Dự báo rủi ro vỡ nợ khoản vay")
st.caption("Kết quả chỉ phục vụ minh họa học tập, không dùng để ra quyết định tín dụng thực tế.")

model_path = ROOT / "models" / "loan_default.joblib"
if not model_path.exists():
    st.error("Chưa có mô hình. Hãy chạy `python -m src.train` trước.")
    st.stop()

uploaded = st.file_uploader("Chọn CSV có cùng cấu trúc dữ liệu huấn luyện", type="csv")
if uploaded is not None:
    input_data = pd.read_csv(uploaded)
    artifact = load_artifact(model_path)
    result = pd.concat([input_data, predict(input_data, artifact)], axis=1)
    st.metric("Ngưỡng quyết định", f"{artifact['threshold']:.2f}")
    st.dataframe(result, use_container_width=True)
    st.download_button(
        "Tải kết quả",
        result.to_csv(index=False).encode("utf-8-sig"),
        file_name="loan_default_predictions.csv",
        mime="text/csv",
    )
