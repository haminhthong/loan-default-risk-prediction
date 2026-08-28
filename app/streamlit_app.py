"""Giao diện dự báo thử từ một tệp CSV."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import load_artifact, predict  # noqa: E402

MODEL_PATH = ROOT / "artifacts" / "loan_default.joblib"


@st.cache_resource
def get_artifact():
    """Nạp mô hình một lần trong mỗi phiên chạy Streamlit."""
    return load_artifact(MODEL_PATH)


def main() -> None:
    """Hiển thị giao diện tải CSV và kết quả dự báo."""
    st.set_page_config(page_title="Rủi ro vỡ nợ", page_icon="🏦", layout="wide")
    st.title("Dự báo rủi ro vỡ nợ khoản vay")
    st.caption(
        "Kết quả chỉ phục vụ minh họa học tập, "
        "không dùng để ra quyết định tín dụng thực tế."
    )

    if not MODEL_PATH.exists():
        st.error("Chưa có mô hình. Hãy chạy `python -m src.train` trước.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Chọn CSV có cùng cấu trúc dữ liệu huấn luyện",
        type="csv",
    )
    if uploaded_file is None:
        return

    input_data = pd.read_csv(uploaded_file)
    artifact = get_artifact()
    result = pd.concat([input_data, predict(input_data, artifact)], axis=1)

    st.metric("Ngưỡng quyết định", f"{artifact['threshold']:.2f}")
    st.dataframe(result, use_container_width=True)
    st.download_button(
        "Tải kết quả",
        result.to_csv(index=False).encode("utf-8-sig"),
        file_name="loan_default_predictions.csv",
        mime="text/csv",
    )


main()
