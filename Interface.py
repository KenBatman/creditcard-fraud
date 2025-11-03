import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import ConfusionMatrixDisplay
from models import train_and_save_models, load_and_preprocess_data
from data import (
    load_data,
    plot_class_distribution,
    plot_time_density,
    plot_boxplots,
    plot_correlation
)
st.set_option('client.showErrorDetails', True)
@st.cache_resource(show_spinner=False)
def get_models(df=None):
    """Huấn luyện hoặc load models đã tồn tại."""
    return train_and_save_models(df=df)

# ======= PAGE CONFIG =======
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")

# ======= SIDEBAR =======
st.sidebar.header("⚙️ Dataset & Navigation")

# Cho phép người dùng upload file CSV
uploaded_file = st.sidebar.file_uploader("📂 Tải lên file CSV của bạn:", type=["csv"])

if uploaded_file is not None:
    try:
        df, info = load_data(uploaded_file)
        st.session_state["df"] = df
        st.session_state["info"] = info
        st.sidebar.success("✅ Đã tải và xử lý dữ liệu thành công!")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi đọc file: {e}")
else:
    st.sidebar.info("Vui lòng tải lên file dataset (.csv) để bắt đầu.")

# Navigation
if "view" not in st.session_state:
    st.session_state["view"] = "Overview"

nav = st.sidebar.radio(
    "🧭 Chọn màn hình:",
    ["Overview", "Visualizations", "Train Models", "User Prediction"],
    index=["Overview", "Visualizations", "Train Models", "User Prediction"].index(st.session_state["view"])
)
st.session_state["view"] = nav

# ======= MAIN CONTENT =======
if "df" not in st.session_state:
    st.info("Hãy Upload dataset bạn tại đây.")
else:
    df = st.session_state["df"]
    info = st.session_state["info"]

    # ============ TAB 1: OVERVIEW ============
    if nav == "Overview":
        st.subheader("📋 Thông tin tổng quan")
        c1, c2, c3 = st.columns(3)
        c1.metric("Số dòng ban đầu", info["shape_original"][0])
        c2.metric("Số cột", info["shape_original"][1])
        c3.metric("Số dòng sau xử lý", info["shape_final"][0])

        st.markdown("### 🔁 Dòng trùng lặp bị xóa")
        st.write(f"**{info['duplicates_removed']}** dòng")

        st.markdown("### ❗ Missing values")
        st.dataframe(info["missing_df"])

        st.markdown("### 🚫 Cột hằng")
        if info["constant_cols"]:
            st.write(", ".join(info["constant_cols"]))
        else:
            st.success("Không có cột hằng")

        st.markdown("### ⚠️ Outliers (IQR)")
        if not info["outliers"].empty:
            st.dataframe(info["outliers"])
        else:
            st.success("Không phát hiện outlier đáng kể")

    # ============ TAB 2: VISUALIZATIONS ============
    elif nav == "Visualizations":
        st.subheader("📊 Visualizations - Các biểu đồ tổng quan")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ⚖️ Phân bố Class")
            plot_class_distribution(df)
        with col2:
            st.markdown("#### ⏱️ Phân bố theo Thời gian")
            plot_time_density(df)

        st.markdown("---")
        st.markdown("#### 📦 Boxplots (V Features)")
        plot_boxplots(df)

        st.markdown("---")
        st.markdown("#### 🔥 Correlation Heatmap")
        with st.expander("Hiển thị Heatmap (Plotly)", expanded=False):
            corr = df.corr().round(3)
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale='Reds',
                zmin=-1, zmax=1,
                colorbar=dict(title="Pearson r")
            ))
            fig.update_layout(height=600, margin=dict(l=60, r=10, t=40, b=60))
            st.plotly_chart(fig, use_container_width=True)

    # ============ TAB 3: TRAIN MODELS ============
    elif nav == "Train Models":
        st.subheader("🤖 Huấn luyện & Đánh giá Model")

        st.markdown("""
        Chức năng này sẽ **huấn luyện hoặc tự động tải lại** hai mô hình:
        - Logistic Regression  
        - XGBoost  
        
        👉 Nếu các model `.pkl` đã tồn tại trong thư mục `models/`, hệ thống **sẽ không train lại** mà chỉ đánh giá lại trên dữ liệu mới nhất.
        """)
        # Nút train / load model
        if st.button("🚀 Bắt đầu huấn luyện hoặc tải model"):
            with st.spinner("⏳ Đang xử lý... (lần đầu có thể hơi lâu)"):
                results = get_models(df=st.session_state.get("df"))  # Cache + kiểm tra .pkl tự động
            st.success("✅ Hoàn tất! Models đã sẵn sàng sử dụng.")

            # Hiển thị kết quả chi tiết cho từng model
            for model_name, res in results.items():
                st.markdown(f"### 📈 {model_name}")

                colA, colB = st.columns([1, 1])
                colA.metric("AUC-ROC", f"{res['auc']:.4f}")

                report_df = pd.DataFrame(res["report"]).transpose()
                with colB:
                    st.dataframe(report_df.style.format(precision=3))

                # Vẽ Confusion Matrix với tỉ lệ cố định
                cm = res["confusion_matrix"]
                fig, ax = plt.subplots(figsize=(2, 2.5)) 
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    cbar=False, square=True,  # square=True giữ ô vuông
                    xticklabels=['Non-Fraud', 'Fraud'],
                    yticklabels=['Non-Fraud', 'Fraud'],
                    ax=ax
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix - {model_name}", fontsize=10, pad=6)

                # Giữ bố cục gọn, không lệch label
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            st.markdown("---")
        # ============ TAB 4: USER PREDICTION ============
    elif nav == "User Prediction":
        st.subheader("🧾 Dự đoán giao dịch mới")

        st.markdown("""
        Nhập giá trị cho các đặc trưng **V1–V28**, **Time** và **Amount** để hệ thống dự đoán xem
        giao dịch có phải là **gian lận (Fraud)** hay không.
        """)

        # --- Chọn model ---
        model_choice = st.selectbox("Chọn mô hình để dự đoán:", ["Logistic Regression", "XGBoost"])

        # --- Form để tránh rerun liên tục ---
        with st.form(key="prediction_form"):
            st.markdown("### ✏️ Nhập dữ liệu đầu vào")
            col1, col2 = st.columns(2)
            with col1:
                time_input = st.number_input("⏱️ Time", value=0.0)
            with col2:
                amount_input = st.number_input("💰 Amount", value=0.0)

            st.markdown("#### 🔢 Các đặc trưng V1 - V28")
            v_inputs = {}
            for i in range(1, 29):
                v_inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.01)

            # Nút submit trong form
            submitted = st.form_submit_button("🚀 Dự đoán giao dịch")

        # Chỉ xử lý khi submit
        if submitted:
            # Tạo input_data từ các inputs
            input_data = {"Time": time_input, "Amount": amount_input}
            input_data.update(v_inputs)

            try:
                from models import user_predict  # hàm này bạn đã viết ở model.py
                with st.spinner("⏳ Đang xử lý dữ liệu..."):
                    prediction, probability = user_predict(input_data, model_name=model_choice)

                st.markdown("---")
                if prediction == 1:
                    st.error(f"🚨 Giao dịch có khả năng **GIAN LẬN** (xác suất: {probability:.2%})")
                else:
                    st.success(f"✅ Giao dịch **BÌNH THƯỜNG** (xác suất gian lận: {probability:.2%})")

            except Exception as e:
                st.error(f"❌ Lỗi khi dự đoán: {e}")
        
