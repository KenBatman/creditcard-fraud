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
st.set_page_config(page_title="Ken's Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")

# ======= SIDEBAR =======
st.sidebar.header("Dataset & Navigation")

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
nav = st.sidebar.radio(
    "Chọn màn hình:",
    ["Overview", "Visualizations", "Train Models", "User Prediction"],
    key="view"
)

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
        st.subheader("Huấn luyện & Đánh giá Model")

        st.markdown("""
        Chức năng này sẽ **huấn luyện hoặc tự động tải lại** các mô hình đã được huấn luyện trước đó:
        - Logistic Regression  
        - XGBoost""")
        # Nút train / load model
        if st.button("Bắt đầu huấn luyện hoặc tải model"):
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
        # Hàm tách riêng UI + logic cho tab này (clean code)
        def user_prediction_ui(_df):
            st.subheader("Dự đoán giao dịch từ người dùng")
            
            # --- Load Sample từ Dataset (nếu có df) ---
            fraud_sample = None
            non_fraud_sample = None
            if _df is not None:
                df_sample = _df.copy()
                if len(df_sample[df_sample['Class'] == 1]) > 0:
                    fraud_sample = df_sample[df_sample['Class'] == 1].sample(n=1, random_state=42)
                if len(df_sample[df_sample['Class'] == 0]) > 0:
                    non_fraud_sample = df_sample[df_sample['Class'] == 0].sample(n=1, random_state=42)
            
            # Hiển thị samples nếu có (columns động)
            num_samples = sum(1 for s in [fraud_sample, non_fraud_sample] if s is not None)
            if num_samples > 0:
                st.markdown("### 📋 Load Sample từ Dataset")
                if num_samples == 1:
                    col_sample = st.columns(1)
                    current_col = col_sample[0]
                else:
                    col_sample1, col_sample2 = st.columns(2)
                    current_col = col_sample1 if fraud_sample is not None else col_sample2
                
                # Sample Fraud (ưu tiên col1)
                if fraud_sample is not None:
                    with (col_sample1 if num_samples == 2 else current_col):
                        st.markdown("**Fraud Sample**")
                        sample_data_fraud = fraud_sample.iloc[0].to_dict()
                        if st.button("📥 Load Fraud Sample", key="load_fraud"):
                            st.session_state["input_sample"] = sample_data_fraud
                            if "last_input" in st.session_state:
                                del st.session_state["last_input"]
                            st.rerun()  # Rerun chỉ để load sample vào form (chưa predict, không mất kết quả)
                        st.write(f"Preview: Time={sample_data_fraud['Time']:.1f}, Amount={sample_data_fraud['Amount']:.2f}")
                    
                    if num_samples == 2:
                        current_col = col_sample2
                
                # Sample Non-Fraud
                if non_fraud_sample is not None:
                    with (col_sample2 if num_samples == 2 else current_col):
                        st.markdown("**Non-Fraud Sample**")
                        sample_data_nonfraud = non_fraud_sample.iloc[0].to_dict()
                        if st.button("📥 Load Non-Fraud Sample", key="load_nonfraud"):
                            st.session_state["input_sample"] = sample_data_nonfraud
                            if "last_input" in st.session_state:
                                del st.session_state["last_input"]
                            st.rerun()
                        st.write(f"Preview: Time={sample_data_nonfraud['Time']:.1f}, Amount={sample_data_nonfraud['Amount']:.2f}")
            
            st.markdown("---")  # Divider
            
            st.markdown("""
            Nhập giá trị cho các đặc trưng **V1–V28**, **Time** và **Amount** để hệ thống dự đoán xem
            giao dịch có phải là **gian lận (Fraud)** hay không.
            """)
            
            # --- Chọn model ---
            model_choice = st.selectbox("Chọn mô hình để dự đoán:", ["Logistic Regression", "XGBoost"])
            
            # --- Form để tránh rerun liên tục ---
            with st.form(key="prediction_form"):
                st.markdown("### Nhập dữ liệu đầu vào")
                
                # Lấy sample từ session_state
                sample_data = st.session_state.get("input_sample", None)
                
                col1, col2 = st.columns(2)
                with col1:
                    default_time = float(sample_data['Time']) if sample_data is not None else 0.0
                    time_input = st.number_input("⏱️ Time", value=default_time, step=0.01)
                with col2:
                    default_amount = float(sample_data['Amount']) if sample_data is not None else 0.0
                    amount_input = st.number_input("💰 Amount", value=default_amount, step=0.01)
                
                st.markdown("#### 🔢 Các đặc trưng V1 - V28")
                v_inputs = {}
                for i in range(1, 29):
                    default_v = float(sample_data[f'V{i}']) if sample_data is not None else 0.0
                    v_inputs[f"V{i}"] = st.number_input(f"V{i}", value=default_v, step=0.01)
                
                # Nút submit
                submitted = st.form_submit_button("Dự đoán giao dịch")
            
            # --- Hiển thị kết quả persistent (nếu có từ session_state hoặc vừa submit) ---
            if "prediction_result" in st.session_state:
                st.markdown("---")
                stored_prediction, stored_probability, stored_model = st.session_state["prediction_result"]
                if stored_prediction == 1:
                    st.error(f"🚨 Giao dịch có khả năng **GIAN LẬN** (xác suất: {stored_probability:.2%}) - Model: {stored_model}")
                else:
                    st.success(f"✅ Giao dịch **BÌNH THƯỜNG** (xác suất gian lận: {stored_probability:.2%}) - Model: {stored_model}")
                
                # Nút clear results (user chủ động reset)
                if st.button("🗑️ Clear Kết Quả & Form"):
                    if "input_sample" in st.session_state:
                        del st.session_state["input_sample"]
                    if "prediction_result" in st.session_state:
                        del st.session_state["prediction_result"]
                    st.rerun()  # Rerun chỉ khi clear
            
            # Chỉ xử lý khi submit (lưu vào session_state để persistent)
            if submitted:
                # Tạo input_data
                input_data = {"Time": time_input, "Amount": amount_input}
                input_data.update(v_inputs)
                
                try:
                    from models import user_predict
                    with st.spinner("⏳ Đang xử lý dữ liệu..."):
                        prediction, probability = user_predict(input_data, model_name=model_choice)
                    
                    # Lưu kết quả vào session_state (bao gồm model để display)
                    st.session_state["prediction_result"] = (prediction, probability, model_choice)
                    
                    # Clear sample sau predict (không rerun ngay, để giữ kết quả)
                    if "input_sample" in st.session_state:
                        del st.session_state["input_sample"]
                    
                    # Rerun để refresh UI hiển thị kết quả mới (nhưng persistent nhờ state)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi dự đoán: {e}")
        
        # Gọi hàm tách 
        user_prediction_ui(df)