import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os
import streamlit as st  # Thêm import st để dùng session_state nếu cần (tùy chọn)

def load_and_preprocess_data(df=None, csv_path="Data/creditcard.csv"):
    """
    Load và preprocess data. Ưu tiên dùng df nếu cung cấp (từ upload), fallback load CSV.
    """
    try:
        if df is not None:
            # Dùng df từ upload (không cần read CSV)
            print("📂 Sử dụng df từ upload...")
            df_local = df.copy()
        else:
            # Fallback load CSV
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"❌ File CSV không tồn tại: {csv_path}. Vui lòng upload dataset hoặc đặt file ở Data/creditcard.csv.")
            print(f"📂 Đang load từ {csv_path}...")
            df_local = pd.read_csv(csv_path)
        
        df_local = df_local.drop_duplicates()
        print(f"📊 Dữ liệu sau drop duplicates: {df_local.shape}")
        
        # Fit và transform Amount/Time với scaler riêng, rồi dump chúng
        amount_scaler = StandardScaler()
        df_local['normAmount'] = amount_scaler.fit_transform(df_local[['Amount']])
        os.makedirs("models", exist_ok=True)
        joblib.dump(amount_scaler, "models/amount_scaler.pkl")
        print("✅ Dumped amount_scaler.pkl")
        
        time_scaler = StandardScaler()
        df_local['normTime'] = time_scaler.fit_transform(df_local[['Time']])
        joblib.dump(time_scaler, "models/time_scaler.pkl")
        print("✅ Dumped time_scaler.pkl")
        
        df_local = df_local.drop(['Amount', 'Time'], axis=1)
        tar_col = "Class"
        X = df_local.drop('Class', axis=1)
        y = df_local['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        print("✅ Preprocess hoàn tất. Train shape:", X_train.shape)
        return X_train, X_test, y_train, y_test, X.columns
        
    except FileNotFoundError as e:
        # Raise để UI catch và show error
        raise RuntimeError(f"Lỗi load data: {e}. Đảm bảo upload dataset creditcard.csv.")
    except Exception as e:
        raise RuntimeError(f"Lỗi preprocess: {e}")

def train_logistic_regression(X_train, y_train):
    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.3)),
        ('model', LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_xgboost(X_train, y_train):
    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.3)),
        ('model', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    return {"report": report, "auc": auc, "confusion_matrix": cm}

def train_and_save_models(df=None):  # Thêm param df để pass từ Interface.py
    os.makedirs("models", exist_ok=True)

    # ⚡ Nếu models đã tồn tại, load lại thay vì train lại
    if os.path.exists("models/logistic_regression.pkl") and os.path.exists("models/xgboost.pkl"):
        print("📂 Models đã tồn tại — đang load lại từ thư mục /models ...")

        # Load models
        lr_model = joblib.load("models/logistic_regression.pkl")
        xgb_model = joblib.load("models/xgboost.pkl")

        # Load dữ liệu để đánh giá lại model (pass df nếu có)
        X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(df=df)
        results = {
            "Logistic Regression": evaluate_model(lr_model, X_test, y_test),
            "XGBoost": evaluate_model(xgb_model, X_test, y_test)
        }
        return results

    # ⚙️ Nếu chưa có model — tiến hành train mới
    print("🚀 Đang huấn luyện models mới...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(df=df)
    results = {}

    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    joblib.dump(lr_model, "models/logistic_regression.pkl")
    results["Logistic Regression"] = evaluate_model(lr_model, X_test, y_test)

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    joblib.dump(xgb_model, "models/xgboost.pkl")
    results["XGBoost"] = evaluate_model(xgb_model, X_test, y_test)

    print("✅ Đã huấn luyện và lưu models vào thư mục /models")
    return results

def user_predict(input_data: dict, model_name="Logistic Regression"):
    """
    Dự đoán giao dịch mới dựa trên input user nhập vào.
    input_data: dict có các cột ['V1'...'V28', 'Amount', 'Time']
    model_name: "Logistic Regression" hoặc "XGBoost"
    """
    try:
        # Load hai scaler đã fit
        if not os.path.exists("models/amount_scaler.pkl"):
            raise FileNotFoundError("❌ Chưa train models. Vui lòng chạy Tab 'Train Models' trước.")
        amount_scaler = joblib.load("models/amount_scaler.pkl")
        time_scaler = joblib.load("models/time_scaler.pkl")

        # Load model
        model_path = (
            "models/logistic_regression.pkl"
            if model_name == "Logistic Regression"
            else "models/xgboost.pkl"
        )
        model = joblib.load(model_path)

        # Tạo DataFrame từ input
        df_input = pd.DataFrame([input_data])
        
        # Transform Amount và Time riêng
        df_input['normAmount'] = amount_scaler.transform(df_input[['Amount']])
        df_input['normTime'] = time_scaler.transform(df_input[['Time']])
        
        # Drop cột gốc
        df_input = df_input.drop(['Amount', 'Time'], axis=1)
        
        # Dự đoán
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        
        return int(prediction), float(probability)  # Trả về tuple để khớp Interface.py
        
    except Exception as e:
        raise RuntimeError(f"Lỗi trong quá trình dự đoán: {e}")