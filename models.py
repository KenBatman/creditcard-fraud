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

def load_and_preprocess_data(csv_path="Data/creditcard.csv"):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df['normTime'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    df = df.drop(['Amount', 'Time'], axis=1)
    tar_col = "Class"
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    num_cols = [col for col in num_cols if col != tar_col]
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test, X.columns

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

def train_and_save_models():
    os.makedirs("models", exist_ok=True)

    # ⚡ Nếu models đã tồn tại, load lại thay vì train lại
    if os.path.exists("models/logistic_regression.pkl") and os.path.exists("models/xgboost.pkl"):
        print("📂 Models đã tồn tại — đang load lại từ thư mục /models ...")

        # Load models
        lr_model = joblib.load("models/logistic_regression.pkl")
        xgb_model = joblib.load("models/xgboost.pkl")

        # Load dữ liệu để đánh giá lại model
        X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
        results = {
            "Logistic Regression": evaluate_model(lr_model, X_test, y_test),
            "XGBoost": evaluate_model(xgb_model, X_test, y_test)
        }
        return results

    # ⚙️ Nếu chưa có model — tiến hành train mới
    print("🚀 Đang huấn luyện models mới...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
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
  
