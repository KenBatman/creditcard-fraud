# 💳 Credit Card Fraud Detection

An interactive machine learning web application for detecting fraudulent credit card transactions.  
Built with **Python**, **Streamlit**

---

## 🚀 Features
- 📊 **Interactive Dashboard** — explore dataset and visualize fraud patterns.
- 🤖 **Model Training** — Logistic Regression and XGBoost with SMOTE balancing.
- 📈 **Evaluation Metrics** — classification reports, AUC, and confusion matrices.
- 💾 **Model Caching** — avoids retraining with `@st.cache_resource`.
- 🌐 **Web Deployment** — hosted publicly via Streamlit Cloud.

---

## 🧠 Tech Stack
- **Frontend:** Streamlit  
- **Backend:** Python  
- **ML Libraries:** scikit-learn, XGBoost, imbalanced-learn  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Data:** Kaggle Credit Card Fraud Dataset  

---

## ⚙️ How to Run Locally
```bash
git clone https://github.com/KenBatman/creditcard-fraud.git
cd creditcard-fraud-detection
pip install -r requirements.txt
streamlit run Interface.py

creditcard-fraud-detection/
│
├── Data/
│ └── creditcard.csv
│
├── models/
│ ├── logistic_regression.pkl
│ ├── xgboost.pkl
│
├── Interface.py
├── model.py
├── data.py
├── requirements.txt
└── README.md