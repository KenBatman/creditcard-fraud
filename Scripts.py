# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
import datetime
import sklearn as sk
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from plotly.offline import iplot

# %% [markdown]
# # Data Exploration

# %%
df = pd.read_csv('Data/creditcard.csv')
print(df.shape)
df

# %%
print(df.info())

# %%
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/ df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# %% [markdown]
# *Kiểm tra missing data như Null + rỗng + ký hiệu đặc biệt*

# %%
df.describe()

# %% [markdown]
# *Nhìn vào cột Time, có thể thấy bộ dữ liệu có tổng cộng 284,807 giao dịch, tổng cộng có 31 cột*
# 

# %%
df.shape

# %%
creditcardtransdata = pd.read_csv('Data/creditcard.csv' , low_memory=False)
creditcardtransdata.hist(figsize = (25, 25),color = "#DAA520")
plt.show()

# %% [markdown]
# *Biểu đồ này là biểu đồ tần suất (histogram), trục X là giá trị của biến, trục Y là tần suất xuất hiện của biến đó, thể hiện phân bố giá trị của từng đặc trưng trong dataset. Ta thấy hầu hết các biến V1–V28 có dạng gần chuẩn quanh 0 (do đã qua PCA), trong khi biến Time có hai đỉnh rõ rệt, cho thấy hai giai đoạn giao dịch khác nhau trong dữ liệu*

# %%
classes=df['Class'].value_counts()
normal_share=classes[0]/df['Class'].count()*100
fraud_share=classes[1]/df['Class'].count()*100

plt.bar(['Non-Fraud','Fraud'], df['Class'].value_counts(), color=['#17a2b8','#c2185b'])
plt.xlabel('Class')
plt.ylabel('Number of transactions')
plt.annotate('{}\n({:.4}%)'.format(classes[0],
                                         df['Class'].value_counts()[0]/df['Class'].count()*100),
             (0.20, 0.45), xycoords='axes fraction')
plt.annotate('{}\n({:.4}%)'.format(classes[1],
                                         df['Class'].value_counts()[1]/df['Class'].count()*100),
             (0.70, 0.45), xycoords='axes fraction')
plt.tight_layout()
plt.show()

# %% [markdown]
# *chỉ có 492 giao dịch là giao dịch gian lận, từ đó ta có thể thấy bộ dataset bất cân bằng, cần được xử lý khi trước khi đưa vào model*

# %%
class_0 = df.loc[df['Class'] == 0]["Time"]
class_1 = df.loc[df['Class'] == 1]["Time"]

hist_data = [class_0, class_1]
group_labels = ['Non-Fraud', 'Fraud']

fig = ff.create_distplot(
    hist_data,
    group_labels,
    show_hist=False,
    show_rug=False,
    colors=['#1f77b4', '#d62728']
)

fig.update_layout(
    title=dict(
        text='Credit Card Transactions - Time Density Distribution',
        x=0.5,
        font=dict(size=20, family='Arial', color='#333')
    ),
    xaxis=dict(
        title='Normalized Transaction Time',
        gridcolor='rgba(200,200,200,0.3)',
        zeroline=False
    ),
    yaxis=dict(
        title='Density',
        gridcolor='rgba(200,200,200,0.3)',
        zeroline=False
    ),
    plot_bgcolor='white',
    legend=dict(
        title='Transaction Type',
        orientation='h',
        yanchor='bottom',
        y=0.98,
        xanchor='center',
        x=0.5
    )
)
fig.show()

# %% [markdown]
# *Các giao dịch gian lận có sự phân bố đồng đều hơn các giao dịch hợp lệ – được phân bố đều theo thời gian, bao gồm cả những thời điểm có ít giao dịch thực tế, như vào ban đêm theo múi giờ Châu Âu.*
# *Trục hoành là thời gian ( tính theo giây) được tính kể từ giao dịch đầu tiên
# Trục tung là mật độ xuất hiện giao dịch*

# %% [markdown]
# ### Phân tích mối quan hệ giữa các biến V và Class

# %%
selected_v_cols = ['V14', 'V17', 'V10', 'V12']

# Adjusted figsize
plt.figure(figsize=(12, 8))
for i, col in enumerate(selected_v_cols):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='Class', y=col, data=df, palette='viridis')
    plt.title(f'Distribution of {col} by Class')
plt.tight_layout()
plt.show()

# %% [markdown]
# *Insight từ biểu đồ:*
# *   Biểu đồ trên cho thấy sự khác biệt đáng kể trong phân bố giá trị của các biến V10, V12, V14, và V17 giữa các giao dịch không gian lận (Class 0) và giao dịch gian lận (Class 1).
# *   Đặc biệt, các biến V14 và V17 (là kết quả từ PCA) có vẻ là những đặc trưng rất quan trọng trong việc phân biệt hai lớp, với các giá trị trung bình và phân bố khác biệt rõ rệt cho lớp gian lận. Điều này gợi ý rằng những biến này mang thông tin mạnh mẽ liên quan đến hành vi gian lận.
# *   Sự khác biệt trong phân bố này xác nhận rằng các biến V đã được biến đổi thông qua PCA vẫn giữ được khả năng phân tách giữa hai loại giao dịch, và chúng sẽ là những đặc trưng hữu ích cho mô hình phân loại.

# %% [markdown]
# ### Mối tương quan giữa các biến V và Class

# %%
tar_col = "Class"
num_cols = df.select_dtypes(include=['number']).columns.tolist() #lấy tất cả các cột kiểu số trong df (numeric)
# Loại bỏ cột Class khỏi danh sách
num_cols = [col for col in num_cols if col != tar_col]
print(num_cols)

# %%
# Tính toán mối tương quan giữa các biến V và Class
v_class_corr = df.corr()['Class'].sort_values(ascending=False)

# Adjusted figsize
plt.figure(figsize=(6, 7))
sns.barplot(x=v_class_corr.values, y=v_class_corr.index, palette='coolwarm')
plt.title('Correlation of V variables with Class')
plt.xlabel('Correlation Coefficient')
plt.ylabel('V Variable')
plt.show()

# %% [markdown]
# *Insight từ biểu đồ:*
# *   Biểu đồ mối tương quan xác nhận rằng một số biến V (đặc biệt là V17, V14, V12, V10) có mối tương quan âm mạnh nhất với biến Class. Điều này có nghĩa là khi giá trị của các biến này giảm, khả năng giao dịch là gian lận (Class 1) tăng lên.
# *   Ngược lại, các biến như V2, V4, V11 có mối tương quan dương với Class, cho thấy khi giá trị của chúng tăng, khả năng giao dịch là gian lận cũng tăng.
# *   Những biến có độ tương quan cao (cả dương và âm) với Class là những đặc trưng tiềm năng quan trọng nhất để mô hình học và phân biệt giao dịch gian lận.

# %%
constant_cols = [col for col in df.columns if df[col].nunique() == 1] #check duyệt qua từng cột xem có cột nào có 1 giá trị hay thôi, nunique (đếm số lượng giá trị duy nhất)
if constant_cols:
    print("Các cột hằng tìm thấy:", constant_cols)
else:
    print("Không có cột hằng nào.")

# %%
#2. Kiểm tra hàng trùng lặp (duplicate rows)
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    print(f"Số hàng trùng lặp: {duplicate_count}")
else:
    print("Không có hàng trùng lặp.")

# %%
#Xóa các hàng trùng lặp và cập nhật DataFrame
df = df.drop_duplicates()
# Kiểm tra lại số lượng hàng sau khi xóa
print("Số hàng sau khi xóa trùng lặp:", df.shape[0])

# %%
Q1 = df[num_cols].quantile(0.25) #xem lại công thức tính giá trị ngoại lai
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

# Xác định outliers (giá trị nhỏ hơn Q1 - 1.5*IQR hoặc lớn hơn Q3 + 1.5*IQR)
outliers_iqr = ((df[num_cols] < (Q1 - 1.5 * IQR)) |
                (df[num_cols] > (Q3 + 1.5 * IQR)))

# Đếm số lượng outlier trên mỗi cột
outlier_counts_iqr = outliers_iqr.sum()
print("Số lượng giá trị ngoại lai (IQR) trên mỗi cột:")
print(outlier_counts_iqr[outlier_counts_iqr > 0])

# %%
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=False)
plt.show();

# %% [markdown]
# *Insight từ biểu đồ Boxplot (Amount vs Class):*
# 
# *   **Sự khác biệt trong phân bố Amount:** Biểu đồ cho thấy sự khác biệt rõ rệt về phân bố số tiền giao dịch giữa hai lớp. Các giao dịch không gian lận (Class 0) có phạm vi số tiền lớn hơn và nhiều outlier với giá trị rất cao. Ngược lại, các giao dịch gian lận (Class 1) tập trung ở mức số tiền nhỏ hơn nhiều, với ít outlier có giá trị cao.
# *   **Tầm quan trọng của Outliers:** Biểu đồ `showfliers=True` cho thấy các outlier trong lớp không gian lận có thể có giá trị rất lớn, trong khi các outlier trong lớp gian lận có xu hướng nhỏ hơn nhiều. Điều này củng cố nhận định rằng các giao dịch gian lận thường có giá trị nhỏ hơn.
# *   **Giá trị trung bình và phân vị:** Dù biểu đồ `showfliers=False` ẩn đi các outlier cực đoan, nó vẫn cho thấy hộp (box) của lớp gian lận nằm ở mức giá trị thấp hơn đáng kể so với lớp không gian lận. Điều này cho thấy phần lớn các giao dịch gian lận có số tiền nhỏ.
# *   **Kết luận về tính hữu ích:** Biểu đồ này **cần thiết** vì nó trực quan hóa trực tiếp một trong những đặc trưng ban đầu (`Amount`) và mối quan hệ của nó với biến mục tiêu (`Class`). Nó xác nhận rằng `Amount` là một đặc trưng có khả năng phân biệt giữa hai lớp, dù có nhiều outlier trong lớp không gian lận. Việc hiểu phân bố này giúp chúng ta quyết định các bước tiền xử lý phù hợp cho biến `Amount` (như chuẩn hóa) và giải thích tại sao các mô hình có thể gặp khó khăn trong việc phân loại các giao dịch gian lận có số tiền lớn (nếu có).

# %%
df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df['normTime'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1,1))
df = df.drop(['Amount','Time'], axis=1)

# %% [markdown]
# why do i need to do this

# %%
num_cols = df.select_dtypes(include=['number']).columns.tolist()
num_cols = [col for col in num_cols if col != tar_col]
num_cols = np.array(num_cols)

# %% [markdown]
# also why

# %%
df

# %%
plt.figure(figsize = (8, 6))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()

# %%
# 1) Tách feature và target
X = df.drop(tar_col, axis=1)
y = df[tar_col]

# 2) Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% dùng để test
    stratify=y,           # giữ nguyên tỷ lệ lớp (rất quan trọng khi imbalanced)
    random_state=42       # để kết quả có thể tái lập
)

# 3) Kiểm tra kích thước và phân bố
print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:",  X_test.shape,  y_test.shape)
print("Train class ratio:\n", y_train.value_counts(normalize=True))
print("Test  class ratio:\n", y_test.value_counts(normalize=True))

# %%
from imblearn.pipeline import Pipeline  #SMOTE nhẹ hơn
pipeline = Pipeline(steps=[
    ('scaler', RobustScaler()),               # scale dữ liệu trước
    ('smote', SMOTE(random_state=42 ,sampling_strategy = 0.3)),        # oversample chỉ trên train, chỉ SMOTE 30% thôi
    ('model', LogisticRegression(max_iter=1000, class_weight=None))
])

# %%
pipeline.fit(X_train, y_train)

# %%
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]

# %%
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_proba))

# %% [markdown]
# ### Confusion Matrix for Logistic Regression

# %%
cm_lr = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 3))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# %% [markdown]
# Model dự đoán 56254 giao dịch là bình thường (và thật sự bình thường)
# Có 397 giao dịch hợp lệ nhưng bị model nhầm là gian lận
# Có 14 giao dịch gian lận nhưng model lại tưởng là bình thường Có 81 giao dịch gian lận được model phát hiện đúng”

# %%
model_lr = pipeline.named_steps['model']
coef = model_lr.coef_[0]

tmp = pd.DataFrame({
    'Feature': X.columns,
    'Feature importance': np.abs(coef)
}).sort_values(by='Feature importance', ascending=False)

# Vẽ biểu đồ giống RF
plt.figure(figsize=(7, 4))
plt.title('Feature Importance (Logistic Regression)', fontsize=14)
s = sns.barplot(x='Feature', y='Feature importance', data=tmp, color='steelblue')
s.set_xticklabels(s.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()

# %%
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek # Loại bỏ các điểm gây nhiễu (noise) và làm rõ ranh giới giữa hai lớp.
pipe_rf = Pipeline(steps=[
    ('scaler', RobustScaler()),
    ('resample', SMOTETomek(
        sampling_strategy=0.3,
        random_state=42)),
    ('model', RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# %%
pipe_rf.fit(X_train, y_train)

# %%
y_pred_rf = pipe_rf.predict(X_test)
y_proba_rf = pipe_rf.predict_proba(X_test)[:,1]

# %%
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest ROC AUC:", roc_auc_score(y_test, y_proba_rf))
print("Báo cáo phân loại Random Forest:\n", classification_report(y_test, y_pred_rf))

# %% [markdown]
# ### Confusion Matrix for Random Forest

# %%
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 3)) # Adjusted figsize
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()

# %%
model_rf = pipe_rf.named_steps['model']
tmp_rf = pd.DataFrame({
    'Feature': X.columns,
    'Feature importance': model_rf.feature_importances_
}).sort_values(by='Feature importance', ascending=False)

plt.figure(figsize=(7, 4))
plt.title('Feature Importance (Random Forest)', fontsize=14)
sns.barplot(x='Feature', y='Feature importance', data=tmp_rf, color='forestgreen')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %%
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# %%
pipe_xgb = Pipeline(steps=[
    ('scaler', RobustScaler()),             # scale dữ liệu
    ('smote', SMOTE(random_state=42)),      # xử lý imbalance
    ('model', XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
    ))
])


# %%
pipe_xgb.fit(X_train, y_train)

# %%
y_pred_xgb = pipe_xgb.predict(X_test)
y_proba_xgb = pipe_xgb.predict_proba(X_test)[:, 1]

# %%
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost ROC AUC:", roc_auc_score(y_test, y_proba_xgb))
print(classification_report(y_test, y_pred_xgb))

# %% [markdown]
# # Trade off giữa việc SMOTE và không SMOTE ở mô hình này là
# XGBoost Accuracy: 0.9993303492757198
# XGBoost ROC AUC: 0.9220077872922762
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00     56651
#            1       0.87      0.71      0.78        95
# 
#     accuracy                           1.00     56746
#    macro avg       0.93      0.85      0.89     56746
# weighted avg       1.00      1.00      1.00     56746

# %% [markdown]
# Kết quả thực nghiệm cho thấy việc áp dụng SMOTE mang lại sự trade-off rõ ràng giữa độ chính xác tổng thể và khả năng phát hiện giao dịch gian lận. Cụ thể, khi sử dụng SMOTE, mô hình XGBoost đạt recall và AUC cao hơn, thể hiện khả năng nhận diện tốt hơn lớp thiểu số, dù accuracy giảm nhẹ do mô hình hy sinh một phần độ chính xác ở lớp đa số. Điều này cho thấy SMOTE giúp mô hình bớt thiên lệch về lớp đa số và tăng tính công bằng trong phân loại, đặc biệt phù hợp cho các bài toán nhạy cảm như phát hiện gian lận thẻ tín dụng.

# %% [markdown]
# ### Confusion Matrix for XGBoost

# %%
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(5, 3)) # Adjusted figsize
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost')
plt.show()

# %%
model_xgb = pipe_xgb.named_steps['model']
tmp_xgb = pd.DataFrame({
    'Feature': X.columns,
    'Feature importance': model_xgb.feature_importances_
}).sort_values(by='Feature importance', ascending=False)

plt.figure(figsize=(7, 4))
plt.title('Feature Importance (XGBoost)', fontsize=14)
sns.barplot(x='Feature', y='Feature importance', data=tmp_xgb, color='darkorange')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Model Evaluation

# %%
# Get ROC curve data for Logistic Regression
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_proba)

# Get ROC curve data for Random Forest
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_proba_rf)

# Get ROC curve data for XGBoost
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_proba_xgb)

plt.figure(figsize=(6, 4)) # Adjusted figsize

# Plot ROC curves
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_proba):.4f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_proba_rf):.4f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, y_proba_xgb):.4f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Models')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Insight

# %%
from sklearn.metrics import precision_recall_curve, auc

# Get Precision-Recall curve data for Logistic Regression
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_proba)
auc_lr_pr = auc(recall_lr, precision_lr)

# Get Precision-Recall curve data for Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_proba_rf)
auc_rf_pr = auc(recall_rf, precision_rf)

# Get Precision-Recall curve data for XGBoost
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_proba_xgb)
auc_xgb_pr = auc(recall_xgb, precision_xgb)

plt.figure(figsize=(6, 4)) # Adjusted figsize

plt.plot(recall_lr, precision_lr, label=f'Logistic Regression (AUC = {auc_lr_pr:.4f})')
plt.plot(recall_rf, precision_rf, label=f'Random Forest (AUC = {auc_rf_pr:.4f})')
plt.plot(recall_xgb, precision_xgb, label=f'XGBoost (AUC = {auc_xgb_pr:.4f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison of Models')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Insight


