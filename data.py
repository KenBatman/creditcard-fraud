import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import streamlit as st  # Để render plots

def load_data(file_path):
    """
    Load CSV file — hỗ trợ cả file local và file upload từ Streamlit.
    Trả về: df (DataFrame) và info (thông tin thống kê, duplicates, missing, outliers, constant cols)
    """
    # ✅ Đọc file linh hoạt: hỗ trợ cả file upload và đường dẫn local
    if hasattr(file_path, "read"):  # Nếu là file được upload qua Streamlit
        df = pd.read_csv(file_path)
    else:  # Nếu là đường dẫn file local
        df = pd.read_csv(str(file_path))

    # Basic info
    shape_original = df.shape
    
    # Set target column & numeric columns
    tar_col = "Class"
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    num_cols = [col for col in num_cols if col != tar_col]
 
    # Check & xóa duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
    
    # Check missing
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count() * 100).sort_values(ascending=False)
    missing_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    # Check outliers IQR (trên num_cols gốc)
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).sum()
    
    # Check constant cols
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    
    # Info dict
    info = {
        'shape_original': shape_original,
        'shape_final': df.shape,
        'duplicates_removed': duplicate_count,
        'missing_df': missing_df,
        'outliers': outliers_iqr[outliers_iqr > 0],
        'constant_cols': constant_cols if constant_cols else []
    }
    
    return df, info

def plot_class_distribution(df):
    """Bar chart cho class distribution."""
    classes = df['Class'].value_counts()
    normal_share = classes[0] / len(df) * 100
    fraud_share = classes[1] / len(df) * 100
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Non-Fraud', 'Fraud'], classes, color=['#17a2b8', '#c2185b'])
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of transactions')
    ax.annotate(f'{classes[0]}\n({normal_share:.4}%)', (0.2, 0.45), xycoords='axes fraction', ha='center')
    ax.annotate(f'{classes[1]}\n({fraud_share:.4}%)', (0.7, 0.45), xycoords='axes fraction', ha='center')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_time_density(df):
    """Time density distplot (dùng 'Time' gốc)."""
    if 'Time' not in df.columns:
        st.error("Cột 'Time' không tồn tại!")
        return
    
    class_0 = df.loc[df['Class'] == 0, 'Time']
    class_1 = df.loc[df['Class'] == 1, 'Time']
    
    hist_data = [class_0, class_1]
    group_labels = ['Non-Fraud', 'Fraud']
    
    fig = ff.create_distplot(
        hist_data, group_labels, show_hist=False, show_rug=False, colors=['#1f77b4', '#d62728']
    )
    
    fig.update_layout(
        title=dict(text='Credit Card Transactions - Time Density Distribution', x=0.5, font=dict(size=20, family='Arial', color='#333')),
        xaxis=dict(title='Transaction Time (seconds)', gridcolor='rgba(200,200,200,0.3)', zeroline=False),
        yaxis=dict(title='Density', gridcolor='rgba(200,200,200,0.3)', zeroline=False),
        plot_bgcolor='white',
        legend=dict(title='Transaction Type', orientation='h', yanchor='bottom', y=0.98, xanchor='center', x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_boxplots(df):
    """Boxplots cho selected V cols và Amount (gốc)."""
    selected_v_cols = ['V14', 'V17', 'V10', 'V12']
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, col in enumerate(selected_v_cols):
        row, col_idx = divmod(i, 2)
        sns.boxplot(x='Class', y=col, data=df, palette='viridis', ax=axs[row, col_idx])
        axs[row, col_idx].set_title(f'Distribution of {col} by Class')
    # Thêm subplot cho Amount nếu muốn (tùy chọn)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_correlation(df):
    """Correlation heatmap (toàn bộ df)."""
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=0.1, cmap="Reds", ax=ax)
    ax.set_title('Credit Card Transactions features correlation plot (Pearson)')
    st.pyplot(fig)
    plt.close(fig)