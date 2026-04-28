import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import altair as alt

# --- Page Configuration ---
st.set_page_config(
    page_title="Linear Regression CRISP-DM Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("📈 Linear Regression: CRISP-DM Explorer")
st.markdown("""
This interactive application demonstrates a complete **CRISP-DM** (Cross-Industry Standard Process for Data Mining) 
workflow for a Linear Regression problem. Use the sidebar to manipulate the data generation process and see real-time updates.
""")

# --- Sidebar: Phase 1 & 2 - Business & Data Understanding ---
st.sidebar.header("🛠️ Data Generation Parameters")
st.sidebar.markdown("Define the parameters for the synthetic dataset.")

n_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=5000, value=1000, step=100)
noise_std = st.sidebar.slider("Noise Level (Std Dev)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("True Equation Settings")
true_slope = st.sidebar.slider("True Slope (a)", min_value=-50.0, max_value=50.0, value=10.0, step=0.1)
true_intercept = st.sidebar.slider("True Intercept (b)", min_value=-100.0, max_value=100.0, value=30.0, step=1.0)

# --- Data Generation Logic ---
np.random.seed(42)
X = np.random.uniform(-10, 10, n_samples).reshape(-1, 1)
noise = np.random.normal(0, noise_std, n_samples).reshape(-1, 1)
y = true_slope * X + true_intercept + noise

# --- CRISP-DM Tabs ---
tabs = st.tabs([
    "🚀 模型評估與視覺化 (預設)", 
    "📊 數據理解與準備"
])

# --- Modeling & Evaluation Logic (Shared) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- Tab 1: Integrated Evaluation & Visualization ---
with tabs[0]:
    st.header("線性回歸模型成果")
    
    # Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("均方誤差 (MSE)", f"{mse:.2f}", delta_color="inverse")
    m2.metric("決定係數 (R² Score)", f"{r2:.4f}")
    m3.metric("訓練樣本數", f"{len(X_train)}")
    
    st.divider()
    
    # Visualization and Equation Row
    col_plot, col_info = st.columns([2, 1])
    
    with col_plot:
        st.subheader("互動式擬合圖表")
        plot_df = pd.DataFrame(np.hstack([X, y]), columns=['Feature (X)', 'Target (y)'])
        
        points = alt.Chart(plot_df).mark_circle(size=60, opacity=0.5, color='#3498db').encode(
            x=alt.X('Feature (X)', title='特徵 (X)'),
            y=alt.Y('Target (y)', title='目標 (y)'),
            tooltip=['Feature (X)', 'Target (y)']
        ).interactive()
        
        line_X = np.linspace(plot_df['Feature (X)'].min(), plot_df['Feature (X)'].max(), 100).reshape(-1, 1)
        line_y = model.predict(line_X)
        line_df = pd.DataFrame({'Feature (X)': line_X.flatten(), 'Target (y)': line_y.flatten()})
        
        line = alt.Chart(line_df).mark_line(color='#e74c3c', size=3).encode(
            x='Feature (X)', y='Target (y)'
        )
        
        st.altair_chart((points + line).properties(width='container', height=450), use_container_width=True)

    with col_info:
        st.subheader("學習結果")
        st.code(f"y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}", language="text")
        
        coeff_df = pd.DataFrame({
            "參數": ["斜率 (a)", "截距 (b)"],
            "預測值": [round(model.coef_[0][0], 4), round(model.intercept_[0], 4)],
            "真實值": [true_slope, true_intercept]
        })
        st.table(coeff_df)
        
        if r2 > 0.95:
            st.success("模型完美擬合！")
        else:
            st.warning("受噪聲影響，存在偏差。")

# --- Tab 2: Data Understanding ---
with tabs[1]:
    st.header("數據詳情與準備")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("原始數據預覽")
        st.dataframe(pd.DataFrame(np.hstack([X, y]), columns=['X', 'y']).head(100), use_container_width=True)
    with col2:
        st.subheader("統計摘要")
        st.write(pd.DataFrame(np.hstack([X, y]), columns=['X', 'y']).describe())

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity CRISP-DM Toolkit v1.0")
