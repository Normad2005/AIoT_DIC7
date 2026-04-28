import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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
    "📊 1-3. Understanding & Preparation", 
    "🤖 4-5. Modeling & Evaluation", 
    "🚀 6. Deployment (Visualization)"
])

# Phase 1-3: Data Preview & Preparation
with tabs[0]:
    st.header("Phases 1-3: Data Understanding & Preparation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Synthetic Data Preview")
        df = pd.DataFrame(np.hstack([X, y]), columns=['Feature (X)', 'Target (y)'])
        st.dataframe(df.head(100), use_container_width=True, height=300)
    
    with col2:
        st.subheader("Dataset Statistics")
        st.write(df.describe())
        
    st.info("💡 **Preparation Note:** The data is automatically split into an 80/20 Training/Testing ratio for the modeling phase.")

# Phase 4-5: Modeling & Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with tabs[1]:
    st.header("Phases 4-5: Modeling & Evaluation")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Squared Error (MSE)", f"{mse:.2f}", delta_color="inverse")
    m2.metric("R² Score", f"{r2:.4f}")
    m3.metric("Samples", f"{n_samples}")
    
    st.divider()
    
    st.subheader("Model Performance Analysis")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Learned Coefficients")
        coeff_df = pd.DataFrame({
            "Parameter": ["Slope (m)", "Intercept (c)"],
            "True Value": [true_slope, true_intercept],
            "Learned Value": [round(model.coef_[0][0], 4), round(model.intercept_[0], 4)]
        })
        st.table(coeff_df)
        
    with c2:
        st.markdown("#### Regression Equation")
        st.code(f"y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}", language="text")
        if r2 > 0.95:
            st.success("The model has captured the underlying pattern with high precision!")
        elif r2 > 0.8:
            st.warning("The model shows a strong trend but is affected by significant noise.")
        else:
            st.error("Low R² score. The noise levels are potentially dominating the underlying signal.")

# Phase 6: Deployment (Visualization)
with tabs[2]:
    st.header("Phase 6: Deployment (Visualization)")
    st.write("This interactive chart allows you to explore the data points and the regression model's fit.")
    
    # Prepare data for Altair
    plot_df = df.copy()
    
    # Create the scatter plot
    points = alt.Chart(plot_df).mark_circle(size=60, opacity=0.5, color='#3498db').encode(
        x=alt.X('Feature (X)', title='Feature (X)'),
        y=alt.Y('Target (y)', title='Target (y)'),
        tooltip=['Feature (X)', 'Target (y)']
    ).interactive()
    
    # Create the regression line
    line_X = np.linspace(df['Feature (X)'].min(), df['Feature (X)'].max(), 100).reshape(-1, 1)
    line_y = model.predict(line_X)
    line_df = pd.DataFrame({
        'Feature (X)': line_X.flatten(),
        'Target (y)': line_y.flatten()
    })
    
    line = alt.Chart(line_df).mark_line(color='#e74c3c', size=3).encode(
        x='Feature (X)',
        y='Target (y)'
    )
    
    # Combine layers
    chart = (points + line).properties(
        width='container',
        height=500
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    st.markdown("""
    ### 💡 如何互動？
    - **縮放/平移**：在圖表上捲動或拖曳。
    - **查看數值**：將滑鼠懸停在藍色數據點上。
    - **即時更新**：調整左側邊欄的參數，圖表會自動重繪。
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity CRISP-DM Toolkit v1.0")
