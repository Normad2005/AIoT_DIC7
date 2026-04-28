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
    /* Main background tweak */
    .main {
        background-color: #0e1117;
    }
    
    /* Premium Metric Card */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    /* Tab active state */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #94a3b8;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🚀 Linear Regression: CRISP-DM Dashboard")
st.caption("一個結合數據科學標準流程與即時互動技術的專業分析工具")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("⚙️ 參數配置")
    st.markdown("---")
    
    with st.container():
        st.subheader("📊 數據生成")
        n_samples = st.slider("樣本點總數", min_value=100, max_value=5000, value=1000, step=100)
        noise_std = st.slider("噪聲強度 (σ)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    
    st.markdown("---")
    
    with st.container():
        st.subheader("📐 方程式係數")
        true_slope = st.slider("預期斜率 (a)", min_value=-50.0, max_value=50.0, value=10.0, step=0.1)
        true_intercept = st.slider("預期截距 (b)", min_value=-100.0, max_value=100.0, value=30.0, step=1.0)

    st.markdown("---")
    st.info("💡 調整滑桿即可觸發即時重新建模與視覺化。")

# --- Data Generation Logic ---
np.random.seed(42)
X = np.random.uniform(-10, 10, n_samples).reshape(-1, 1)
noise = np.random.normal(0, noise_std, n_samples).reshape(-1, 1)
y = true_slope * X + true_intercept + noise

# --- Modeling Logic ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- CRISP-DM Tabs ---
tabs = st.tabs([
    "🎯 模型成果與視覺化", 
    "📁 數據深度分析"
])

# --- Tab 1: Integrated Evaluation & Visualization ---
with tabs[0]:
    st.markdown("### 📈 核心指標回報")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("均方誤差 (MSE)", f"{mse:.2f}", delta_color="inverse")
    with m2:
        st.metric("決定係數 (R²)", f"{r2:.4f}")
    with m3:
        st.metric("有效樣本", f"{len(X_train)} pts")
    
    st.markdown("---")
    
    col_plot, col_info = st.columns([7, 3])
    
    with col_plot:
        st.markdown("#### 🔍 互動式擬合分析")
        plot_df = pd.DataFrame(np.hstack([X, y]), columns=['X', 'y'])
        
        # Altair chart with enhanced aesthetics
        points = alt.Chart(plot_df).mark_circle(size=50, opacity=0.4, color='#38bdf8').encode(
            x=alt.X('X', title='自變量 (Feature X)', scale=alt.Scale(domain=[-11, 11])),
            y=alt.Y('y', title='因變量 (Target Y)'),
            tooltip=[alt.Tooltip('X', title='X值'), alt.Tooltip('y', title='Y值')]
        )
        
        line_X = np.linspace(-10, 10, 100).reshape(-1, 1)
        line_y = model.predict(line_X)
        line_df = pd.DataFrame({'X': line_X.flatten(), 'y': line_y.flatten()})
        
        line = alt.Chart(line_df).mark_line(color='#fb7185', strokeWidth=3).encode(
            x='X', y='y'
        )
        
        final_chart = (points + line).configure_axis(
            gridColor='#334155',
            labelColor='#94a3b8',
            titleColor='#f1f5f9'
        ).configure_view(
            strokeWidth=0
        ).properties(
            width='container',
            height=450
        ).interactive()
        
        st.altair_chart(final_chart, use_container_width=True)

    with col_info:
        st.markdown("#### 🧠 學習成果")
        with st.container():
            st.markdown("**估計方程式：**")
            st.code(f"y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}", language="text")
            
            st.markdown("**參數對照：**")
            res_df = pd.DataFrame({
                "屬性": ["斜率 (a)", "截距 (b)"],
                "預測": [f"{model.coef_[0][0]:.3f}", f"{model.intercept_[0]:.3f}"],
                "設定": [f"{true_slope:.1f}", f"{true_intercept:.1f}"]
            })
            st.table(res_df)
            
            if r2 > 0.9:
                st.success("✅ 模型展現極佳的解釋能力")
            elif r2 > 0.6:
                st.warning("⚠️ 數據包含顯著噪聲")
            else:
                st.error("❌ 噪聲水平過高，關係不明確")

# --- Tab 2: Data Understanding ---
with tabs[1]:
    st.markdown("### 🗂️ 數據理解與分佈")
    d1, d2 = st.columns([6, 4])
    
    with d1:
        st.markdown("#### 數據採樣清單 (Top 100)")
        st.dataframe(pd.DataFrame(np.hstack([X, y]), columns=['Feature X', 'Target Y']).head(100), use_container_width=True)
    
    with d2:
        st.markdown("#### 描述性統計")
        st.write(pd.DataFrame(np.hstack([X, y]), columns=['X', 'y']).describe().T)
        
    with st.expander("ℹ️ 關於 CRISP-DM 流程"):
        st.markdown("""
        本專案嚴格遵循 **CRISP-DM** 標準流程：
        1. **業務理解**：定義線性回歸預測目標。
        2. **數據理解**：生成並探索模擬數據。
        3. **數據準備**：執行 Train-Test Split。
        4. **建模**：構建 Linear Regression 模型。
        5. **評估**：量化指標回報 (MSE, R²)。
        6. **部署**：透過本互動介面展示成果。
        """)

st.sidebar.caption("🚀 AIoT DIC7 Dashboard v2.1")
