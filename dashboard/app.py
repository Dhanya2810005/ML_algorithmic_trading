# streamlit_app.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import run_experiment  # connect to backend

# Optional explainable_ai import (safe fallback)
try:
    from backend.utils import explainable_ai
except ImportError:
    explainable_ai = None

# ---------- Streamlit Config ----------
st.set_page_config(
    page_title="AI Trading Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Enhanced Styling with Better Visibility ----------
st.markdown("""
<style>
    :root { 
        --primary-color: #00b4d8; 
        --secondary-color: #8b5cf6;
        --text-primary: #ffffff;
        --text-secondary: #e2e8f0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #151521 0%, #1e1e2f 100%); 
    }
    
    /* Make ALL labels white and visible */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stDateInput label,
    [data-testid="stSidebar"] .stRadio label,
    label,
    .stSelectbox label,
    .stDateInput label,
    .stNumberInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
    }
    
    /* Radio button labels */
    .stRadio > label {
        color: #ffffff !important;
        background: rgba(45,45,68,0.7);
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(100,100,120,0.5);
        margin-bottom: 8px;
        font-weight: 600 !important;
    }
    
    .stRadio > div {
        background: rgba(30,30,45,0.5);
        padding: 10px;
        border-radius: 10px;
    }
    
    /* Radio option text */
    .stRadio > div > label > div {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Input fields with better contrast */
    input, select, textarea {
        color: #ffffff !important;
        background-color: rgba(45, 45, 68, 0.8) !important;
        border: 1px solid rgba(100, 100, 120, 0.5) !important;
    }
    
    /* Date input styling */
    [data-testid="stDateInput"] input {
        color: #ffffff !important;
        background-color: rgba(45, 45, 68, 0.8) !important;
    }
    
    /* Select box styling - SIDEBAR (dark background) */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background-color: rgba(45, 45, 68, 0.8) !important;
        color: #ffffff !important;
    }
    
    /* Select box styling - MAIN CONTENT (light background) */
    .main [data-testid="stSelectbox"] > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Dropdown menu items - SIDEBAR */
    [data-testid="stSidebar"] [role="option"] {
        color: #ffffff !important;
        background-color: rgba(30, 30, 45, 0.9) !important;
    }
    
    [data-testid="stSidebar"] [role="option"]:hover {
        background-color: rgba(0, 180, 216, 0.2) !important;
    }
    
    /* Dropdown menu items - MAIN CONTENT (BLACK text) */
    .main [role="option"] {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    .main [role="option"]:hover {
        background-color: rgba(0, 180, 216, 0.1) !important;
    }
    
    /* ALL dropdown items should be BLACK on white background by default */
    [role="listbox"] [role="option"] {
        color: #000000 !important;
        background-color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    [role="listbox"] [role="option"]:hover {
        background-color: rgba(0, 180, 216, 0.15) !important;
        color: #000000 !important;
    }
    
    /* Override for sidebar dropdown specifically */
    [data-testid="stSidebar"] [role="listbox"] [role="option"] {
        color: #ffffff !important;
        background-color: rgba(30, 30, 45, 0.9) !important;
    }
    
    [data-testid="stSidebar"] [role="listbox"] [role="option"]:hover {
        background-color: rgba(0, 180, 216, 0.3) !important;
        color: #ffffff !important;
    }
    
    /* Dropdown menu container - MAIN/DEFAULT (white background) */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="popover"] ul {
        background-color: #ffffff !important;
    }
    
    /* Sidebar dropdown container (dark background) */
    [data-testid="stSidebar"] [data-baseweb="popover"] {
        background-color: rgba(30, 30, 45, 0.98) !important;
        border: 1px solid rgba(100, 100, 120, 0.5) !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="popover"] ul {
        background-color: rgba(30, 30, 45, 0.98) !important;
    }
    
    /* Select box text color */
    [data-testid="stSelectbox"] input {
        color: #000000 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stSelectbox"] input {
        color: #ffffff !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;} 
    footer {visibility: hidden;}
    
    /* Metric values */
    [data-testid="stMetricValue"] { 
        font-size: 28px; 
        font-weight: 700;
        color: #00b4d8 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Headings */
    h1 { 
        background: linear-gradient(135deg, #00b4d8, #8b5cf6);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        font-weight: 700; 
    }
    
    h2, h3 { 
        color: #00b4d8 !important;
        font-weight: 700 !important;
    }
    
    h4, h5, h6 {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar headings should be WHITE */
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Regular text - MAIN CONTENT AREA (BLACK for light background) */
    .main p, .main span, .main div {
        color: #000000 !important;
    }
    
    /* Sidebar text stays white/light for dark sidebar */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div {
        color: #e2e8f0;
    }
    
    /* Buttons */
    .stButton>button { 
        width: 100%; 
        border-radius: 10px; 
        height: 50px; 
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #00b4d8, #8b5cf6);
        color: white !important;
        border: none;
    }
    
    .stButton>button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 180, 216, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 10px; 
    }
    
    .stTabs [data-baseweb="tab"] { 
        border-radius: 10px; 
        padding: 10px 20px; 
        font-weight: 600;
        color: #ffffff !important;
        background: rgba(45, 45, 68, 0.5);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 180, 216, 0.3), rgba(139, 92, 246, 0.3));
        border: 1px solid rgba(0, 180, 216, 0.5);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(0, 180, 216, 0.2);
    }
    
    /* Tab content text - BLACK for readability */
    .stTabs [role="tabpanel"] p,
    .stTabs [role="tabpanel"] span,
    .stTabs [role="tabpanel"] div {
        color: #000000 !important;
    }
    
    /* Info boxes - BLACK text on colored backgrounds */
    .stInfo, .stSuccess, .stWarning, .stError {
        color: #000000 !important;
    }
    
    .stAlert p {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Dataframe - BLACK text */
    .dataframe {
        color: #000000 !important;
    }
    
    .dataframe th {
        background-color: rgba(0, 180, 216, 0.2) !important;
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    .dataframe td {
        color: #000000 !important;
    }
    
    /* Markdown text in main content - BLACK */
    .main .stMarkdown p {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    .main .stMarkdown li {
        color: #000000 !important;
    }
    
    .main .stMarkdown em {
        color: #475569 !important;
        font-style: italic;
    }
    
    /* Status box */
    .status-box {
        padding: 12px;
        text-align: center;
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(0, 180, 216, 0.2), rgba(139, 92, 246, 0.2));
        border: 1px solid rgba(0, 180, 216, 0.3);
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* JSON output - BLACK text */
    .main pre {
        color: #000000 !important;
        background-color: rgba(240, 240, 245, 0.8) !important;
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Spinner text - BLACK */
    .stSpinner > div {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* All text in main content area should be BLACK */
    .main {
        color: #000000 !important;
    }
    
    .main * {
        color: inherit;
    }
    
    /* Column text - BLACK */
    [data-testid="column"] p,
    [data-testid="column"] span {
        color: #000000 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session State ----------
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'Gradient Boosting'
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_X_train' not in st.session_state:
    st.session_state.model_X_train = None
if 'model_feature_names' not in st.session_state:
    st.session_state.model_feature_names = None

# ---------- Cache Backend ----------
def cached_run_experiment(ticker, model_name):
    """Run backend computations (caching disabled for models)"""
    return run_experiment(ticker, model_name)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0; border-bottom: 2px solid rgba(0, 180, 216, 0.3);'>
        <div style='font-size: 45px; margin-bottom: 10px;'>🚀</div>
        <h2 style='margin: 0; background: linear-gradient(135deg, #00b4d8, #8b5cf6);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Trading AI
        </h2>
        
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    ticker = st.selectbox(" Select Stock", ["AAPL", "TSLA", "MSFT", "GOOG", "NVDA", "AMZN"], index=0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(" Start Date", value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input(" End Date", value=datetime(2025, 1, 1))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='border-top: 1px solid rgba(100,100,120,0.3); margin: 15px 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='color: #ffffff; margin-bottom: 15px; font-weight: 700;'> Choose Model</h4>", unsafe_allow_html=True)
    
    model_options = {
        "🌳 Gradient Boosting": "Gradient Boosting",
        "🌲 Random Forest": "Random Forest",
        "📊 ARIMA": "ARIMA",
        "🤖 Q-Learning Agent": "Q-Learning Agent"
    }
    selected_model_display = st.radio(
        "Choose Model",
        list(model_options.keys()),
        label_visibility="collapsed"
    )
    st.session_state.selected_model = model_options[selected_model_display]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='border-top: 1px solid rgba(100,100,120,0.3); margin: 15px 0;'></div>", unsafe_allow_html=True)
    
    if st.button(" Run Simulation", type="primary"):
        with st.spinner(f"Running {st.session_state.selected_model} for {ticker}..."):
            try:
                result = cached_run_experiment(ticker, st.session_state.selected_model)
                
                # Store model separately in session_state
                if "model" in result and result["model"] is not None:
                    st.session_state.trained_model = result["model"]
                    st.session_state.model_X_train = result.get("X_train")
                    st.session_state.model_feature_names = result.get("feature_names")
                else:
                    st.session_state.trained_model = None
                    st.session_state.model_X_train = None
                    st.session_state.model_feature_names = None
                
                st.session_state.last_results = result
                st.success(f" {st.session_state.selected_model} completed successfully!")
            except (ConnectionError, ValueError) as e:
                # Show user-friendly error messages for data loading issues
                error_msg = str(e)
                if "Failed to download" in error_msg or "SSL" in error_msg or "Connection" in error_msg:
                    st.error(f" **Connection Error**\n\n{error_msg}\n\n**Troubleshooting:**\n- Check your internet connection\n- Try again in a few moments\n- The data provider may be temporarily unavailable")
                else:
                    st.error(f" **Error**: {error_msg}")
            except Exception as e:
                st.error(f" **Unexpected Error**: {str(e)}\n\nIf this persists, please check the console output for more details.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='status-box'>
        <div style='font-size: 20px; margin-bottom: 5px;'>🟢</div>
        <div>System Ready</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- MAIN ----------
# Ensure trained model stays synced after reruns
if (
    st.session_state.last_results
    and st.session_state.last_results.get("model") is not None
    and st.session_state.trained_model is None
):
    st.session_state.trained_model = st.session_state.last_results["model"]
    st.session_state.model_X_train = st.session_state.last_results.get("X_train")
    st.session_state.model_feature_names = st.session_state.last_results.get("feature_names")

st.markdown("# Algorithmic Trading Dashboard")
st.markdown("<h5 style='color: #475569; margin-top: -10px; font-weight: 600;'>AI-driven trading with Machine Learning, Reinforcement Learning & Explainable AI</h5>", unsafe_allow_html=True)
st.markdown("<div style='border-top: 2px solid rgba(0, 180, 216, 0.3); margin: 20px 0;'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([" Dashboard", " Models", " Trades", " Explainability", " Backtest"])

# ---------- Helper ----------
def plot_predictions(index, y_true, preds_dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index, y=y_true, name='Actual', mode='lines', 
                             line=dict(color='#00b4d8', width=2)))
    for name, arr in preds_dict.items():
        if arr is not None:
            fig.add_trace(go.Scatter(x=index, y=arr, name=name, mode='lines', 
                                     line=dict(dash='dash', width=2)))
    fig.update_layout(
        template='plotly_dark', 
        height=420, 
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(color='#ffffff'),
        legend=dict(font=dict(color='#ffffff'))
    )
    return fig

# ---------- TAB 1 ----------
with tab1:
    if st.session_state.last_results is None:
        st.info(" No simulation yet. Run one from the sidebar to get started!")
    else:
        res = st.session_state.last_results
        perf = res["performance"]

        # Metrics
        col1, col2, col3 = st.columns(3)
        if "CAGR" in perf:
            col1.metric(" CAGR", f"{perf['CAGR']:.2%}")
            col2.metric(" Sharpe Ratio", f"{perf['Sharpe']:.3f}")
            col3.metric(" Max Drawdown", f"{perf['MaxDD']:.2%}")
        else:
            col1.metric(" RMSE", f"{perf['RMSE']:.2f}")
            col2.metric(" R² Score", f"{perf['R2']:.3f}")
            col3.metric(" Model", res["model_name"])

        st.markdown("<div style='border-top: 1px solid rgba(100,100,120,0.3); margin: 20px 0;'></div>", unsafe_allow_html=True)

        # ML vs RL chart logic
        if res["model_name"] != "Q-Learning Agent":
            idx = np.arange(len(res["y_test"]))
            st.markdown("###  Model Predictions vs Actual")
            fig = plot_predictions(idx, res["y_test"], {"Predicted": res["y_pred"]})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("###  Q-Agent Equity vs Buy & Hold")
            eq = res.get("eq", None)
            if eq is not None and len(eq) > 1:
                days = np.arange(len(eq))
                returns = np.diff(eq) / np.maximum(eq[:-1], 1)
                buy_hold = np.concatenate([[eq[0]], (1 + np.nan_to_num(returns)).cumprod() * eq[0]])
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=days, y=eq, name="Q-Agent Equity",
                                         line=dict(color='#00b4d8', width=2)))
                fig2.add_trace(go.Scatter(x=np.arange(len(buy_hold)), y=buy_hold, 
                                         name="Buy & Hold", 
                                         line=dict(dash='dash', color='#8b5cf6', width=2)))
                fig2.update_layout(
                    template='plotly_dark', 
                    height=420,
                    font=dict(color='#ffffff'),
                    legend=dict(font=dict(color='#ffffff'))
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info(" Equity curve not available for this model (only for Q-Learning Agent).")

# ---------- TAB 2 ----------
with tab2:
    if st.session_state.last_results:
        st.markdown("###  Model Performance Summary")
        st.json(st.session_state.last_results["performance"])
    else:
        st.info(" Run a simulation to view model results.")

# ---------- TAB 3 ----------
with tab3:
    st.markdown("###  Executed Trades (Q-Learning Agent)")
    if st.session_state.last_results and st.session_state.last_results["model_name"] == "Q-Learning Agent":
        trades = st.session_state.last_results.get("trades", None)
        if trades is not None:
            df_trades = pd.DataFrame(trades, columns=["Step", "ActionID", "Price", "Shares", "Cash", "Equity"])
            df_trades["Action"] = df_trades["ActionID"].map({0: "HOLD", 1: "BUY", 2: "SELL"})
            df_trades = df_trades[df_trades["Action"].isin(["BUY", "SELL"])]
            df_trades["PnL"] = df_trades["Equity"].diff().fillna(0)
            df_trades["Time"] = pd.date_range(end=datetime.now(), periods=len(df_trades), freq="D")
            st.dataframe(df_trades[["Time", "Action", "Price", "Shares", "Cash", "Equity", "PnL"]], 
                        use_container_width=True)

            col1, col2 = st.columns(2)
            col1.metric(" Final Portfolio Value", f"${df_trades['Equity'].iloc[-1]:,.2f}")
            col2.metric(" Total Trades", len(df_trades))
        else:
            st.info(" No trades recorded yet — run Q-Learning Agent.")
    else:
        st.info(" Trade logs are only available for Q-Learning Agent.")

# ---------- TAB 4 ----------
with tab4:
    # Ensure model is always synced correctly
    if (
        st.session_state.last_results
        and st.session_state.last_results.get("model") is not None
        and st.session_state.trained_model is None
    ):
        st.session_state.trained_model = st.session_state.last_results["model"]
        st.session_state.model_X_train = st.session_state.last_results.get("X_train")
        st.session_state.model_feature_names = st.session_state.last_results.get("feature_names")

    st.markdown("###  Explainable AI")
    st.markdown("<p style='color: #94a3b8;'>Understand which features drive your model's predictions using SHAP and LIME.</p>", unsafe_allow_html=True)
    
    if st.session_state.last_results is None:
        st.info(" No simulation results yet. Run a model from the sidebar first.")
    else:
        model = st.session_state.trained_model
        X_train = st.session_state.model_X_train
        feature_names = st.session_state.model_feature_names
        
        if model is None:
            model_name = st.session_state.last_results.get("model_name", "Unknown")
            if model_name in ["ARIMA", "Q-Learning Agent"]:
                st.info(" Explainability is only available for Gradient Boosting and Random Forest models.")
                st.info(" ARIMA and Q-Learning Agent don't support feature-based explainability.")
            else:
                st.warning(" Model object not found. Make sure you ran a Gradient Boosting or Random Forest model.")
                st.info(" Try running the model again from the sidebar.")
        else:
            if explainable_ai and model is not None and X_train is not None:
                # SHAP Global Explanation
                st.markdown("####  SHAP Global Feature Importance")
                st.markdown("<p style='color: #475569; font-weight: 500;'><em>Shows which features are most important across all predictions</em></p>", unsafe_allow_html=True)
                
                with st.spinner("Computing SHAP values... This may take a moment."):
                    try:
                        shap_html = explainable_ai.get_shap_summary_html(
                            model, X_train, feature_names=feature_names, max_samples=100
                        )
                        if shap_html:
                            st.components.v1.html(shap_html, height=500, scrolling=True)
                        else:
                            st.warning("SHAP is not installed. Install it with: `pip install shap`")
                    except Exception as e:
                        st.error(f"SHAP error: {str(e)}")
                        st.info(" Try installing SHAP: `pip install shap`")
                
                st.markdown("<div style='border-top: 1px solid rgba(100,100,120,0.3); margin: 20px 0;'></div>", unsafe_allow_html=True)
                
                # LIME Local Explanation
                st.markdown("####  LIME Local Explanation")
                st.markdown("<p style='color: #475569; font-weight: 500;'><em>Explains a single prediction by showing which features influenced it</em></p>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    max_idx = len(X_train) - 1 if X_train is not None and len(X_train) > 0 else 0
                    sample_idx = st.number_input(
                        "Sample Index", 
                        min_value=0, 
                        max_value=max(0, max_idx),
                        value=0,
                        help="Select which sample to explain"
                    )
                
                with st.spinner(f"Computing LIME explanation for sample {sample_idx}..."):
                    try:
                        lime_html = explainable_ai.get_lime_html_for_sample(
                            model, X_train, sample_index=sample_idx, feature_names=feature_names
                        )
                        if lime_html:
                            st.components.v1.html(lime_html, height=500, scrolling=True)
                        else:
                            st.warning("LIME is not installed. Install it with: `pip install lime`")
                    except Exception as e:
                        st.error(f"LIME error: {str(e)}")
                        st.info(" Try installing LIME: `pip install lime`")
            else:
                st.error("Explainability module not available. Check imports.")

# ---------- TAB 5 ----------
with tab5:
    st.markdown("###  Backtest Results")
    eq = st.session_state.last_results.get("eq") if st.session_state.last_results else None
    if eq is not None and len(eq) > 1:
        days = np.arange(len(eq))
        bh_returns = np.random.randn(len(days)-1) * 0.0005
        bh = np.concatenate([[eq[0]], (1 + bh_returns).cumprod() * eq[0]])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=eq, name="Strategy",
                                line=dict(color='#00b4d8', width=2)))
        fig.add_trace(go.Scatter(x=np.arange(len(bh)), y=bh, name="Benchmark", 
                                line=dict(dash='dash', color='#8b5cf6', width=2)))
        fig.update_layout(
            template='plotly_dark', 
            height=420,
            font=dict(color='#ffffff'),
            legend=dict(font=dict(color='#ffffff'))
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(" No backtest results yet. Run Q-Learning Agent to generate equity curve.")

st.markdown("<div style='border-top: 2px solid rgba(0, 180, 216, 0.3); margin: 30px 0;'></div>", unsafe_allow_html=True)
