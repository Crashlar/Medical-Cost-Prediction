import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# --- PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(
    page_title="Medical Cost Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main page background */
    .main {
        background-color: #f5f5f5;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
    }
    /* Title styling */
    h1 {
        color: #1E3A8A; /* Dark Blue */
        text-align: center;
    }
    /* Header styling */
    h2, h3 {
        color: #1E3A8A;
    }
    /* Metric styling */
    .stMetric {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Dataframe styling */
    .stDataFrame {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_data
def load_assets():
    """Loads all necessary models, scaler, and data."""
    try:
        scaler = load('standard_scaler.joblib')
        ols_model = load('ols_model.joblib')
        ridge_model = load('ridge_model.joblib')
        feature_names = load('feature_names.joblib')
        data = pd.read_csv('insurance.csv')
        return scaler, ols_model, ridge_model, feature_names, data
    except FileNotFoundError as e:
        st.error(f"Missing required file: {e.filename}. Please run `run_pipeline.py` first.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

scaler, ols_model, ridge_model, feature_names, data = load_assets()

# --- SIDEBAR ---
st.sidebar.title("Navigation")
# Reordered page options: Cost Predictor is now the default/first page
page = st.sidebar.radio("Choose a page", ["Cost Predictor", "Data Explorer"])

st.sidebar.markdown("---")
st.sidebar.info(
    "This app compares OLS and Ridge regression for predicting medical costs, "
    "highlighting Ridge's stability with correlated features."
)

# --- PAGE 1: DATA EXPLORER ---
def page_data_explorer():
    st.title("Data Explorer")
    st.markdown("Explore the synthetic dataset used for training the models.")

    st.subheader("Raw Data")
    st.dataframe(data.head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Statistics")
        st.dataframe(data.describe())
    with col2:
        st.subheader("Correlation Matrix")
        corr_matrix = data[['BMI', 'Obesity_Score', 'Charges']].corr()
        st.dataframe(corr_matrix.style.background_gradient(cmap='Reds', axis=None))

    st.subheader("Visualizations")
    st.scatter_chart(data, x='BMI', y='Obesity_Score', color='#1E3A8A')
    st.markdown(
        "The scatter plot above clearly shows the strong linear relationship between BMI and Obesity Score, "
        "which is the source of multicollinearity in the dataset."
    )

# --- PAGE 2: COST PREDICTOR ---
def page_cost_predictor():
    st.title("Cost Predictor")
    st.markdown("Get real-time medical cost predictions from our models.")

    # --- Input Form ---
    with st.container():
        st.subheader("Patient Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 64, 35)
            children = st.slider("Children", 0, 5, 1)
        with col2:
            bmi = st.slider("BMI", 15.0, 50.0, 30.0, 0.1)
            smoker = st.selectbox("Smoker", ["No", "Yes"])
        with col3:
            obesity_score_noise = np.random.normal(0, 0.5)
            obesity_score = bmi * 1.5 + obesity_score_noise
            st.metric(label="Calculated Obesity Score", value=f"{obesity_score:.2f}")

    # --- Real-time Prediction ---
    smoker_binary = 1 if smoker == "Yes" else 0
    input_dict = {
        'Age': age,
        'BMI': bmi,
        'Obesity_Score': obesity_score,
        'Smoker': smoker_binary,
        'Children': children
    }
    input_df = pd.DataFrame([input_dict], columns=feature_names)
    scaled_input = scaler.transform(input_df)

    ols_pred = max(0, ols_model.predict(scaled_input)[0])
    ridge_pred = max(0, ridge_model.predict(scaled_input)[0])

    st.markdown("---")
    st.subheader("Predictions (in ₹ Rupees)")
    col1, col2 = st.columns(2)
    with col1:
        # Charges displayed in Rupees (₹)
        st.metric("OLS Prediction", f"₹{ols_pred:,.2f}")
    with col2:
        # Charges displayed in Rupees (₹)
        st.metric("Ridge Prediction", f"₹{ridge_pred:,.2f}")

    # --- Coefficient Comparison ---
    with st.expander("See Model Coefficients"):
        ols_coeffs = pd.Series(ols_model.coef_, index=feature_names)
        ridge_coeffs = pd.Series(ridge_model.coef_, index=feature_names)
        coeffs_df = pd.DataFrame({
            'OLS Coefficient': ols_coeffs,
            'Ridge Coefficient': ridge_coeffs
        })
        st.dataframe(coeffs_df.style.highlight_min(axis=0, color='lightgreen'))

# --- PAGE ROUTING ---
# Routing logic updated to show "Cost Predictor" first
if page == "Cost Predictor":
    page_cost_predictor()
else:
    page_data_explorer()
