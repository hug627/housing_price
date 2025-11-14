# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import training utility functions you already have
from utils import (
    train_linear_regression,
    train_lasso_regression,
    train_ridge_regression
)

# ----------------------- App Config / CSS -----------------------
st.set_page_config(page_title="House Price Prediction", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    .main-header { font-size: 2.6rem; font-weight:700; color:#FF9900; text-align:center; margin-bottom:1rem; }
    .sub-header { font-size:1.2rem; color:#232F3E; margin-top:1rem; margin-bottom:0.5rem; }
    .small-muted { color: #6b7280; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------- Sidebar -----------------------
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio("Choose a page", ["🏠 Home", "📊 Data Overview", "📈 Visualizations", "🤖 Model Training", "🔮 Predictions", "📉 Model Comparison"])

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data source")
data_source = st.sidebar.radio("Choose data source:", ["Use default file", "Upload CSV"])

# ✅ FIX — use dataset inside repository (NOT local Windows path)
DEFAULT_PATH = "usa_housing.csv"     # <-- THIS IS THE FIX

# ----------------------- Data Loading -----------------------
@st.cache_data
def load_data_cached(path_or_buffer):
    """Load CSV and basic preprocessing. No Streamlit UI calls here (so caching is safe)."""
    if isinstance(path_or_buffer, str):
        df = pd.read_csv(path_or_buffer)
    else:
        # uploaded file-like
        df = pd.read_csv(path_or_buffer)
    df.columns = df.columns.str.strip()
    return df

def load_data(path_choice):
    """Wrapper to call cached loader and optionally fix simple issues."""
    try:
        df = load_data_cached(path_choice)
        return df
    except Exception as e:
        st.error(f"Error while loading dataset: {e}")
        return None

# ----------------------- Helpers -----------------------
EXPECTED_FEATURES = ['Avg. Area Income', 'Avg. Area House Age',
                     'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
                     'Area Population']
TARGET = 'Price'

def validate_dataset(df):
    """Check if expected columns exist."""
    missing = [c for c in EXPECTED_FEATURES + [TARGET] if c not in df.columns]
    return missing

def show_data_preview(df):
    st.subheader("📋 Dataset Preview")
    st.write("**Columns:**", df.columns.tolist())
    left, right = st.columns(2)
    with left:
        st.write("**First 5 rows**")
        st.dataframe(df.head())
    with right:
        st.write("**Last 5 rows**")
        st.dataframe(df.tail())
    st.write("**Quick info**")
    st.write(f"- Records: **{len(df):,}**")
    st.write(f"- Features: **{len(df.columns)}**")
    st.write(f"- Missing values (total): **{int(df.isnull().sum().sum())}**")
    if st.checkbox("Show descriptive statistics"):
        st.dataframe(df.describe().T)

# ----------------------- Main -----------------------
def main():
    # Load data (either default or uploaded)
    df = None
    if data_source == "Use default file":
        df = load_data(DEFAULT_PATH)
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = load_data(uploaded)

    # If data failed to load, show message and stop
    if df is None:
        st.title("🏠 House Price Prediction System")
        st.markdown("Please load a dataset using the sidebar (default path or upload).")
        return

    # Validate columns
    missing = validate_dataset(df)
    if missing:
        st.warning(f"The dataset is missing expected columns: {missing}. The app expects: {EXPECTED_FEATURES + [TARGET]}")

    # PAGE ROUTING
    if page == "🏠 Home":
        st.markdown('<div class="main-header">🏠 House Price Prediction System</div>', unsafe_allow_html=True)
        st.write("Welcome! This app lets you explore data, train regression models, compare them, and make predictions.")
        show_data_preview(df)

    elif page == "📊 Data Overview":
        st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
        show_data_preview(df)

        if st.checkbox("Show missing value map"):
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.heatmap(df.isnull(), cbar=False)
            st.pyplot(fig)

    elif page == "📈 Visualizations":
        st.markdown('<h2 class="sub-header">📈 Data Visualizations</h2>', unsafe_allow_html=True)
        col = st.selectbox("Select a column to visualize", df.columns, index=0)
        if pd.api.types.is_numeric_dtype(df[col]):
            st.write(f"### Distribution of `{col}`")
            fig = px.histogram(df, x=col, nbins=50, marginal="box", title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"### Value counts for `{col}`")
            vc = df[col].value_counts().reset_index()
            vc.columns = [col, 'count']
            st.dataframe(vc.head(50))

        if st.checkbox("Show correlation heatmap"):
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] < 2:
                st.info("Not enough numeric columns for correlation.")
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm")
                st.pyplot(fig)

    elif page == "🤖 Model Training":
        st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
        if missing:
            st.error("Cannot train models. Dataset missing required columns.")
            return

        model_option = st.selectbox("Choose model to train", ["Linear Regression", "Ridge Regression", "Lasso Regression"])
        train_button = st.button("Train Selected Model")

        if train_button:
            with st.spinner("Training model..."):
                try:
                    if model_option == "Linear Regression":
                        res = train_linear_regression(df)
                    elif model_option == "Ridge Regression":
                        res = train_ridge_regression(df)
                    else:
                        res = train_lasso_regression(df)
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    return

            st.session_state['trained_results'] = res
            st.success(f"{model_option} trained successfully!")

            st.write("### ✅ Test Metrics")
            st.json(res['metrics']['test'])

            # Actual vs Predicted
            if 'test_predictions' in res and 'y_test' in res:
                y_true = pd.Series(res['y_test']).reset_index(drop=True)
                y_pred = pd.Series(res['test_predictions']).reset_index(drop=True)
                comp_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})

                fig = px.scatter(comp_df, x='Actual', y='Predicted', trendline="ols", title="Actual vs Predicted")
                st.plotly_chart(fig, use_container_width=True)

                comp_df['Residual'] = comp_df['Actual'] - comp_df['Predicted']
                fig2 = px.histogram(comp_df, x='Residual', nbins=50, title="Residuals Distribution")
                st.plotly_chart(fig2, use_container_width=True)

            # Download model
            try:
                model_bytes = pickle.dumps(res['model'])
                st.download_button(
                    label="📥 Download trained model (pickle)",
                    data=model_bytes,
                    file_name=f"trained_model_{model_option.replace(' ','_')}.pkl"
                )
            except Exception:
                st.info("Could not prepare model download.")

    elif page == "🔮 Predictions":
        st.markdown('<h2 class="sub-header">🔮 Make Predictions</h2>', unsafe_allow_html=True)

        if 'trained_results' not in st.session_state:
            st.warning("No trained model found. Train one on the Model Training page.")
            return

        res = st.session_state['trained_results']
        model = res['model']

        st.write("Enter feature values to predict house price:")
        cols = st.columns(2)
        user_values = {}
        for i, feat in enumerate(EXPECTED_FEATURES):
            default = float(df[feat].median()) if feat in df.columns else 0.0
            if i % 2 == 0:
                user_values[feat] = cols[0].number_input(feat, value=default, format="%.4f")
            else:
                user_values[feat] = cols[1].number_input(feat, value=default, format="%.4f")

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([user_values])
                pred = model.predict(input_df)[0]
                st.success(f"🏡 Predicted House Price: **${pred:,.2f}**")
                st.write("Input used:")
                st.dataframe(input_df.T.rename(columns={0: "value"}))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.markdown("---")
        st.write("Batch prediction on top rows:")
        nrows = st.number_input("Rows to predict", min_value=1, max_value=min(500, len(df)), value=20)

        if st.button("Predict on top rows"):
            try:
                batch_df = df[EXPECTED_FEATURES].head(nrows)
                preds = model.predict(batch_df)
                out = batch_df.copy()
                out['Predicted_Price'] = preds
                st.dataframe(out.head(50))
                csv = out.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download predictions CSV", csv, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

    elif page == "📉 Model Comparison":
        st.markdown('<h2 class="sub-header">📉 Model Comparison</h2>', unsafe_allow_html=True)

        if missing:
            st.error("Cannot compare models. Dataset missing required columns.")
            return

        if st.button("Train & Compare All Models"):
            with st.spinner("Training all models..."):
                try:
                    linear = train_linear_regression(df)
                    ridge = train_ridge_regression(df)
                    lasso = train_lasso_regression(df)
                except Exception as e:
                    st.error(f"Model training error: {e}")
                    return

            comp = pd.DataFrame([
                ["Linear Regression", linear['metrics']['test']['mae'], linear['metrics']['test']['rmse'], linear['metrics']['test']['r2']],
                ["Ridge Regression", ridge['metrics']['test']['mae'], ridge['metrics']['test']['rmse'], ridge['metrics']['test']['r2']],
                ["Lasso Regression", lasso['metrics']['test']['mae'], lasso['metrics']['test']['rmse'], lasso['metrics']['test']['r2']]
            ], columns=["Model", "MAE", "RMSE", "R² Score"])

            st.write("### 📋 Comparison Table")
            st.dataframe(comp)

            st.write("### 📈 Charts")
            fig_rmse = px.bar(comp, x="Model", y="RMSE", text="RMSE", title="RMSE Comparison")
            st.plotly_chart(fig_rmse, use_container_width=True)

            fig_r2 = px.bar(comp, x="Model", y="R² Score", text="R² Score", title="R² Score Comparison")
            st.plotly_chart(fig_r2, use_container_width=True)

            st.success("Comparison complete.")
        else:
            st.info("Click button to train and compare all models.")

# Start
if __name__ == "__main__":
    main()
