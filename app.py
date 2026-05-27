import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# ======================================================
# PAGE SETTINGS
# ======================================================

st.set_page_config(
    page_title="Restaurant Demand Forecasting Dashboard",
    page_icon="🍽️",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1, h2, h3 {
    color: #1f2937;
}

[data-testid="metric-container"] {
    background-color: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# TITLE
# ======================================================

st.title("🍽️ Restaurant Demand Forecasting Dashboard")

st.markdown("""
### AI-Based Restaurant Sales Forecasting & Inventory Optimization System
Built using Tuned XGBoost Regressor and Time Series Forecasting
""")

# ======================================================
# PATHS
# ======================================================

processed_dir = Path("Processed_Files")
models_dir = Path("models")

# ======================================================
# LOAD DATA
# ======================================================

test_df = pd.read_csv(
    processed_dir / "test_features.csv",
    parse_dates=["date"]
)

# ======================================================
# LOAD MODEL
# ======================================================

model = joblib.load(
    models_dir / "xgboost_tuned_week3.pkl"
)

# ======================================================
# FEATURE COLUMNS
# ======================================================

FEATURE_COLS = [

    'total_promo_items',
    'transactions',
    'oil_price',

    'year',
    'month',
    'day',
    'dayofweek',
    'weekofyear',
    'quarter',

    'is_weekend',
    'is_month_start',
    'is_month_end',

    'is_holiday',
    'holiday_level',

    'sales_lag_7',
    'sales_lag_14',
    'sales_lag_28',

    'rolling_mean_7',
    'rolling_std_14',
    'rolling_max_7',

    'promo_lag_7',
    'promo_rolling_7'
]

# ======================================================
# PREDICTION
# ======================================================

X_test = test_df[FEATURE_COLS]

pred_log = model.predict(X_test)

pred_raw = np.expm1(pred_log)

pred_raw = np.clip(
    pred_raw,
    0,
    None
)

actual = test_df['total_sales'].values

# ======================================================
# EVALUATION
# ======================================================

mae = mean_absolute_error(
    actual,
    pred_raw
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        pred_raw
    )
)

# ======================================================
# FORECAST DATAFRAME
# ======================================================

forecast_df = pd.DataFrame({

    'Date': test_df['date'],

    'Actual Sales': actual,

    'Predicted Sales': pred_raw
})

forecast_df['Forecast Error'] = (

    forecast_df['Actual Sales']

    - forecast_df['Predicted Sales']
)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3075/3075977.png",
    width=100
)

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(

    "Select Page",

    [
        "Overview",
        "Forecast Chart",
        "Residual Analysis",
        "Feature Importance",
        "Forecast Table",
        "Dataset"
    ]
)

# ======================================================
# OVERVIEW
# ======================================================

if page == "Overview":

    st.subheader("📊 Model Performance Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{mae:,.2f}"
    )

    col2.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )

    col3.metric(
        "Test Rows",
        len(test_df)
    )

    st.markdown("---")

    st.subheader("📅 Dataset Information")

    info1, info2 = st.columns(2)

    info1.info(
        f"Start Date: {test_df['date'].min().date()}"
    )

    info2.info(
        f"End Date: {test_df['date'].max().date()}"
    )

    st.markdown("---")

    st.subheader("📌 Business Insights")

    avg_actual = forecast_df[
        'Actual Sales'
    ].mean()

    avg_pred = forecast_df[
        'Predicted Sales'
    ].mean()

    high_days = forecast_df[
        forecast_df['Predicted Sales']
        >
        forecast_df['Predicted Sales'].quantile(0.90)
    ]

    st.success(
        f"Average Actual Daily Sales : {avg_actual:,.2f}"
    )

    st.success(
        f"Average Predicted Daily Sales : {avg_pred:,.2f}"
    )

    st.warning(
        f"High Demand Forecast Days : {len(high_days)}"
    )

# ======================================================
# FORECAST CHART
# ======================================================

elif page == "Forecast Chart":

    st.subheader("📈 Forecast vs Actual Sales")

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=test_df['date'],

            y=actual,

            mode='lines',

            name='Actual Sales',

            line=dict(
                color='blue',
                width=3
            )
        )
    )

    fig.add_trace(

        go.Scatter(

            x=test_df['date'],

            y=pred_raw,

            mode='lines',

            name='Predicted Sales',

            line=dict(
                color='red',
                width=3,
                dash='dash'
            )
        )
    )

    fig.update_layout(

        template='plotly_white',

        height=550,

        title='Forecast vs Actual Sales',

        xaxis_title='Date',

        yaxis_title='Sales'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ======================================================
# RESIDUAL ANALYSIS
# ======================================================

elif page == "Residual Analysis":

    st.subheader("📉 Residual Error Analysis")

    residuals = actual - pred_raw

    # ==================================================
    # RESIDUAL OVER TIME
    # ==================================================

    fig1 = go.Figure()

    fig1.add_trace(

        go.Scatter(

            x=test_df['date'],

            y=residuals,

            mode='lines',

            name='Residuals',

            line=dict(
                color='purple',
                width=2
            )
        )
    )

    fig1.add_hline(
        y=0,
        line_dash='dash',
        line_color='red'
    )

    fig1.update_layout(

        template='plotly_white',

        height=500,

        title='Residuals Over Time',

        xaxis_title='Date',

        yaxis_title='Actual - Predicted'
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # ==================================================
    # HISTOGRAM
    # ==================================================

    st.subheader("📊 Residual Distribution")

    hist_fig = px.histogram(

        residuals,

        nbins=40,

        title='Residual Distribution'
    )

    hist_fig.update_layout(

        template='plotly_white',

        height=500,

        xaxis_title='Residual',

        yaxis_title='Count'
    )

    st.plotly_chart(
        hist_fig,
        use_container_width=True
    )

    # ==================================================
    # METRICS
    # ==================================================

    st.subheader("📌 Residual Statistics")

    r1, r2 = st.columns(2)

    r1.metric(
        "Mean Residual",
        f"{residuals.mean():,.2f}"
    )

    r2.metric(
        "Std Residual",
        f"{residuals.std():,.2f}"
    )

# ======================================================
# FEATURE IMPORTANCE
# ======================================================

elif page == "Feature Importance":

    st.subheader("🔥 Feature Importance")

    feat_imp = pd.DataFrame({

        'Feature': FEATURE_COLS,

        'Importance':
        model.feature_importances_
    })

    feat_imp = feat_imp.sort_values(
        by='Importance',
        ascending=False
    )

    fig = px.bar(

        feat_imp,

        x='Importance',

        y='Feature',

        orientation='h',

        title='Feature Importance Ranking',

        height=650
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("### Top 5 Features")

    st.dataframe(
        feat_imp.head(5)
    )

# ======================================================
# FORECAST TABLE
# ======================================================

elif page == "Forecast Table":

    st.subheader("📋 Forecast Results Table")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    csv = forecast_df.to_csv(
        index=False
    ).encode('utf-8')

    st.download_button(

        label="⬇ Download Forecast CSV",

        data=csv,

        file_name='forecast_results.csv',

        mime='text/csv'
    )

# ======================================================
# DATASET
# ======================================================

elif page == "Dataset":

    st.subheader("📁 Test Dataset")

    st.dataframe(
        test_df,
        use_container_width=True
    )

    st.markdown("---")

    st.write("### Dataset Shape")

    st.write(test_df.shape)

    st.write("### Dataset Columns")

    st.write(list(test_df.columns))