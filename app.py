import streamlit as st
import pandas as pd
import os
import json

st.set_page_config(
    page_title="Pakistan Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

CLEAN_CSV = "data/car_df_clean.csv"
METRICS_PATH = "outputs/metrics.json"

st.title("🚗 Pakistan Used Car Price Predictor")

st.markdown(
    """
An end-to-end machine learning project that predicts used car prices
in Pakistan from real OLX listing data — covering data cleaning,
exploratory analysis, model comparison, hyperparameter tuning, and
deployment as an interactive app.

Use the sidebar to navigate:

- **🔮 Predict** — get a live price estimate for a car
- **📊 Model Performance** — compare model accuracy and see the tuning results
- **📈 Dataset Insights** — explore the underlying data
- **ℹ️ About** — project details, tech stack, and links
"""
)

st.divider()

# ============================================================
# Quick stats row
# ============================================================

col1, col2, col3, col4 = st.columns(4)

if os.path.exists(CLEAN_CSV):
    df = pd.read_csv(CLEAN_CSV)
    col1.metric("Listings", f"{len(df):,}")
    col2.metric("Manufacturers", df["Make"].nunique())
    col3.metric("Cities Covered", df["Registration_City"].nunique())
else:
    col1.metric("Listings", "—")
    col2.metric("Manufacturers", "—")
    col3.metric("Cities Covered", "—")

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    r2 = metrics["tuned_random_forest"]["R2 Score"]
    col4.metric("Best Model R²", f"{r2:.3f}")
else:
    col4.metric("Best Model R²", "—")

st.divider()

st.markdown(
    """
### How it works
1. Raw listings are scraped from OLX Pakistan (Kaggle dataset)
2. Data is cleaned: duplicates removed, columns renamed, missing values handled
3. Features engineered: car age, mileage per year
4. Four regression models are trained and compared: Linear Regression,
   Decision Tree, Random Forest, and Gradient Boosting
5. The best-performing model (Random Forest) is hyperparameter-tuned
   with grid search and saved for deployment
6. This Streamlit app serves the trained model for live predictions
"""
)

st.info("👈 Start with the **Predict** page in the sidebar to try it out.")
