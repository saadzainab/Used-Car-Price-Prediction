import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Model Performance | Car Price Predictor", page_icon="📊", layout="wide")

METRICS_PATH = "outputs/metrics.json"
COMPARISON_IMG = "outputs/model_comparison_r2.png"

st.title("📊 Model Performance")

if not os.path.exists(METRICS_PATH):
    st.warning(
        f"`{METRICS_PATH}` not found. Run `train_model.py` to generate metrics, "
        "or manually place a `metrics.json` file in the `outputs/` folder."
    )
    st.stop()

with open(METRICS_PATH) as f:
    metrics = json.load(f)

# ============================================================
# Baseline model comparison
# ============================================================

st.subheader("Baseline Model Comparison")
st.write("Four regression models were trained and evaluated on a held-out test set (80/20 split).")

results_df = pd.DataFrame(metrics["baseline_results"]).T
results_df = results_df.rename(columns={"MAE": "MAE (PKR)", "RMSE": "RMSE (PKR)"})
results_df = results_df.round(2)

st.dataframe(results_df, use_container_width=True)

if os.path.exists(COMPARISON_IMG):
    st.image(COMPARISON_IMG, caption="R² Score by model")
else:
    st.bar_chart(pd.DataFrame(metrics["baseline_results"]).T["R2 Score"])

st.success(f"**Best baseline model:** {metrics['best_baseline_model']}")

st.divider()

# ============================================================
# Tuned Random Forest
# ============================================================

st.subheader("Hyperparameter-Tuned Random Forest")
st.write("The best baseline model was tuned further with grid search cross-validation (5-fold).")

tuned = metrics["tuned_random_forest"]

col1, col2, col3 = st.columns(3)
col1.metric("MAE (PKR)", f"{tuned['MAE']:,.0f}")
col2.metric("RMSE (PKR)", f"{tuned['RMSE']:,.0f}")
col3.metric("R² Score", f"{tuned['R2 Score']:.4f}")

st.write("**Best hyperparameters found:**")
st.json(tuned["best_params"])

st.divider()

col1, col2 = st.columns(2)
col1.metric("Training samples", f"{metrics.get('train_samples', '—'):,}" if metrics.get("train_samples") else "—")
col2.metric("Test samples", f"{metrics.get('test_samples', '—'):,}" if metrics.get("test_samples") else "—")
