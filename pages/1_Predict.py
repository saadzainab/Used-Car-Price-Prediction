import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "models/car_price_prediction_model.pkl"
CLEAN_CSV = "data/car_df_clean.csv"

st.set_page_config(page_title="Predict | Car Price Predictor", page_icon="🔮", layout="wide")

st.title("🔮 Predict a Car's Price")
st.write("Fill in the car details below to get an estimated price.")

# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at `{MODEL_PATH}`. Make sure it's in the `models/` folder.")
    st.stop()

model = load_model()

# ============================================================
# Load cleaned data (used only to populate dropdown options)
# ============================================================

@st.cache_data
def load_options():
    if os.path.exists(CLEAN_CSV):
        df = pd.read_csv(CLEAN_CSV)
        return {
            "Make": sorted(df["Make"].dropna().unique().tolist()),
            "Model": sorted(df["Model"].dropna().unique().tolist()),
            "Fuel": sorted(df["Fuel"].dropna().unique().tolist()),
            "Registration_City": sorted(df["Registration_City"].dropna().unique().tolist()),
            "Documents": sorted(df["Documents"].dropna().unique().tolist()),
            "Assembly": sorted(df["Assembly"].dropna().unique().tolist()),
            "Transmission": sorted(df["Transmission"].dropna().unique().tolist()),
        }
    return None

options = load_options()

if options is None:
    st.warning(
        f"`{CLEAN_CSV}` not found, so dropdowns will use free text instead of "
        "real values from the dataset. Run `data_prep.py` to enable dropdowns."
    )

# ============================================================
# Input form
# ============================================================

with st.form("car_form"):
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year", min_value=1990, max_value=2026, value=2018, step=1)
        mileage = st.number_input("Mileage (KM's driven)", min_value=0, value=60000, step=1000)

        if options:
            make = st.selectbox("Make", options["Make"])
            fuel = st.selectbox("Fuel", options["Fuel"])
            registration_city = st.selectbox("Registration City", options["Registration_City"])
        else:
            make = st.text_input("Make", "Toyota")
            fuel = st.text_input("Fuel", "Petrol")
            registration_city = st.text_input("Registration City", "Lahore")

    with col2:
        if options:
            model_name = st.selectbox("Model", options["Model"])
            documents = st.selectbox("Documents", options["Documents"])
            assembly = st.selectbox("Assembly", options["Assembly"])
            transmission = st.selectbox("Transmission", options["Transmission"])
        else:
            model_name = st.text_input("Model", "Corolla")
            documents = st.text_input("Documents", "Complete")
            assembly = st.text_input("Assembly", "Local")
            transmission = st.text_input("Transmission", "Automatic")

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# ============================================================
# Predict
# ============================================================

if submitted:
    current_year = 2026
    car_age = current_year - year
    mileage_per_year = mileage / (car_age + 1)

    new_car = pd.DataFrame([{
        "Year": year,
        "Mileage": mileage,
        "Car_Age": car_age,
        "Mileage_Per_Year": mileage_per_year,
        "Make": make,
        "Model": model_name,
        "Fuel": fuel,
        "Registration_City": registration_city,
        "Documents": documents,
        "Assembly": assembly,
        "Transmission": transmission,
    }])

    predicted_price = model.predict(new_car)[0]

    st.success(f"### Estimated Price: PKR {predicted_price:,.0f}")

    with st.expander("See input data used for prediction"):
        st.dataframe(new_car, use_container_width=True)
