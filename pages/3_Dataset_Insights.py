import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Dataset Insights | Car Price Predictor", page_icon="📈", layout="wide")

CLEAN_CSV = "data/car_df_clean.csv"

st.title("📈 Dataset Insights")

if not os.path.exists(CLEAN_CSV):
    st.warning(f"`{CLEAN_CSV}` not found. Run `data_prep.py` first to generate the cleaned dataset.")
    st.stop()

df = pd.read_csv(CLEAN_CSV)

st.write(f"Exploring **{len(df):,}** cleaned car listings.")

with st.expander("Preview raw rows"):
    st.dataframe(df.head(20), use_container_width=True)

st.divider()

# ============================================================
# Price distribution
# ============================================================

st.subheader("Price Distribution")
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["Price"], bins=50, kde=True, ax=ax)
ax.set_xlabel("Price (PKR)")
ax.set_ylabel("Number of Cars")
st.pyplot(fig)

st.divider()

# ============================================================
# Top manufacturers + cities
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 15 Manufacturers")
    fig, ax = plt.subplots(figsize=(6, 5))
    df["Make"].value_counts().head(15).plot(kind="bar", ax=ax)
    ax.set_xlabel("Manufacturer")
    ax.set_ylabel("Number of Cars")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

with col2:
    st.subheader("Top 15 Cities")
    fig, ax = plt.subplots(figsize=(6, 5))
    df["Registration_City"].value_counts().head(15).plot(kind="bar", ax=ax)
    ax.set_xlabel("City")
    ax.set_ylabel("Number of Listings")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

st.divider()

# ============================================================
# Average price by brand
# ============================================================

st.subheader("Average Price by Manufacturer (Top 15)")
brand_price = df.groupby("Make")["Price"].mean().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(10, 4))
brand_price.plot(kind="bar", ax=ax)
ax.set_ylabel("Average Price (PKR)")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)

st.divider()

# ============================================================
# Year vs Price / Mileage vs Price
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Manufacturing Year vs Price")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x="Year", y="Price", alpha=0.4, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Mileage vs Price")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x="Mileage", y="Price", alpha=0.4, ax=ax)
    st.pyplot(fig)

st.divider()

# ============================================================
# Correlation heatmap
# ============================================================

st.subheader("Numerical Feature Correlation")
numeric_df = df.select_dtypes(include=np.number)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
