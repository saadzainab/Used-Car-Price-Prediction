
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

RAW_CSV = "data/OLX_cars_dataset00.csv"
CLEAN_CSV = "data/car_df_clean.csv"
CURRENT_YEAR = 2026

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(RAW_CSV)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("Columns in dataset:", df.columns.tolist())

df.info()
print(df.describe())

missing_values = df.isnull().sum()
print("\nMissing values:\n", missing_values[missing_values > 0])

duplicates = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicates)

for column in df.columns:
    print("\nColumn:", column, "| Unique values:", df[column].nunique())

# ============================================================
# EDA plots 
# ============================================================

plt.figure(figsize=(10, 5))
sns.histplot(df["Price"], bins=50, kde=True)
plt.title("Distribution of Car Prices")
plt.xlabel("Price (PKR)")
plt.savefig("outputs/price_distribution.png", bbox_inches="tight")
#plt.show()

plt.figure(figsize=(12, 6))
df["Make"].value_counts().head(15).plot(kind="bar")
plt.title("Top 15 Car Manufacturers")
plt.xticks(rotation=45)
plt.savefig("outputs/top_manufacturers.png", bbox_inches="tight")
#plt.show()

plt.figure(figsize=(12, 6))
df["Registration city"].value_counts().head(15).plot(kind="bar")
plt.title("Number of Cars Listed by City")
plt.xticks(rotation=45)
plt.savefig("outputs/cars_by_city.png", bbox_inches="tight")
#plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Year", y="Price", alpha=0.5)
plt.title("Manufacturing Year vs Price")
plt.savefig("outputs/year_vs_price.png", bbox_inches="tight")
#plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="KM's driven", y="Price", alpha=0.5)
plt.title("Mileage vs Car Price")
plt.savefig("outputs/mileage_vs_price.png", bbox_inches="tight")
#plt.show()

numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Numerical Feature Correlation")
plt.savefig("outputs/correlation_heatmap.png", bbox_inches="tight")
#plt.show()

# ============================================================
# Cleaning
# ============================================================

# Missing Values
car_df = df.copy()
car_df.isnull().sum()

# Remove Unnecessary Columns

columns_to_drop = [
    "Ad ID",
    "Car Name",
    "Seller Location",
    "Description",
    "Car Features",
    "Car Profile"
]


car_df.drop(
    columns=columns_to_drop,
    inplace=True
)

print(car_df.columns)
print(car_df.head())

# Duplicate Rows

print(
    "Duplicate rows:",
    car_df.duplicated().sum()
)

car_df.drop_duplicates(
    inplace=True
)

print(
    "Updated Duplicate rows:", car_df.duplicated().sum()
)
print(car_df.shape)


# Rename Columns

car_df.rename(
    columns={
        "KM's driven": "Mileage",
        "Registration city": "Registration_City",
        "Car documents": "Documents"
    },
    inplace=True
)


print ( "Updates columns : ",  car_df.columns)

# Feature Engineering: Car Age

current_year = 2026

car_df["Car_Age"] = current_year - car_df["Year"]

# Unique Values in Categorical Columns
categorical_columns = car_df.select_dtypes(
    include="object"
).columns


for col in categorical_columns:
    print("\n",col)
    print("Unique values:", car_df[col].nunique())


for col in ["Condition", "Images URL's"]:
    if col in car_df.columns:
        car_df.drop(columns=[col], inplace=True)

# ============================================================
# Feature Engineering
# ============================================================

car_df["Car_Age"] = CURRENT_YEAR - car_df["Year"]
car_df["Mileage_Per_Year"] = car_df["Mileage"] / (car_df["Car_Age"] + 1)

print(car_df.head())
print(car_df.shape)
car_df.info()

# ============================================================
# Save cleaned data
# ============================================================

car_df.to_csv(CLEAN_CSV, index=False)
print(f"\nSaved cleaned data to: {CLEAN_CSV}")
