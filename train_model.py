import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CLEAN_CSV = "data/car_df_clean.csv"
MODEL_PATH = "models/car_price_prediction_model.pkl"

# ============================================================
# Load cleaned data
# ============================================================

car_df = pd.read_csv(CLEAN_CSV)

X = car_df.drop(columns=["Price"])
y = car_df["Price"]

categorical_features = [
    "Make",
    "Model",
    "Fuel",
    "Registration_City",
    "Documents",
    "Assembly",
    "Transmission",
]

numerical_features = [
    "Year",
    "Mileage",
    "Car_Age",
    "Mileage_Per_Year",
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples :", X_test.shape)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ============================================================
# Train + compare candidate models
# ============================================================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results = {}

for name, model in models.items():
    print("\nTraining:", name)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2 Score": r2}
    print("MAE:", mae, "| RMSE:", rmse, "| R2 Score:", r2)

results_df = pd.DataFrame(results).T
print("\nModel comparison:\n", results_df)

plt.figure(figsize=(10, 5))
sns.barplot(x=results_df.index, y=results_df["R2 Score"])
plt.xticks(rotation=45)
plt.title("Model Performance Comparison (R2 Score)")
plt.savefig("outputs/model_comparison_r2.png", bbox_inches="tight")
plt.show()

best_model = results_df["R2 Score"].idxmax()
print("\nBest baseline model:", best_model)

# ============================================================
# Hyperparameter tuning (Random Forest)
# ============================================================

rf_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(random_state=42))]
)

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nTuned Random Forest:")
print("MAE:", mae, "| RMSE:", rmse, "| R2 Score:", r2)

# ============================================================
# Save the final model
# ============================================================

joblib.dump(best_rf, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

# ============================================================
# Save metrics (used by the Streamlit "Model Performance" page)
# ============================================================

metrics = {
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "baseline_results": results,
    "best_baseline_model": best_model,
    "tuned_random_forest": {
        "best_params": grid_search.best_params_,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
    },
}

with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print("Metrics saved to: outputs/metrics.json")
