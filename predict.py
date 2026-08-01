import pandas as pd
import joblib

MODEL_PATH = "models/car_price_prediction_model.pkl"

model = joblib.load(MODEL_PATH)

new_car = pd.DataFrame([{
    "Year": 2018,
    "Mileage": 60000,
    "Car_Age": 8,
    "Mileage_Per_Year": 6666.7,
    "Make": "Toyota",
    "Model": "Corolla",
    "Fuel": "Petrol",
    "Registration_City": "Lahore",
    "Documents": "Complete",
    "Assembly": "Local",
    "Transmission": "Automatic",
}])

predicted_price = model.predict(new_car)[0]
print("Predicted Price (PKR):", round(predicted_price, 2))


