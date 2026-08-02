# Pakistan Used Car Price Prediction

Predicts used car prices from OLX Pakistan listing data.

## Structure

```
car-price-prediction/
├── app.py                          # Streamlit Home page
├── pages/
│   ├── 1_Predict.py                 # Live prediction form
│   ├── 2_Model_Performance.py       # Model comparison + tuning results
│   ├── 3_Dataset_Insights.py        # EDA visualizations
│   └── 4_About.py                   # Project write-up
├── data_prep.py       # Load, explore, clean data, feature engineering
├── train_model.py     # Train + compare models, tune Random Forest, save model + metrics
├── predict.py          # CLI: load the saved model and predict new prices
├── requirements.txt
├── data/               # put OLX_cars_dataset00.csv here
├── models/             # trained model gets saved here
└── outputs/            # plots + metrics.json get saved here
```

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/abdullahkhanuet22/olx-cars-dataset

and place `OLX_cars_dataset00.csv` inside the `data/` folder.

(Note: your original notebook had a Kaggle API key hardcoded in it —
rotate that key at kaggle.com/settings, since it's no longer needed here.)

## Run

```bash
python data_prep.py     # cleans data -> data/car_df_clean.csv
python train_model.py   # trains + tunes model -> models/car_price_prediction_model.pkl + outputs/metrics.json
python predict.py       # loads the model and predicts a sample car's price
```

### Launch the app

```bash
streamlit run app.py
```

Streamlit automatically picks up everything in `pages/` and builds a
sidebar with: Home, Predict, Model Performance, Dataset Insights, and
About.

If you already have a trained model but haven't run `data_prep.py`
or `train_model.py` locally, the app will still work — the Predict
page falls back to text inputs, and the Model Performance / Dataset
Insights pages will show a note asking you to generate the missing
files.

