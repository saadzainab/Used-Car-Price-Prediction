import streamlit as st

st.set_page_config(page_title="About | Car Price Predictor", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About This Project")

st.markdown(
    """
## Pakistan Used Car Price Predictor

This project predicts the resale price of used cars listed on OLX
Pakistan, using historical listing data and a tuned Random Forest
regression model.

### Problem
Pricing a used car is hard to get right — sellers often over- or
under-price listings, and buyers have no quick way to check whether
an asking price is fair. This tool gives an instant, data-driven
price estimate based on a car's year, mileage, make, model, and
other listing details.

### Approach
- **Data**: ~OLX Pakistan car listings, sourced from Kaggle
- **Cleaning**: removed duplicates and irrelevant columns, handled
  missing values, standardized column names
- **Feature engineering**: derived car age and mileage-per-year
  from raw year/mileage figures
- **Modeling**: compared Linear Regression, Decision Tree, Random
  Forest, and Gradient Boosting regressors
- **Tuning**: grid search cross-validation on the best-performing
  model
- **Deployment**: this multi-page Streamlit app, serving the trained
  model for live predictions

### Tech Stack
| Layer | Tools |
|---|---|
| Data processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Modeling | scikit-learn |
| App / deployment | Streamlit |

### Project Structure
```
car-price-prediction/
├── app.py                          # Home page
├── pages/
│   ├── 1_Predict.py                 # Live prediction form
│   ├── 2_Model_Performance.py       # Model comparison + tuning results
│   ├── 3_Dataset_Insights.py        # EDA visualizations
│   └── 4_About.py                   # This page
├── data_prep.py                     # Cleans raw data
├── train_model.py                   # Trains + tunes + saves the model
├── predict.py                       # CLI prediction script
├── data/                            # Raw + cleaned CSVs
├── models/                          # Saved model (.pkl)
└── outputs/                         # Saved plots + metrics
```

### Links
- **GitHub Repo**: _add your repo link here_
- **Live App**: _add your deployed Streamlit URL here_
- **Dataset**: [OLX Cars Dataset on Kaggle](https://www.kaggle.com/datasets/abdullahkhanuet22/olx-cars-dataset)

### Author
_Zainab Saad, https://www.linkedin.com/in/zaiinabb/, and https://github.com/saadzainab here._
"""
)
