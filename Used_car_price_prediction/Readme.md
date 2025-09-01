# 🚗 Used Car Price Prediction (BMW)

This project builds a **machine learning regression model** to predict the **resale price of used BMW cars** using historical data. The aim is to provide a **fair market estimate** based on key vehicle features such as model, year, mileage, fuel type, and engine specifications.

---

## 🎯 Objective

Due to the rising cost of new vehicles and the growing demand for affordable alternatives, the used car market is expanding rapidly. However, pricing in this domain is highly inconsistent and often unregulated.

This project aims to:
- Predict the price of a used BMW car based on its attributes
- Assist **buyers** in avoiding overpayment
- Help **sellers** price vehicles competitively
- Enhance market **transparency** and **efficiency**

---

## 📦 Dataset Description

📌 Source: [Kaggle - BMW Used Car Dataset](https://www.kaggle.com/code/celestioushawk/bmw-car-price-prediction/input?select=bmw.csv)

- **Total Records**: 10,782 rows × 9 columns  
- **Target**: `price` (in USD)

| Feature        | Type        | Description                                      |
|----------------|-------------|--------------------------------------------------|
| `model`        | Categorical | BMW model (e.g., 3 Series, X5)                   |
| `year`         | Numeric     | Year of manufacture                             |
| `mileage`      | Numeric     | Total distance driven (in miles)                |
| `mpg`          | Numeric     | Fuel efficiency (miles per gallon)              |
| `fuelType`     | Categorical | Fuel type (Petrol, Diesel, etc.)                |
| `transmission` | Categorical | Transmission type (Automatic, Manual, etc.)     |
| `tax`          | Numeric     | Annual tax amount                                |
| `engineSize`   | Numeric     | Engine capacity in liters                        |
| `price`        | Numeric     | Car's listed resale price (target variable)     |

---

## 🧠 Technology Stack

| Task                 | Tools / Libraries                     |
|----------------------|----------------------------------------|
| Language             | Python                                 |
| Data Manipulation    | Pandas, NumPy                          |
| Visualization        | Matplotlib, Seaborn                    |
| Model Development    | scikit-learn, RandomForestRegressor    |
| Metrics              | Root Mean Squared Error (RMSE), R²     |

---

## 🔍 Methodology

1. **Data Collection**  
   Sourced from online listings and dealer records on Kaggle.

2. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Train-test split (80/20)
   - Outlier removal

3. **Feature Engineering**
   - Derived new ratios (e.g., `mpg/age`, `engineSize/age`)
   - Correlation matrix to identify feature importance

4. **Modeling**
   - Used **Random Forest Regression** to predict price
   - Hyperparameter tuning to reduce overfitting
   - Evaluated using RMSE

5. **Prediction**
   - Compared predicted vs actual prices
   - Visualized result distributions

---

## 📊 Visualization Highlights

- **Correlation Matrix**: Identified strong negative correlation between mileage and price  
- **Bar Charts**: Vehicle count per model  
- **Scatter Plots**:  
   - `Price vs Year`: depreciation over time  
   - `Price vs Mileage`: downward trend  
   - `Price vs EngineSize`: positive relation up to a threshold

---

## 🧪 Model Performance

| Metric         | Value       |
|----------------|-------------|
| RMSE           | *e.g., 1500* |
| R² Score       | *e.g., 0.91* |
| Best Features  | Year, EngineSize, Mileage, MPG               |

---
---

## ✅ Conclusion

The developed model is effective for estimating the **fair resale price** of used BMW cars based on important features. It enables informed decision-making and improves price transparency in the used car market. Future enhancements can include:
- Adding more features (e.g., car type, insurance status)
- Supporting more brands (Audi, Mercedes, etc.)
- Deploying as a web app using Streamlit or Flask

---

## 📌 References

- https://www.kaggle.com/code/celestioushawk/bmw-car-price-prediction
- https://scikit-learn.org/
- https://pandas.pydata.org/
- https://seaborn.pydata.org/

---
