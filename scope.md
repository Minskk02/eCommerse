# Project Scope — E-Commerce Dropshipping Predictive Model

## Dataset
Global E-Commerce Sales Dataset 2021–2024 (Kaggle)
Columns: Order_ID, Order_Date, Year, Month, Quarter, Season, Customer_ID, Customer_Gender,
Customer_Segment, Region, Country, Category, Sub_Category, Product_Name, Unit_Price,
Quantity, Discount, Revenue, Cost, Profit, Profit_Margin_%, Shipping_Cost, Shipping_Method,
Shipping_Days, Payment_Method, Order_Status

---

## Objectives

### 1. Exploratory Data Analysis (EDA)
- Distribution plots for all numerical columns
- Identify skewed columns and apply log/sqrt compression where appropriate
- Correlation heatmap
- Category-level breakdowns (revenue, quantity, profit by category/region/season)
- Outlier detection

### 2. Feature Engineering
- Compress skewed numeric columns (e.g. log transform on Unit_Price, Revenue, Cost)
- Bin continuous variables into tiers where useful (e.g. price tiers)
- Extract temporal features from Order_Date (day_of_week, is_weekend, etc.)
- Identify and drop leaky columns before model training

### 3. Encoding
- OrdinalEncoder for ordered categoricals (Month, Quarter, Season)
- OneHotEncoder for low-cardinality nominals (Region, Gender, Shipping_Method)
- TargetEncoder for high-cardinality categoricals (Country, Category, Sub_Category, Customer_Segment)

### 4. Modeling (sklearn first)
- Start with baseline: DummyRegressor/DummyClassifier
- Try: Random Forest, Gradient Boosting, XGBoost
- Evaluate with appropriate metrics (R², RMSE for regression; accuracy, F1 for classification)
- Target accuracy: >90%

---

## Target Variable — Order_Status (Classification)

**Goal:** Predict whether an order will be Cancelled (or Returned) vs Delivered.
This gives dropshippers a risk score per product/region/season combination before sourcing.

**Approach:** Time series — train on 2021–2023, test on 2024. This reflects real-world
deployment where the model is trained on historical data and predicts on unseen future orders.

**Why this target:**
- Cancellation risk is the #1 sourcing concern for dropshippers
- No post-sale leakage: all features (price, category, region, season, shipping method) are
  known at order placement time
- Classification makes >90% accuracy achievable
- Actionable output: "This product+region+season has X% cancellation risk"

**Class mapping:**
- 0 = low risk (Delivered)
- 1 = high risk (Cancelled or Returned)

---

## Desired Output
- A ranked list of product/region/season combinations with predicted score
- Model saved and reusable (joblib or pickle)
- Clean figures saved to `figures/` for README
- `result.csv` with final predictions

---

## File Structure
```
e_commers/
├── scope.md            # this file
├── eda.py              # EDA + figures
├── preprocessor.py     # feature engineering + cache to parquet
├── main.py             # model training + prediction + result.csv
├── data/
│   └── processed.parquet
├── figures/
│   └── *.png
└── result.csv
```
