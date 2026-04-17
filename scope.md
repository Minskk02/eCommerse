# Project Scope — E-Commerce Dropshipping Predictive Model

## Dataset
Global E-Commerce Sales Dataset 2021–2024 (Kaggle)
Columns: Order_ID, Order_Date, Year, Month, Quarter, Season, Customer_ID, Customer_Gender,
Customer_Segment, Region, Country, Category, Sub_Category, Product_Name, Unit_Price,
Quantity, Discount, Revenue, Cost, Profit, Profit_Margin_%, Shipping_Cost, Shipping_Method,
Shipping_Days, Payment_Method, Order_Status

---

## Why ML was dropped
EDA revealed the dataset is fully synthetic with Order_Status randomly assigned independent
of all features. ANOVA tests showed only Category has meaningful effect on Revenue (p≈0.000),
all other columns (Season, Region, Gender, Segment, Shipping_Method) are noise. The dataset
is not suitable for predictive modeling but is well-structured for visualization practice.

## Objectives — Data Visualisation

### 1. Distribution Analysis
- Histograms + log-compressed comparisons for skewed numeric columns
- Boxplots / violin plots across categories

### 2. Category & Regional Breakdown
- Revenue and Profit by Category, Sub_Category, Region, Country
- Bar charts, grouped comparisons

### 3. Time Series
- Revenue/Profit trends over time (monthly, quarterly, seasonal)
- Line charts with trend lines

### 4. Correlation & Relationships
- Heatmap of numeric columns
- Scatter plots for key relationships

### 5. Dashboard-style Summary
- A clean multi-panel figure suitable for a README or portfolio

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
