import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('figures', exist_ok=True)

# ── 1. LOAD RAW DATA ──────────────────────────────────────────────────────────
# Load the CSV from your kagglehub cache path into a dataframe called `data`
# Print .info() and .describe() to get a first look at dtypes and basic stats
file_name = "ecommerce_sales_dataset.csv"
data = pd.read_csv(file_name)
data['Order_Date'] = pd.to_datetime(data['Order_Date'])
data = data.sort_values('Order_Date').reset_index(drop=True)
print(data)
'''
print(data)
print(
    data.info(),
    data.head()
)
print(data['Revenue'].skew())
data['log_rev'] = np.log(data['Revenue'])
print(data['log_rev'].skew())
'''
# ── 2. CHECK DATA QUALITY ────────────────────────────────────────────────────
# Check for: missing values (.isnull().sum()), duplicates (.duplicated().sum())
# Print Order_Status value counts — this is your target, understand its class balance
print(data.isnull().sum(),
    data.duplicated().sum())
print(data['Order_Status'].value_counts()) #most of them are delivered, quite few are returned. see cancelled
#print(data['Quantity'].skew())
# ── 3. DISTRIBUTION OF NUMERIC COLUMNS ───────────────────────────────────────
# Plot histograms for: Unit_Price, Revenue, Cost, Profit, Shipping_Cost, Quantity
# Use plt.subplots to show them in a grid (e.g. 2 rows x 3 cols)
# Look for skew — if a distribution has a long right tail, it needs log compression
# Save figure to figures/numeric_distributions.png
data['log_price'] = np.log(data['Unit_Price'])

plt.figure()
plt.hist(data['Unit_Price'], 100)
#plt.show()

plt.figure()
plt.hist(data['log_price'], 100)
#plt.show()
numeric_cols = ['log_price', 'Profit', 'Revenue', 'Quantity', 'Discount', 'Unit_Price', 'Cost', 'Shipping_Cost']
corr_matrix = data[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True)
#plt.show()
plt.savefig("figures/corr_matrix.png")
#print(data['Year'].value_counts())

# ── 4. LOG COMPRESSION CHECK ──────────────────────────────────────────────────
# For each skewed column, compare: original vs np.log1p(column)
# Print .skew() before and after — a good transform brings skew closer to 0
# Columns likely needing log: Unit_Price, Revenue, Cost, Profit
skewed = data[numeric_cols].skew()
compressed = np.log(data[numeric_cols]).skew()
print(("compare raw with compressed:\n", skewed), ("compressed:\n", compressed))

# ── 5. CORRELATION HEATMAP ────────────────────────────────────────────────────
# Select numeric columns and compute .corr()
# Plot with sns.heatmap (annot=True, fmt='.2f', cmap='coolwarm')
# Save to figures/corr_matrix.png


# ── 6. CATEGORICAL BREAKDOWNS ─────────────────────────────────────────────────
# Bar plots for: Order_Status by Category, by Region, by Season
# This shows which categories/regions have higher cancellation rates
# Use df.groupby('Category')['Order_Status'].value_counts(normalize=True)
# Save figures to figures/status_by_category.png etc.


# ── 7. TIME SERIES VIEW ───────────────────────────────────────────────────────
# Convert Order_Date to datetime
# Group by month (use dt.to_period('M')) and count cancellations vs deliveries
# Plot a line chart showing cancellation rate over time
# This is key — we are doing time series, so trend matters
# Save to figures/cancellation_trend.png


# ── 8. CLASS IMBALANCE CHECK ──────────────────────────────────────────────────
# Print the percentage of each Order_Status class
# If one class dominates (e.g. 90% Delivered), note it — you will need to handle
# this during modeling (class_weight='balanced' or resampling)
