import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)

# ── 1. LOAD RAW DATA ──────────────────────────────────────────────────────────
# Load the CSV from your kagglehub cache path into a variable called `data`


# ── 2. PARSE DATES & SORT ─────────────────────────────────────────────────────
# Convert Order_Date to datetime
# Sort the dataframe by Order_Date ascending — critical for time series
# so that train/test split respects chronological order


# ── 3. ENCODE TARGET COLUMN ───────────────────────────────────────────────────
# Map Order_Status to binary:
#   0 = low risk  (Delivered)
#   1 = high risk (Cancelled or Returned)
# Use .map() with a dictionary and store in a new column called `target`
# Print value_counts() to see class balance


# ── 4. SELECT & RENAME FEATURES ───────────────────────────────────────────────
# Keep only columns known at order placement time (no post-sale leakage)
# Safe to use: Category, Sub_Category, Country, Region, Season, Month, Year,
#              Customer_Gender, Customer_Segment, Shipping_Method,
#              Unit_Price, Shipping_Cost, Discount
# Also keep: Order_Date (needed for step 5), target
# Rename all columns to lowercase with underscores for consistency


# ── 5. FEATURE ENGINEERING ────────────────────────────────────────────────────
# Add day_of_week from Order_Date  (use .dt.dayofweek — Monday=0, Sunday=6)
# Add is_weekend: 1 if day_of_week >= 5, else 0
# Add log_unit_price    = np.log1p(unit_price)    — reduces right skew
# Add log_shipping_cost = np.log1p(shipping_cost)
# Add price_tier: use pd.qcut on log_unit_price into 3 bins
#                 labels=['budget', 'mid', 'premium']


# ── 6. DROP COLUMNS NO LONGER NEEDED ─────────────────────────────────────────
# Drop Order_Date — temporal info is now captured in month, year, day_of_week
# Drop raw unit_price and shipping_cost — replaced by log versions
# Keep price_tier as an additional categorical signal


# ── 7. SAVE TO PARQUET ────────────────────────────────────────────────────────
# Save the processed dataframe to: data/processed.parquet
# Print df.shape and df['target'].value_counts(normalize=True) to confirm
