import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


data = pd.read_csv("/Users/minseokkang/.cache/kagglehub/datasets/abdelfattahibrahim/global-e-commerce-sales-dataset-20212024/versions/1/ecommerce_sales_dataset.csv")
print(data)

df = data
df.info()
#df.isnull().sum()
#df.duplicated().sum()
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

''' VALIDATING THE SYNTHETIC DATASET ACCURACY 
df['cal_profit_margin'] = df['Profit'] / df['Revenue'] * 100
# print(np.isclose(df['cal_profit_margin'], df['Profit_Margin_%'], atol = 1))
df['margin_diff'] = df['cal_profit_margin'] - df['Profit_Margin_%']
print(df['Order_Status'].value_counts())
print(df.groupby('Order_Status')[['Revenue', 'Profit']].mean())
print(df[df['Revenue'] > 2000][['Category', 'Sub_Category', 'Product_Name', 'Unit_Price', 'Quantity', 'Revenue']])
print(df[['Unit_Price', 'Shipping_Cost']].dtypes)
'''
df = pd.DataFrame({
    # categorial columns
    'seasons': data['Season'],
    'country': data['Country'],
    'region': data['Region'],
    'category': data['Category'],
    'sub_category': data['Sub_Category'],
    'shipping_methods': data['Shipping_Method'],
    'gender': data['Customer_Gender'],
    'customer_seg': data['Customer_Segment'],
    #numeric columns
    'unit_price': data['Unit_Price'],
    'shipping_cost': data['Shipping_Cost'],
    'month': data['Month'],
    'year': data['Year'],
    #taget
    'rev': data['Revenue'],
    'profit': data['Profit'],
    'Quantity': data['Quantity']
})

#feature engineering useful metrics to predict for drop shippers
df['unit_price_tier'] = pd.qcut(np.log(df['unit_price']), q=3, labels=['budget', 'mid', 'premium'])
numeric_cols = ['unit_price', 'shipping_cost', 'month', 'rev', 'profit']
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix — Revenue, Profit & Dropship Metrics')
plt.tight_layout()
plt.savefig('figures/corr_matrix.png')

df.to_parquet('data/processed.parquet')
print(df[['shipping_cost', 'rev']].corr())

print(pd.qcut(np.log(df['unit_price']), q=3, labels=['budget', 'mid', 'premium']).value_counts())
print(np.log(df['unit_price']).describe())

