import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

#split for train and testing
df = pd.read_parquet('data/processed.parquet')
X = df.drop(columns=['year', 'shipping_cost', 'rev', 'profit', 'Quantity', 'unit_price', 'unit_price_tier'])
train_df = df[df['year']<2024]
test_df = df[df['year']>2023]
'''
#start preprocessing by encoding categorical and numerical columns accordingly
preprocessor = ColumnTransformer(transformers=[
    ('ordinal', OrdinalEncoder(), ['month']),
    ('one-hot', OneHotEncoder(drop='first', sparse_output=False), ['seasons', 'region', 'shipping_methods', 'gender']),
    ('target_enc', TargetEncoder(smooth='auto'), ['country', 'category', 'sub_category', 'customer_seg'])
], remainder='drop')

y_train = train_df['Quantity']
y_test  = test_df['Quantity']
X_train = train_df.drop(columns=['year', 'shipping_cost', 'rev', 'profit', 'Quantity', 'unit_price', 'unit_price_tier'])
X_test  = test_df.drop(columns=['year', 'shipping_cost', 'rev', 'profit', 'Quantity', 'unit_price', 'unit_price_tier'])

#start revenue preprocess and predict
rev_pipeline = Pipeline([
    ('processor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
rev_pipeline.fit(X_train, y_train)
test_df['predicted_quantity'] = rev_pipeline.predict(X_test)

#validation
RMSE = root_mean_squared_error(y_test, test_df['predicted_quantity'])
r2   = r2_score(y_test, test_df['predicted_quantity'])
print(f"Quantity RMSE: {RMSE:.2f}")
print(f"Quantity R²:   {r2:.4f}")

#scoring system using the data-driven weight calculation using entropy weights
def entorpy_weights(matrix):
    norm = (matrix - matrix.min(axis=0))/ (matrix.max(axis=0) - matrix.min(axis=0) + 1e-9)
    p = norm / (norm.sum(axis=0) + 1e-9)
    k = 1 / (np.log(len(matrix) + 1e-9))
    entropy = -k * (p * np.log(p+ 1e-9)).sum(axis=0)
    divergence = 1 - entropy
    weights    = divergence / divergence.sum()
    return weights

matrix = test_df[['predicted_quantity', 'rev']]
weights = entorpy_weights(matrix)

print("Entropy-Derived Weights:")
print(f"  pred_quant: {weights.iloc[0]:.2%}")
print(f"  pred_rev: {weights.iloc[1]:.2%}")
test_df['entropy_dropship_score'] = matrix @ weights

#final prediction of overal dropship worthy itmes
result = test_df[['category', 'sub_category', 'country', 'Quantity', 'predicted_quantity']].copy()
result = result.sort_values('predicted_quantity', ascending=False)

#print(result.to_string(index=False))
result.to_csv('result.csv', index=False)
print(y_train.describe())
print(y_test.describe())
print(y_train.value_counts().sort_index())


'''
'''
print(y_test.describe())

feature_importance = rev_pipeline.named_steps['model'].feature_importances_
print(feature_importance)
feature_names = rev_pipeline.named_steps['processor'].get_feature_names_out()
print(feature_names)  # index of 0.34
print(X_train.columns.tolist())
feature_names = rev_pipeline.named_steps['processor'].get_feature_names_out()
for name, imp in zip(feature_names, feature_importance):
    print(f"{name}: {imp:.4f}")
    
    

print(y_train.describe())
print(y_test.describe())

print(X_train.columns.tolist())
print(X_test.columns.tolist())

sample = X_train.iloc[:5]
print(rev_pipeline.predict(sample))
print(y_train.iloc[:5].values)

from sklearn.dummy import DummyRegressor
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print(r2_score(y_test, dummy_pred))
'''

print(df.groupby(['seasons', 'sub_category'])['Quantity'].sum().sort_values(ascending=False))
print(df.groupby(['seasons', 'gender', 'region', 'sub_category'])['Quantity'].sum().sort_values(ascending=False).head(50))
print(df.groupby(['seasons', 'gender', 'region', 'sub_category'])[['Quantity', 'profit', 'rev']].agg({
    'Quantity': 'sum',
    'profit': 'mean',
    'rev': 'mean'
}).sort_values('profit', ascending=False).head(20))
agg =df.groupby(['seasons', 'gender', 'region', 'sub_category']).agg(
    total_quantity=('Quantity', 'sum'),
    avg_profit=('profit', 'mean'),
    avg_rev=('rev', 'mean')
).reset_index()


#scoring system using the data-driven weight calculation using entropy weights
def entorpy_weights(matrix):
    norm = (matrix - matrix.min(axis=0))/ (matrix.max(axis=0) - matrix.min(axis=0) + 1e-9)
    p = norm / (norm.sum(axis=0) + 1e-9)
    k = 1 / (np.log(len(matrix) + 1e-9))
    entropy = -k * (p * np.log(p+ 1e-9)).sum(axis=0)
    divergence = 1 - entropy
    weights    = divergence / divergence.sum()
    return weights

matrix = agg[['total_quantity', 'avg_profit']]
weights = entorpy_weights(matrix)

print("Entropy-Derived Weights:")
print(f"quantity: {weights.iloc[0]:.2%}")
print(f"average_profit: {weights.iloc[1]:.2%}")
test_df['entropy_dropship_score'] = matrix @ weights


agg['dropship_score'] = matrix @ weights
ranked = agg['sub_category'].rank(ascending=True)
print(agg.sort_values('dropship_score', ascending=False).head(50))

ave_q = df.groupby(['sub_category', 'seasons', 'gender', 'region'])['Quantity'].mean()
print(ave_q.rank())

query = agg[(agg['seasons'] == 'Summer') & 
            (agg['gender'] == 'Male') & 
            (agg['region'] == 'Asia')]

print(query.sort_values('dropship_score', ascending=False))