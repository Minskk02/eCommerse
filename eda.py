import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('figures', exist_ok=True)

file_name = "ecommerce_sales_dataset.csv"
data = pd.read_csv(file_name)
data['Order_Date'] = pd.to_datetime(data['Order_Date'])
data = data.sort_values('Order_Date').reset_index(drop=True)

data['YearMonth'] = data['Order_Date'].dt.to_period('M')
data['YearQuarter'] = data['Order_Date'].dt.to_period('Q')


#various histograms
plt.figure()
fig, axes = plt.subplots(2, 3, figsize=(14,8))
axes[0, 0].hist(data['Revenue'], bins=100)
axes[0, 0].set_title('Revenue')
axes[0, 1].hist(data['Profit'], bins=100)
axes[0, 1].set_title('Profit')
axes[0, 2].hist(data['Cost'], bins=100)
axes[0, 2].set_title('Cost')
axes[1, 0].hist(data['Unit_Price'], bins=2500)
axes[1, 0].set_title('Unit_Price')
axes[1, 1].hist(data['Shipping_Cost'], bins=100)
axes[1, 1].set_title('Shipping_Cost')
axes[1, 2].hist(data['Quantity'], bins=10)
axes[1, 2].set_title('Quantity')
plt.savefig('figures/numeric_distributions.png')
plt.show()

print(data.groupby(data['Profit']<0)['Order_Status'].value_counts()) #even delivered items could leave them with neg profit
neg_profit = data[data['Profit'] < 0]
cols = ['Order_Status', 'Customer_Gender', 'Customer_Segment', 'Country', 'Category', 'Unit_Price'] # comparing columns
for col in cols:
    print(f"\n-----{col}-----")
    print(neg_profit[col].value_counts(normalize=True).mul(100).round(2))
# no gneder dependency, 70% of those who generates neg profit are regular and new customers, 
# (mexico, canada, usa) higher in neg profit but need to compare it with the population for true evaluation 
# 
# needs further analysis on the customer segment to compare the true influencing group
'''
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
neg_profit['Order_Status'].value_counts().plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Order_Status')
neg_profit['Customer_Gender'].value_counts().plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Customer_Gender')
neg_profit['Country'].value_counts().plot(kind='bar', ax=axes[0, 2])
axes[0, 2].set_title('Country')
neg_profit['Category'].value_counts().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Category')
axes[1, 1].hist(neg_profit['Unit_Price'], bins=30)
axes[1, 1].set_title('Unit_Price')
neg_profit['Customer_Segment'].value_counts().plot(kind='bar', ax=axes[1, 2])
axes[1, 2].set_title('Customer_Segment')
plt.tight_layout()
plt.savefig('figures/negative_profit_contributor.png')
plt.show()
'''
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
rate = data.groupby('Category').apply(lambda x: (x['Profit'] < 0).sum() / len(x))
rate.sort_values().plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Negative Profit Rate by Category')
rate1 = data.groupby('Country').apply(lambda x: (x['Profit'] < 0).sum() / len(x))
rate1.sort_values().plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Negative Profit Rate by Country')
axes[0, 2].scatter(data['Unit_Price'], data['Profit'], alpha=0.3, s=5)
axes[0, 2].axhline(0, color='red', linewidth=1)
axes[0, 2].set_title('Unit Price vs Profit')
rate3 = data.groupby('Order_Status').apply(lambda x: (x['Profit'] < 0).sum() / len(x))
rate3.sort_values().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Negative Profit Rate by Order status')
rate4 = data.groupby('Customer_Segment').apply(lambda x: (x['Profit'] < 0).sum() / len(x))
rate4.sort_values().plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Negative Profit Rate by Customer segment')
rate5 = data.groupby('Customer_Gender').apply(lambda x: (x['Profit'] < 0).sum() / len(x))
rate5.sort_values().plot(kind='bar', ax=axes[1, 2])
axes[1, 2].set_title('Negative Profit Rate by Customer gender')
plt.tight_layout()
plt.savefig('figures/negative_profit_contribution_rates.png')
plt.show()

#corr heatmap
sns.set_theme(style="dark")
numeric_cols = data[['Revenue', 'Profit', 'Cost', 'Unit_Price', 'Shipping_Cost', 'Quantity', 'Discount', 'Month']]
corr = numeric_cols.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.yticks(rotation=0) 
plt.tight_layout()
plt.savefig('figures/corr_matrix.png')
plt.show()



#sub category and category revenue/profit comparison
totRev = data.groupby('Sub_Category')['Revenue'].sum().nlargest()
profSum = data.groupby('Sub_Category')['Profit'].mean().nlargest()
meanMargin = data.groupby('Sub_Category')['Profit_Margin_%'].mean().nlargest()
f, ax = plt.subplots(1, 3, figsize=(18, 6))
totRev.plot(kind='barh', ax=ax[0])
ax[0].set_title('Total Revenue per Sub Category')
profSum.plot(kind='barh', ax=ax[1])
ax[1].set_title('Mean Profit per Sub Category')
meanMargin.plot(kind='barh', ax=ax[2])
ax[2].set_title('Mean Margin(%) per Sub Category')
plt.tight_layout()
plt.savefig('figures/RevAndProfit_per_Subcategory.png')
plt.show()

totRev = data.groupby('Category')['Revenue'].sum().nlargest()
profSum = data.groupby('Category')['Profit'].mean().nlargest()
meanMargin = data.groupby('Category')['Profit_Margin_%'].mean().nlargest()
f, ax = plt.subplots(1, 3, figsize=(18, 6))
totRev.plot(kind='barh', ax=ax[0])
ax[0].set_title('Total Revenue per Category')
profSum.plot(kind='barh', ax=ax[1])
ax[1].set_title('Mean Profit per Category')
meanMargin.plot(kind='barh', ax=ax[2])
ax[2].set_title('Mean Margin(%) per Category')
plt.tight_layout()
plt.savefig('figures/RevAndProfit_per_Category.png')
plt.show()


#regional
region_1 = data[data['Region'] == 'Asia'].groupby('Country')['Revenue'].sum().nlargest(10)
region_2 = data[data['Region'] == 'Europe'].groupby('Country')['Revenue'].sum().nlargest(10)
region_3 = data[data['Region'] == 'North America'].groupby('Country')['Revenue'].sum().nlargest(10)
region_4 = data[data['Region'] == 'Middle East'].groupby('Country')['Revenue'].sum().nlargest(10)

f, ax = plt.subplots(1, 4, figsize=(18, 6))
region_1.plot(kind='barh', ax=ax[0])
ax[0].set_title("Asia's top5 countries by revenue")
region_2.plot(kind='barh', ax=ax[1])
ax[1].set_title("EU's top5 countries by revenue")
region_3.plot(kind='barh', ax=ax[2])
ax[2].set_title("North America's top3 countries by revenue")
region_4.plot(kind='barh', ax=ax[3])
ax[3].set_title("Middle East's top5 countries by revenue")
plt.tight_layout()
plt.savefig('figures/Rev_by_Region.png')
plt.show()

#time series data visualisation
yearly = data.groupby('Year')[['Revenue', 'Profit']].sum()
monthly = data[data['Year']==2023].groupby('YearMonth')[['Revenue', 'Profit']].sum()
fig, ax = plt.subplots(1, 2, figsize=(12,6))
yearly[['Revenue']].plot(ax=ax[0], label='Revenue')
yearly[['Profit']].plot(ax=ax[0], label='Profit')
ax[0].set_title('Revenue & Profit yearly')
ax[0].set_xticks(yearly.index)
monthly[['Revenue']].plot(ax=ax[1], label='Revenue')
monthly[['Profit']].plot(ax=ax[1], label='Profit')
ax[1].set_title('Revenue & Profit by Month in 2023')
ax[1].set_xlabel("Months")
plt.tight_layout()
plt.savefig('figures/mean_yearly_and_monthly_revenue_profit.png')
plt.show()



#margin and risk analysis
data['Profit_Margin'] = data['Profit'] / data['Revenue'] * 100

fig, ax = plt.subplots(figsize=(18, 6))
data.boxplot(column='Profit_Margin', by='Product_Name', ax=ax)

ax.set_title('Profit Margin by Product Names')
ax.set_xlabel('Items')
ax.tick_params(axis='x', rotation=80)
ax.set_ylabel('Profit Margin (%)')
ax.axhline(0, color='red', linewidth=1, linestyle='--')  
plt.suptitle('') 
plt.tight_layout()
plt.savefig('figures/profit_margin_by_item.png')
plt.show()

margin_stats = data.groupby('Product_Name')['Profit_Margin'].agg(
    mean_margin='mean',
    risk='std'
)
##Sharpe ratio
margin_stats['score'] = margin_stats['mean_margin'] / (margin_stats['risk'] + 1e-9)
margin_stats = margin_stats.sort_values('score', ascending=False)
print(margin_stats)

fig, ax = plt.subplots(figsize=(10, 10))
margin_stats['score'].sort_values().plot(kind='barh', ax=ax)
ax.set_title('Risk-Adjusted Profit Score by Items')
ax.set_xlabel('mean margin / std (higher the better)')
plt.tight_layout()
plt.savefig('figures/risk_adjusted_score.png')
plt.show()

