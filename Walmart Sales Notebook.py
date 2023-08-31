# Databricks notebook source
import pyspark
import pandas as pd
import numpy as np
import pyspark.pandas as ps
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings

#visualization
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly import express as px
import plotly.figure_factory as ff

warnings.filterwarnings('ignore')
%matplotlib inline
ps.set_option('compute.ops_on_diff_frames', True)


# COMMAND ----------

df = ps.read_csv('/FileStore/tables/Walmart_Store_sales.csv')
df.head(10)

# COMMAND ----------

#missing data exploration
df.isnull().sum()

# COMMAND ----------

df.describe().T

# COMMAND ----------

df['Date'] = df['Date'].astype('datetime64')



# COMMAND ----------

df.dtypes

# COMMAND ----------

df.head(10)

# COMMAND ----------

sales = df.groupby('Store').agg({'Weekly_Sales':'sum'}).reset_index().sort_values(by=['Weekly_Sales'],ascending=False).astype(int)
print (sales)

# COMMAND ----------

# Group the data by Store and calculate the sum of Weekly_Sales
sales = df.groupby('Store').agg({'Weekly_Sales':'sum'}).reset_index().sort_values(by=['Weekly_Sales'],ascending=False).astype(int)
sales['Store']=sales['Store'].astype(str)

# Get the store with the highest sales
top_store = sales.head(1)


# Create the bar chart using Plotly Express
fig = go.Figure(data=go.Bar(x=sales.Store.to_list(), y=sales.Weekly_Sales.to_list()))
fig.update_layout(
    title='Total Weekly Sales by Store',
    xaxis_title='Store',
    yaxis_title='Sales',
    plot_bgcolor='rgb(255, 255, 255)'
)
colors = ['rgb(0, 115, 0)' if store == sales.Store.iloc[0] else 'rgb(100, 115, 0)'for store in sales.Store.iloc[0]]
fig.update_traces(marker_color = colors)  # Set the color of the bars
fig.show()

# COMMAND ----------

x=sales.Store.to_list()
x[:5]

# COMMAND ----------

sales.info()

# COMMAND ----------

df.info()

# COMMAND ----------

#Which Store has maximum sales?
sales = df.groupby('Store')['Weekly_Sales'].sum().round().sort_values(ascending=False)
sales.head(1)


# COMMAND ----------

#Which store has maximum standard deviation i.e., the sales vary a lot. Also, find out the coefficient of mean to standard deviation?
#max sd
sd = df.groupby('Store').agg({'Weekly_Sales':'std'}).round(3).reset_index().sort_values(by=['Weekly_Sales'],ascending=False).astype(int)
sd

sd.Store = sd.Store.astype('str')



#visualization


fig = go.Figure(data=go.Scatter(x=sales.Store.to_list(), y=sales.Weekly_Sales.to_list(), mode='markers'))

fig.update_layout(
    title='Dispersion of Weekly Sales',
    xaxis_title='Store',
    yaxis_title='Standard Deviation',
    plot_bgcolor='rgb(255, 255, 255)'
)

fig.show()

# COMMAND ----------

#mean to sd
store14 = df[df.Store == 14].Weekly_Sales
mean_to_sd = store14.std()/store14.mean()*100
print(format(mean_to_sd, '.2f'))

# COMMAND ----------

#Which store/s has good quarterly growth rate in Q3’2012 ?
q2_sales = df[(df['Date'] >= '2012-04-01') & (df['Date'] <= '2012-06-30')].groupby('Store')['Weekly_Sales'].sum().round()
q3_sales = df[(df['Date'] >= '2012-07-01') & (df['Date'] <= '2012-09-30')].groupby('Store')['Weekly_Sales'].sum().round()

# COMMAND ----------

#q3 growth rate question ^^
new = q3_sales - q2_sales

#new = new.astype('str')
new = new.reset_index().head(10)
new
#since all negatives, there had been no growth in 2012

# COMMAND ----------

fig = go.Figure(data=go.Scatter(x=new.Store.to_list(), y=new.Weekly_Sales.to_list(), mode='lines'))

fig.update_layout(
    title='Quarterly Sales Difference',
    xaxis_title='Store',
    yaxis_title='Sales Difference',
    plot_bgcolor='rgb(255, 255, 255)'
)

fig.show()

# COMMAND ----------

#Some holidays have a negative impact on sales. Find out holidays which have higher sales than the mean sales in non-holiday season for all stores together?

#Super_Bowl: '2010-02-12', '2011-02-11', '2012-02-10'
#Labour_Day: '2010-09-10', '2011-09-09', '2012-09-07'
#Thanksgiving: '2010-11-26', '2011-11-25', '2012-11-23'
#Christmas: '2010-12-31', '2011-12-30', '2012-12-28'

Super_Bowl_sales = df[df['Date'] == ('2010-02-12' or  '2011-02-11' or '2012-02-10')]['Weekly_Sales'].mean()
Labour_Day_sales = df[df['Date']== ('2010-09-10' or  '2011-09-09' or '2012-09-07')]['Weekly_Sales'].mean()
Thanksgiving_sales = df[df['Date']== ('2010-11-26' or  '2011-11-25' or '2012-11-23')]['Weekly_Sales'].mean()
Christmas_sales = df[df['Date']== ('2010-12-31' or  '2011-12-30' or '2012-12-28')]['Weekly_Sales'].mean()

Super_Bowl_sales_r = format(Super_Bowl_sales, '.2f')
Labour_Day_sales_r = format(Labour_Day_sales, '.2f')
Thanksgiving_sales_r = format(Thanksgiving_sales, '.2f')
Christmas_sales_r = format(Christmas_sales, '.2f')

Super_Bowl_sales_r, Labour_Day_sales_r, Thanksgiving_sales_r, Christmas_sales_r


# COMMAND ----------

non_holiday_sales = df[df['Holiday_Flag'] == 0]['Weekly_Sales'].mean()
non_holiday_sales_rounded = format(non_holiday_sales, '.2f')

non_holiday_sales_rounded

# COMMAND ----------

# Which Holiday has the most weekly sales?


result = pd.DataFrame([{'Super Bowl Sales':Super_Bowl_sales_r,
              'Labour Day Sales':Labour_Day_sales_r,
              'Thanksgiving Sales':Thanksgiving_sales_r,
              'Christmas Sales':Christmas_sales_r,
              'Non Holiday Sales':non_holiday_sales_rounded}]).T

result

#Thanksgiving has the most weekly sales

# COMMAND ----------

#visualization

fig = go.Figure(data=go.Bar(x=result.Holiday_Sales.to_list(), y= result.sales.to_list()))
fig.update_layout(
    title='Sales_per_Holiday',
    xaxis_title='Sales',
    yaxis_title='Holiday_Sales',
    plot_bgcolor='rgb(200, 210, 235)'
)

# COMMAND ----------

#Feature Selection/Preprocessing 
#Our target is to forecast the weekly sales



#Step1 : Remove the target column from the data-set
target = 'Weekly_Sales'

features = [i for i in df.columns if i != 'Weekly_Sales']
print(features)
#keeping a copy of the orignal dataset
df_copy = df.copy(deep = True)

df.head()

# COMMAND ----------

#Visualize our Target


#normally distributed data_columns
data = np.log(df['Weekly_Sales'])

# Create distplot with custom bin_size
fig = go.Figure()


fig.add_trace(go.Histogram(x=data.tolist(), nbinsx = 30, marker=dict(color='green'), opacity=0.7))

fig.update_layout(
    title='Target Variable Distribution - Median Value of Homes ($1Ms)',
    xaxis_title='Value',
    yaxis_title='Count',
    bargap=0.2,
    plot_bgcolor='rgb(255, 255, 255)'
)

fig.show()

# COMMAND ----------

#Visualize our Features
# ['Store', 'Date', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']


# COMMAND ----------

#Holiday_Flag column
df['Holiday_Flag'] = df['Holiday_Flag'].astype('category')
category_flag = df.Holiday_Flag.value_counts()
category_flag
#visulaize Holiday Flag category columnn
# Create the bar chart

colors = ['lightslategray',] 
colors[0] = 'crimson'
fig = go.Figure(go.Bar(x=['0','1'], y =category_flag.to_list(), marker_color=colors))


# Update the layout
fig.update_layout(
    title='Holiday_Flag Bar Chart',
    xaxis_title='Categories',
    yaxis_title='count',
    plot_bgcolor='rgb(255, 255, 255)'
)

fig.show()

# COMMAND ----------

# name of the dataset

#normally distributed data_columns
data1 = df['Temperature']
data2 = df['Fuel_Price']
data3 = df['CPI']
data4 = df['Unemployment']


from plotly.figure_factory import create_distplot
fig = go.Figure()
fig.add_trace(go.Histogram(x=data1.tolist(), nbinsx = 30, marker=dict(color='green'), opacity=0.7))
fig.add_trace(go.Histogram(x=data2.tolist(), nbinsx = 30, marker=dict(color='red'), opacity=0.7))
fig.add_trace(go.Histogram(x=data3.tolist(), nbinsx = 30, marker=dict(color='blue'), opacity=0.7))
fig.add_trace(go.Histogram(x=data4.tolist(), nbinsx = 30, marker=dict(color='yellow'), opacity=0.7))



