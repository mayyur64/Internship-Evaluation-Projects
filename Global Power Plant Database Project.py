#!/usr/bin/env python
# coding: utf-8

# # Global Power Plant Database ML Project

# Project Overview:
# 
# Establishing an accessible, dependable, and eco-friendly power sector is imperative for societal progress. Policy decisions, corporate strategies, and utility operations greatly influence and rely on the power sector's dynamics. Variables such as carbon pricing policies, plant operations, and the integration of new facilities contribute to changes in the electricity generation mix, system reliability, and environmental impact. The Global Power Plant Database, a comprehensive collection of grid-scale electricity generating facilities worldwide, currently encompasses approximately 35,000 power plants in 167 countries. Our focus will be on exploring the Indian dataset, which includes 908 entries detailing various power plants across the country, facilitating a comprehensive understanding of India's energy landscape.

# 
# Features of dataset:
# 
# country: symbolic country Name
# 
# country_long: Full country Name
# 
# name : Name of the Power Plant
# 
# gppd_idnr : 10-12 character type ID of the power plant
# 
# capacity_mw : Electricity generating capacity in megawatts
# 
# latitude : Geo location of plant in decimal degerees
# 
# longitude : Geo location of plant in decimal degerees
# 
# primary_fuel : Primary fuel used for electricity genrration.
# 
# other_fuel1 : Energy source used in electricity generation or export
# 
# other_fuel2 : Energy source used in electricity generation or export
# 
# other_fuel3 : Energy source used in electricity generation or export
# 
# commissioning_year: year of opertaion of power plant or when the power plant start.
# 
# owner : Majority shareholder of the power plant
# 
# source: Entity reporting the data
# 
# url : Web document corresponding to the sourcefield
# 
# geolocation_source :Attribution for geolocation information
# 
# wepp_id : A reference to a unique plant identifier in the widely-used PLATTS-WEPP database.
# 
# year_of_capacity_data: year the capacity information was reported
# 
# generation_gwh_2013 : electricity generation in gigawatt-hours reported for the year 2013
# 
# generation_gwh_2014 : electricity generation in gigawatt-hours reported for the year 2014
# 
# generation_gwh_2015 : electricity generation in gigawatt-hours reported for the year 2015
# 
# generation_gwh_2016 : electricity generation in gigawatt-hours reported for the year 2016
# 
# generation_gwh_2017 : electricity generation in gigawatt-hours reported for the year 2017
# 
# generation_data_source : electricity generation in gigawatt-hours reported for the year 2014
# estimated_generation_gwh : attribution for the reported generation information
# 

# In[36]:


import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  GradientBoostingRegressor


# In[37]:


data= pd.read_csv('https://raw.githubusercontent.com/wri/global-power-plant-database/master/source_databases_csv/database_IND.csv')


# In[38]:


pd.set_option('display.max_columns', None)


# In[39]:


data


# In[40]:


data.nunique().to_frame("Unique Values")


# Utilizing the 'nunique' method, we evaluated the distinct data in our dataset columns. Scrutinizing 'country' and 'country_long', we noticed a singular entry in all 908 rows, lacking significant insights. Correspondingly, 'year_of_capacity_data' also holds a single value and missing data, rendering it non-informative. Similarly, 'name' and 'gppd_idnr' comprising unique identifiers across the dataset would not contribute to building a machine learning model and are slated for removal. Additionally, the 'url' column containing web document links and descriptive values irrelevant to machine learning models will also be eliminated. The targeted columns for removal include:
# 
#     country
#     country_long
#     year_of_capacity_data
#     name
#     gppd_idnr
#     url.

# In[41]:


data.isna().sum()


# In[42]:


plt.figure(figsize=(15,10))
sns.heatmap(data.isnull())
plt.show()


# In[43]:


object_datatype = []
for x in data.dtypes.index:
    if data.dtypes[x] == 'object':
        object_datatype.append(x)
print(f"Object Data Type Columns are: ", object_datatype)


# getting the list of float data type column names
float_datatype = []
for x in data.dtypes.index:
    if data.dtypes[x] == 'float64':
        float_datatype.append(x)
print(f"Float Data Type Columns are: ", float_datatype)


# In[44]:


data.drop(columns = ['other_fuel2', 'other_fuel3','wepp_id','estimated_generation_gwh'], axis = 1, inplace = True)


# In[45]:


data


# In[46]:


data.describe()


# In[47]:


data.drop(columns = ['year_of_capacity_data'], inplace = True)


# now dropping year of capicity data from dataset

# In[48]:


data['primary_fuel'].hist(grid = False)
plt.title('primary_fuel')
plt.show()


# In[49]:


data['capacity_mw'].hist(grid = False)
plt.title(' capacity_mw ')
plt.show()


# In[50]:


fig, ax = plt.subplots(1, 1, figsize=(55, 12))
sns.boxplot(data = data, ax=ax)
plt.show()


# In[51]:


data.hist(figsize = (30,20), grid = False)


# The data is now less skewed and more towards normal distribution.

# In[52]:


#Correlation Matrix

corr_mat = data.corr()
m = np.array(corr_mat)
m[np.tril_indices_from(m)] = False

fig = plt.gcf()
fig.set_size_inches(30,24)
sns.heatmap(data = corr_mat, mask = m, square = True, annot = True, cbar = True)


# In[53]:


plt.figure(figsize = (30,35))
graph = 1

for column in data:
  if graph<=40:
    ax = plt.subplot(8,5,graph)
    sns.scatterplot(x = data[column], y = 'capacity_mw', data = data)
    plt.xlabel(column, fontsize = 20)
  graph+=1
plt.show()


# In[54]:


plt.figure(figsize = (30,35))
graph = 1

for column in data:
  if graph<=40:
    ax = plt.subplot(8,5,graph)
    sns.scatterplot(x = data[column], y = 'primary_fuel', data = data)
    plt.xlabel(column, fontsize = 20)
  graph+=1
plt.show()


# In[55]:


data.skew()


# In[56]:


fig,axes=plt.subplots(1,1,figsize=(15,15))
sns.scatterplot(x='primary_fuel',y='capacity_mw',hue='geolocation_source',data=data)


# In[57]:


data['generation_data_source'].value_counts()


# In[58]:


data.drop(columns = ['generation_data_source'], axis = 1, inplace = True)


# In[59]:


sns.pairplot(data, hue = 'primary_fuel')


# In[60]:


from sklearn.impute import KNNImputer
knn_ipm = KNNImputer(n_neighbors = 3)
data_filled = knn_ipm.fit_transform(data[['latitude', 'longitude', 'commissioning_year', 'generation_gwh_2013', 'generation_gwh_2014','generation_gwh_2015', 'generation_gwh_2016','generation_gwh_2017']])

data1 = pd.DataFrame(data_filled)


# In[61]:


object_datatype = []
for x in data.dtypes.index:
    if data.dtypes[x] == 'object':
        object_datatype.append(x)
print(f"Object Data Type Columns are: ", object_datatype)


# getting the list of float data type column names
float_datatype = []
for x in data.dtypes.index:
    if data.dtypes[x] == 'float64':
        float_datatype.append(x)
print(f"Float Data Type Columns are: ", float_datatype) 


# In[62]:


# filling missing data for continous values with mean
data["latitude"].fillna(data["latitude"].mean(),inplace=True)
data["longitude"].fillna(data["longitude"].mean(),inplace=True)

# filling missing data for categorical values with mode
data["commissioning_year"].fillna(data["commissioning_year"].mode()[0],inplace=True)
data["geolocation_source"].fillna(data["geolocation_source"].mode()[0],inplace=True)


# In[63]:


data['country'].value_counts()


# In[64]:


data['country_long'].value_counts()


# In[65]:


data['name'].value_counts()


# In[66]:


data['gppd_idnr'].value_counts()


# In[68]:


for col in object_datatype:
    print(col)
    print(data[col].value_counts())
    print("="*120)


# In[69]:


data.drop(columns = ['gppd_idnr','name', 'country_long','country'], axis = 1, inplace = True)


# In[70]:


#Treating the outliers

# findingout the quantile of data with continuous columns
col = data.drop(columns = ['primary_fuel','other_fuel1','owner','source','url','geolocation_source'])
Q1 = col.quantile(0.25)
Q3 = col.quantile(0.75)
IQR = Q3 - Q1
# REMOVING OUTLIERS USING IQR METHOD
data_new = col[~((col < (Q1 -  1.5*IQR)) |(col > (Q3 +  1.5*IQR))).any(axis=1)]
print("shape before and after")
print("shape before".ljust(20),":", col.shape)
print("shape after".ljust(20),":", data_new.shape)
print("Percentage Loss".ljust(20),":", (col.shape[0]-data_new.shape[0])/col.shape[0])


# In[71]:


from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()


# In[72]:


a = lab_enc.fit_transform(data['primary_fuel'])
b = lab_enc.fit_transform(data['other_fuel1'])
c = lab_enc.fit_transform(data['owner'])
d = lab_enc.fit_transform(data['source'])
e = lab_enc.fit_transform(data['url'])
f = lab_enc.fit_transform(data['geolocation_source'])


# In[73]:


data['primary_fuel'] = a
data['other_fuel1'] = b
data['owner'] = c
data['source'] = d
data['url'] = e 
data['geolocation_source'] = f
data


# In[74]:


data.hist(figsize = (30,20), grid = False)


# Data is less skewed now

# In[75]:


plt.style.use('bmh')
g = sns.pairplot(data)
for ax in g.axes.flat:
    ax.tick_params("x", labelrotation=90)
plt.show()


# # Model with the target variable 'primary_fuel'

# In[77]:


le = LabelEncoder()
data["primary_fuel"] = le.fit_transform(data["primary_fuel"])
data.head()


# In[78]:


upper_triangle = np.triu(data.corr())
plt.figure(figsize=(10,7))
sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True, square=True, fmt='0.3f', 
            annot_kws={'size':10}, cmap="cubehelix", mask=upper_triangle)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# Label Encoder to the target column "primary_fuel" to transform the object data type into a numerical data type for further analysis.

# In the provided heatmap, it's evident that the target labels, "capacity_mw" and "primary_fuel," exhibit both positive and negative correlations with the other feature columns. The observed multicollinearity is minimal, indicating no significant concerns. As a result, I plan to retain these interdependent features.

# In[79]:


plt.style.use('dark_background')
data_corr = data.corr()
plt.figure(figsize=(10,5))
data_corr['primary_fuel'].sort_values(ascending=False).drop('primary_fuel').plot.bar()
plt.title("Correlation of Features vs Classification Label\n", fontsize=16)
plt.xlabel("\nFeatures List", fontsize=14)
plt.ylabel("Correlation Value", fontsize=12)
plt.show()



# In[80]:


data_corr = data.corr()
plt.figure(figsize=(10,5))
data_corr['capacity_mw'].sort_values(ascending=False).drop('capacity_mw').plot.bar()
plt.title("Correlation of Features vs Regression Label\n", fontsize=16)
plt.xlabel("\nFeatures List", fontsize=14)
plt.ylabel("Correlation Value", fontsize=12)
plt.show()


# In[81]:


for col in float_datatype:
    if data.skew().loc[col]>0.55:
        data[col]=np.log1p(data[col])


# In[82]:


X = data.drop('primary_fuel', axis=1)
Y = data['primary_fuel']


# In[83]:


Y.value_counts()


# In[84]:


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X.head()


# # create a machine learning model for classification, we'll utilize appropriate evaluation metrics. 

# In[85]:


def classify(model, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=45)
    
    model.fit(X_train, Y_train)
    
    pred = model.predict(X_test)
    
    class_report = classification_report(Y_test, pred)
    print("\nClassification Report:\n", class_report)
    
    acc_score = (accuracy_score(Y_test, pred))*100
    print("Accuracy Score:", acc_score)
    
    cv_score = (cross_val_score(model, X, Y, cv=5).mean())*100
    print("Cross Validation Score:", cv_score)
    
    result = acc_score - cv_score
    print("\nAccuracy Score - Cross Validation Score is", result)


# In[86]:


fmod_param = {'criterion' : ["gini", "entropy"],
              'n_jobs' : [2, 1, -1],
              'min_samples_split' : [2, 3, 4],
              'max_depth' : [20, 25, 30],
              'random_state' : [42, 45, 111]
             }


# In[87]:


X = data.drop('capacity_mw', axis=1)
Y = data['capacity_mw']


# In[88]:


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X.head() 


# In[ ]:




