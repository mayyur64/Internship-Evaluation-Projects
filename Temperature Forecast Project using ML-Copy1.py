#!/usr/bin/env python
# coding: utf-8

# # Temperature Forecast Project using ML

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/Dataset2/main/temperature.csv')


# In[3]:


df


# In[4]:


df.info()


# In[5]:


#Here we have datatypes float and object combined


# In[6]:


#By checking for null values ,which needs to be cleared 


# In[7]:


df.isnull().sum()


# In[8]:


df.loc[7750, 'station'] = 1.0
df.loc[7751, 'station'] = 2.0


# In[9]:


df.iloc[7750, df.columns.get_loc('Date')] = '31-08-2017'
df.iloc[7751, df.columns.get_loc('Date')] = '31-08-2017'


# In[10]:


df.drop(columns=['lat','lon'],inplace=True)


# In[11]:


for col in df.columns:
    if col not in ['station', 'Date']:
        df[col] = df[col].replace(np.nan, np.nanmedian(df[col]))


# In[12]:


df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')


# In[13]:


df_corr=df.corr(numeric_only=True)
plt.figure(figsize=(25,15))
sns.heatmap(df_corr,vmin=-1,vmax=1,annot=True,square=True,center=0,fmt='.2g',linewidths=0.1)
plt.tight_layout()


# In[14]:


df.skew


# In[15]:


#next step is to Remove outliers 


# In[16]:


from scipy.stats import zscore

z_scores = zscore(df[['Present_Tmax', 'Present_Tmin', 'LDAPS_RHmax', 'LDAPS_Tmax_lapse', 'LDAPS_Tmin_lapse', 'LDAPS_WS', 'LDAPS_LH', 'DEM', 'Slope']])

abs_z_scores = np.abs(z_scores)

filtering_entry = (abs_z_scores < 3).all(axis=1)

df = df[filtering_entry]

# Reset the index of the DataFrame
df.reset_index(inplace=True)


# In[17]:


from sklearn.preprocessing import OrdinalEncoder
enc=OrdinalEncoder()


# In[18]:


df['Date']=enc.fit_transform(df['Date'].values.reshape(-1,1))


# In[19]:


x=df.drop(['Next_Tmax','Next_Tmin'],axis=1)
y=df[['Next_Tmax','Next_Tmin']]


# In[20]:


import numpy as np


for col in x.skew().index:
    if x.skew().loc[col] > 0.5:
        x[col] = np.cbrt(x[col])  
    elif x.skew().loc[col] < -0.5:
        x[col] = np.square(x[col]) 


# In[21]:


x.skew()


# In[22]:


scalar=StandardScaler()
X=scalar.fit_transform(x)
X=pd.DataFrame(X,columns=x.columns)


# In[23]:


#Applied linear regression, finding the maximum R2 score and corresponding random state using train-test split


# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

max_r_score = 0
r_state = 0
for i in range(1, 200):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    r2_scr = r2_score(y_test, y_pred)
    if r2_scr > max_r_score:
        max_r_score = r2_scr
        r_state = i

# Print the maximum R2 score and the corresponding random state
print("Max R2 score is", max_r_score, "on random state", r_state)


# In[25]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=2)


# In[26]:


#Extra Trees Regressor to predict target variable, computing R2 score to evaluate model performance on the dataset


# In[27]:


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

extra_trees = ExtraTreesRegressor()
extra_trees.fit(x_train, y_train)
y_pred_train = extra_trees.predict(x_train)

y_pred_test = extra_trees.predict(x_test)
r2 = r2_score(y_test, y_pred_test) * 100


# In[28]:


grid_params={'max_depth':[8,9,10,12,15,20],
            'n_estimators':[500,700,1000,1200],
            'min_samples_split':[2,3]}


# In[ ]:


GCV=GridSearchCV(ExtraTreesRegressor(),grid_params,cv=5)
GCV.fit(x_train,y_train)


# In[ ]:


model_1=ExtraTreesRegressor(max_depth=20,n_estimators=500,min_samples_split=2)
model_1.fit=(x_train,y_train)
y_pred=model_1.predict(x_train)
pred_model1_predict(x_test)


# In[ ]:




