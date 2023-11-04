#!/usr/bin/env python
# coding: utf-8

# # Loan Application Status Prediction Project

# Project focuses on predicting whether a loan will be approved or not based on customer's profile
# Key features used in prediction include education, applicant income, and credit history

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv")


# In[4]:


df


# The features include applicant's marital status, number of dependents, education level, employment type, income, loan amount, loan term, credit history, and property area.
# The target variable 'loan_status' indicates whether the loan application was approved or not.

# In[5]:


df.head()


# In[7]:


df.tail()


# In[9]:


df.size


# In[10]:


df.describe()


# Generates statistical summary including count, mean, and range for each numerical column in the DataFrame 'df'."
# Summary of numeric columns: count, mean, standard deviation, minimum, maximum, and quartile values for each variable."

# In[11]:


df.info()


# In[12]:


df.isnull().sum()


# The code detects and tallies the quantity of absent values in each column of the DataFrame 'df' using isnull().

# In[13]:


df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())


# The code populates the absent numerical values in 'LoanAmount', 'Loan_Amount_Term', and 'Credit_History' columns with their means.

# In[15]:


df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])


# The script fills the missing categorical values in 'Gender,' 'Married,' 'Dependents,' and 'Self_Employed' columns with their respective modes, the most frequently occurring values within each category

# In[16]:


df.isnull().sum()


# There are no missing values in the dataset for any of the listed columns, as the count of null values for each feature is zero.

# In[22]:


df['Gender'] = df['Gender'].astype('category')


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")  
plt.figure(figsize=(6, 4))  

sns.countplot(x='Gender', data=df)  # Plot the count plot for 'Gender'

plt.show()  # Display the plot


# This script imports the essential libraries, sets the plot's visual style and size, generates a count plot for the 'Gender' attribute from the DataFrame 'df,' and finally displays the distribution of genders using Seaborn and Matplotlib.

# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns

df['Married'] = df['Married'].astype('category')

sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))

sns.countplot(x='Married', data=df)

plt.show()


# In[29]:


df['Dependents'] = df['Dependents'].astype('category')

sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))

sns.countplot(x='Dependents', data=df)

plt.show()


# In[30]:


df['Education'] = df['Education'].astype('category')

sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))

sns.countplot(x='Education', data=df)


plt.show()


# In[31]:


df['Self_Employed'] = df['Self_Employed'].astype('category')

sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))

sns.countplot(x='Self_Employed', data=df)
plt.show()


# In[32]:


df['Property_Area'] = df['Property_Area'].astype('category')

sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))

sns.countplot(x='Property_Area', data=df)
plt.show()


# In[33]:


df['Loan_Status'] = df['Loan_Status'].astype('category')

sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))

sns.countplot(x='Loan_Status', data=df)

plt.show()


# The next step is to visualize the distribution of the 'ApplicantIncome' numerical attribute using Seaborn's distplot, providing insights into its data distribution and statistical properties.

# In[34]:


sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.histplot(df["ApplicantIncome"], kde=True)
plt.show()


# <AxesSubplot:xlabel='ApplicantIncome'>

# In[35]:


sns.distplot(df["CoapplicantIncome"])


# In[36]:


sns.distplot(df["LoanAmount"])


# In[37]:


sns.distplot(df['Loan_Amount_Term'])


# In[38]:


sns.distplot(df['Credit_History'])


# Next steps involve the creation of new attributes. The script creates the 'Total_Income' column by summing the 'ApplicantIncome' and 'CoapplicantIncome' columns, enhancing data for further analysis and insights.

# In[39]:


df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()


# Log transformation is applied to 'ApplicantIncome'. The code creates 'ApplicantIncomeLog', enabling better normalization for improved analysis

# In[41]:


df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome']+1)
sns.distplot(df["ApplicantIncomeLog"])


# In[42]:


df['CoapplicantIncomeLog'] = np.log(df['CoapplicantIncome']+1)
sns.distplot(df["CoapplicantIncomeLog"])


# In[43]:


df['LoanAmountLog'] = np.log(df['LoanAmount']+1)
sns.distplot(df["LoanAmountLog"])


# In[44]:


df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term']+1)
sns.distplot(df["Loan_Amount_Term_Log"])


# In[45]:


df['Total_Income_Log'] = np.log(df['Total_Income']+1)
sns.distplot(df["Total_Income_Log"])


# The provided script below computes a correlation matrix to visualize the connections among different attributes. Utilizing a heatmap plot with the 'BuPu' color map, the code illustrates correlation values, enabling the identification of attribute relationships for effective feature selection and predictive modeling insights

# In[46]:


corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot = True, cmap="BuPu")


# In[48]:


df.head()


# In[49]:


cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Total_Income", 'Loan_ID', 'CoapplicantIncomeLog']
df = df.drop(columns=cols, axis=1)
df.head()


# In[50]:


from sklearn.preprocessing import LabelEncoder
cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status","Dependents"]
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])


# In[51]:


df.head()


# In[52]:


X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# The script separates the DataFrame into predictor and target variables. Using train_test_split from sklearn, it divides data into training and testing sets for machine learning model development."

# In[53]:


from sklearn.model_selection import cross_val_score
def classify(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print("Accuracy is", model.score(x_test, y_test)*100)
    score = cross_val_score(model, x, y, cv=5)
    print("Cross validation is",np.mean(score)*100)


# The script imports cross_val_score from sklearn.model_selection. The classify function divides data into training and testing sets, fits the model, and computes accuracy. Additionally, it calculates cross-validation scores using a 5-fold strategy, providing insights into model performance and consistency across datasets."

# In[54]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)


# Logistic Regression model from sklearn to predict and classify the dataset. By employing the custom 'classify' function, it evaluates the model's accuracy on the test set and employs a 5-fold cross-validation method to assess the model's performance and consistency across various datasets.

# In[55]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model, X, y)


#  Decision Tree Classifier from sklearn to predict and classify the dataset. Utilizing the custom 'classify' function, it evaluates the model's accuracy on the test set and utilizes a 5-fold cross-validation approach to assess the model's performance and consistency across different datasets.

# In[56]:


from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
model = RandomForestClassifier()
classify(model, X, y)


# imports RandomForestClassifier and ExtraTreesClassifier from sklearn.ensemble. It initializes a RandomForestClassifier model and applies the 'classify' function to assess the model's accuracy on the test set and evaluate its performance using a 5-fold cross-validation strategy across diverse datasets.

# In[57]:


model = ExtraTreesClassifier()
classify(model, X, y)


# The script initializes an ExtraTreesClassifier model and utilizes the 'classify' function to evaluate its accuracy on the test set. Additionally, it assesses the model's performance using a 5-fold cross-validation strategy, providing insights into its consistency across different datasets.

# In[58]:


model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=1)
classify(model, X, y)


# In[59]:


model = RandomForestClassifier()
model.fit(x_train, y_train)


# In[60]:


from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# A confusion matrix is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.

# In[61]:


sns.heatmap(cm, annot=True)


# Algorithms used-
# Logistic Regression
# Decision Tree
# Random Forest
# Extra Tress
# Best Model Accuracy: 81.00

# In[ ]:




