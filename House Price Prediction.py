#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[5]:


# Load the California Housing dataset
california_housing = fetch_california_housing(as_frame=True)
df = california_housing['data']
df['PRICE'] = california_housing['target']
print((df))


# In[6]:


df.head()


# In[7]:


# Split the data into features (X) and target variable (y)
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Create a linear regression model
model = LinearRegression()


# In[9]:


# Train the model
model.fit(X_train, y_train)


# In[10]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[11]:


# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[12]:


# Print the evaluation results
print("Mean Squared Error:", mse)
print("Coefficient of Determination (R^2):", r2)


# In[ ]:




