#!/usr/bin/env python
# coding: utf-8

# # Importing all libraries required

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression


# In[2]:


## Reading the dataset
url = 'http://bit.ly/w-data'
st_data = pd.read_csv(url)

st_data.head()


# # Checking data

# In[3]:


st_data.info()


# In[4]:


st_data.describe(include='all')


# In[5]:


### So data has no null values as we can see using info and describe method


# In[6]:


## Declare variables
x= st_data['Hours']
y= st_data['Scores']


# In[7]:


## From a scatter plot between Hours and Scores
plt.scatter(x, y)
plt.title('Hours vs Percentage')
plt.xlabel('Hours', fontsize=18)
plt.ylabel('Scores', fontsize=18)
plt.show()


# In[8]:


### we can see from the graph that there is a linear relationship between study hours and scores
### means there is linearity in data and there is no other feature, so we can perform linear regression


# In[9]:


## Reshape x and y with reshape method
X = x.values.reshape(-1,1)
Y = y.values.reshape(-1,1)


# # Spliting dataset into training and testing data

# In[10]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=69)


# # Performing regression

# In[11]:


st_reg = LinearRegression()
st_reg.fit(x_train, y_train)
print('Model Training Completed')


# In[12]:


y_hat = st_reg.coef_*X + st_reg.intercept_

plt.scatter(X,Y)
plt.plot(X,y_hat, c='red', lw=3)
plt.xlabel('Hours', fontsize=15)
plt.ylabel('Scores', fontsize=15)
plt.show()


# In[13]:


### visually we can see that regression line is fitting the data quite well


# # Predicting values with model and comparing with actual values

# In[14]:


y_predict = st_reg.predict(x_test)


# In[15]:


df = pd.DataFrame(y_test, columns = ['Actual Score'])
df


# In[16]:


df['Predicted score'] = y_predict
df


# In[17]:


## Ploting test scores against predicted score
plt.scatter(y_test,y_predict)
plt.xlabel('y_test (Expected)', size=15)
plt.ylabel('Predictions', size=15)
plt.show()


# In[18]:


## Predicting value of Given 9.25 hours study time
hours = [[9.25, ]]
own_pred = st_reg.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # Evaluating model

# In[19]:


## Calculating Root mean squared error
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)

print('Mean square error: ', mse)
print('Root mean square error: ', rmse)


# In[20]:


## Calculating Mean absolute error
from sklearn.metrics import mean_absolute_error  
mae = mean_absolute_error(y_test, y_predict)
print('Mean absolute error: ', mae)


# In[21]:


## Calculating R-squared
r2 = st_reg.score(x_train, y_train)
print('R-squared: ', r2)


# In[22]:


## Calculating adjt. R-squared
n = x_train.shape[0]
p = x_train.shape[1]
adjust_r2 = 1-(1-r2)*(n-1)/(n-p-1)

print('Adjusted R-squared: ',adjust_r2)

