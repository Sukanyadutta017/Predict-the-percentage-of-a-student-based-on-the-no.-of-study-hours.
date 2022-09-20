#!/usr/bin/env python
# coding: utf-8

# # THE SPARK FOUNDATION
# ## Predict the percentage of a student based on the no. of study hours.
# **1st: Import python library**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **2nd: Import data**

# In[2]:


data = pd.read_csv("data.csv")
print ("Data Introducing")


# **3rd:Checking the imported data**

# In[3]:


print ("Data set is")
data


# In[4]:


print ("Data is imported successfully")


# **4th:Data Analysis**

# In[5]:


data.plot(x='Hours',y='Scores',style='1')
plt.title('Hours_Scores')
plt.xlabel('Hours')
plt.ylabel('Sores')
plt.show()


# In[6]:


data.plot.scatter(x='Hours',y='Scores')


# In[7]:


data.plot.bar(x='Hours',y='Scores')


# In[8]:


#data.sort_values(["Hours"], axis=0, ascending=[True], inplace=True)
#data
#data.plot.bar(x='Hours',y='Scores')


# ### Above analysis decalers that scores increases with no. of study hours

# **5th:Data is prepared for our model**

# In[9]:


x=data.iloc[:, :-1].values
y=data.iloc[:, 1].values
#print(x)


# **6th:Divide the data for training and testing models**

# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)


# **Linear Regression Method**

# In[16]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#from sklearn.ensemble import RandomForestRegressor
#regressor=RandomForestRegressor(n_estimators=1000,random_state=42)

regressor.fit(x_train, y_train)
print("Training Complete")


# **7th:Testing the Model**

# In[12]:


print(x_test)
print("Score Prediction")
y_pred = regressor.predict(x_test)
print(y_pred)


# **8th:Accuracy of the model**

# In[13]:


df=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# **9th:Prediction with Input**

# In[14]:


hours=[[9.25]]
pred=regressor.predict(hours)
print(pred)


# # Model Evalution

# In[15]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))


# # Thank you

# In[ ]:





# In[ ]:




