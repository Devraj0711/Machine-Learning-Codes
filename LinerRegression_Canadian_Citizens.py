#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[54]:


df=pd.read_csv(r"C:\py-master\ML\1_linear_reg\Exercise\canada_per_capita_income.csv")


# In[55]:


df.head()


# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("year")
plt.ylabel("per Capita Income(US$)")
plt.scatter(df.year, df['per capita income (US$)'], color="red", marker='*')


# In[58]:


new_x=df.drop(['per capita income (US$)'], axis='columns')
new_x


# In[59]:


perCapita=df['per capita income (US$)']
perCapita


# In[60]:


#to create linear regression object
reg= linear_model.LinearRegression()
reg.fit(new_x, perCapita)


# In[61]:


reg.predict([[2020]])


# In[62]:


reg.coef_


# In[63]:


reg.intercept_


# In[64]:


2020*828.46507522+-1632210.7578554575


# In[ ]:




