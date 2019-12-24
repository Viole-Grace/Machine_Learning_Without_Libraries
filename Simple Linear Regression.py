#!/usr/bin/env python
# coding: utf-8

# In[102]:


import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
import math
import numba
from numba import jit, njit
import warnings
warnings.simplefilter('ignore')


# In[2]:


df = pd.read_csv("dataset.csv")


# In[4]:


x,y = list(df['Subject 1']), list(df['Subject 2'])


# In[18]:


@jit(parallel = True, fastmath=True)
def mean(arr):
    return sum(arr)/len(arr);


# In[103]:


mean_x, mean_y = mean(x), mean(y)


# In[80]:


@jit(parallel=True,fastmath=True)
def get_bias_and_constant(arr, l, mean_x=mean_x, mean_y=mean_y):
    sn, sd=0,0;
    for i in range(len(arr)):
        sn+=(arr[i]-mean_x)*(l[i]-mean_y)
        sd+=(arr[i]-mean_x)**2
    bias, constant = 0,0
    bias = sn/sd
    constant = mean_y - mean_x*bias
    print("intercept : {}, m : {}".format(constant, bias))
    
    return bias, constant


# In[104]:


m,c = get_bias_and_constant(arr=x, l=y, mean_x=mean_x, mean_y=mean_y)


# In[128]:


@jit(parallel=True, fastmath=True)
def metadata_stats(x=x,y=y,mean_x = mean_x, mean_y = mean_y,pred_y=pred_y,m=m,c=c):
    res_sum_of_sq, tot_sum_of_sq, sq_err = 0,0,0
    for i in range(len(x)):
        res_sum_of_sq+=(y[i]-pred_y[i])**2
        tot_sum_of_sq+=(y[i]-mean_y)**2
    r_sq = math.sqrt(res_sum_of_sq/(len(x)-2))
    sq_err = 1 - (res_sum_of_sq/tot_sum_of_sq)
    print("Intercept : {}\nBias : {}".format(c,m))
    print("Squared Error : {}".format(sq_err[0]))


# In[57]:


df['Predicted'] = df['Subject 1'].apply(lambda x : (m*x +c))


# In[61]:


lr = LinearRegression()
lr.fit(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))


# In[64]:


pred_y = lr.predict(np.array(x).reshape(-1,1))


# In[72]:


pr_y = [pred_y[i][0] for i in range(len(pred_y))]


# In[82]:


df['Library Prediction']=pr_y


# In[98]:


@jit(fastmath=True)
def calculate_error(col1, col2):
    return col1-col2


# In[99]:


df['Absolute error']=df.apply(lambda x : calculate_error(x['Predicted'], x['Library Prediction']), axis=1)


# In[100]:


df.head(5)


# In[130]:


print("Metadata Stats (Predicted):\n")
metadata_stats()


# In[132]:


print("Metadata Stats (Library) :\n")
print("Intercept  : {}\nBias : {}\nSquared Error : {}".format(lr.intercept_[0],lr.coef_[0][0],lr.score(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))))


# In[ ]:




