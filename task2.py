#!/usr/bin/env python
# coding: utf-8
#Task 2
# In[ ]:


#UNEMPLOYMENT ANALYSIS WITH PYTHON


# In[18]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# In[7]:


data=pd.read_csv("Unemployment_Rate_upto_11_2020.csv")


# In[4]:


data


# In[4]:


data.head()


# In[5]:


data.isnull().sum()


# In[13]:


data.columns= ["States","Date","Frequency",
               "Estimated Unemployment Rate",
               "Estimated Employed",
               "Estimated Labour Participation Rate",
               "Region","longitude","latitude"]


# In[8]:


sns.heatmap(data.corr())


# In[9]:


#let’s visualize the data to analyze the unemployment rate


# In[14]:


plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate",hue="Region",data=data)
plt.show()


# In[15]:


plt.title("Indian employment")
sns.histplot(x="Estimated Employed",hue="Region",data=data)
plt.show()


# In[16]:


#dashboard to analyze the unemployment rate


# In[20]:


unemploment = data[["States", "Region", "Estimated Unemployment Rate"]]
figure = px.sunburst(unemploment, path=["Region", "States"], 
                     values="Estimated Unemployment Rate", 
                     width=700, height=700, color_continuous_scale="RdY2Gn", 
                     title="Unemployment Rate in India")
figure.show()


# In[ ]:




