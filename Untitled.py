#!/usr/bin/env python
# coding: utf-8

# In[24]:


# Task 1 :  IRIS FLOWER CLASSIFICATION


# In[46]:


#modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("Iris.csv")


# In[25]:


data.head()


# In[26]:


data=data.drop(columns=["Id"])


# In[27]:


data.head()


# In[28]:


data.describe()


# In[29]:


data.info()


# In[30]:


print("target labels",data['Species'].unique())


# In[31]:


#preprocessing data
data.isnull().sum()


# In[32]:


sns.countplot(data['Species']);


# In[33]:


import plotly.express as px
fig=px.data.medals_wide()
fig=px.bar(data,x="SepalWidthCm", y="SepalLengthCm", color="Species")
fig.show()


# In[34]:


# Visualize the whole dataset
sns.pairplot(data, hue='Species')


# In[ ]:


#From above visualization, we can tell that iris-setosa is well separated from the other two flowers.
#And iris virginica is the longest flower and iris setosa is the shortest.


# In[35]:


#Iris Classification Model
x=data.drop("Species",axis=1)
y=data["Species"]


#training the model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


#k nearest neighbours
from sklearn.neighbors import KNeighborsClassifier 
knn= KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)


# In[40]:


#prediction
x_new = np.array([[5, 2.8, 1, 0.2]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))


# In[41]:


#model accuracy
print("accuracy",knn.score(x_test,y_test)*100)


# In[42]:


#decision_tree
from sklearn.tree import DecisionTreeClassifier
model2=DecisionTreeClassifier()


# In[44]:


model2.fit(x_train,y_train)


# In[45]:


#model2 accuracyrr
print("accuracy",model2.score(x_test,y_test)*100)


# In[ ]:





# In[ ]:




