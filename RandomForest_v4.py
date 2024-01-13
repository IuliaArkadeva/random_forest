#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree


# In[2]:


features = pd.read_csv('C:/Users/aliba/Desktop/cs/randforest_python_weather/HDbinary.csv')


# In[3]:


X = features.drop('HeartDisease',axis=1)
y = features['HeartDisease']


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)


# In[5]:


classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)


# In[6]:


classifier_rf.fit(X_train, y_train)


# In[ ]:





# In[7]:


rf = RandomForestClassifier(random_state=42, n_jobs=-1)


# In[8]:


params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}


# In[9]:


grid_search = GridSearchCV(estimator=rf, param_grid=params, cv = 4, n_jobs=-1, verbose=1, scoring="accuracy")


# In[10]:


grid_search.fit(X_train, y_train)


# In[11]:


rf_best = grid_search.best_estimator_


# In[12]:


plt.figure(figsize=(80,40))


# In[13]:


plot_tree(rf_best.estimators_[5], feature_names = X.columns,class_names=['CleanVeins', "HealthyLungs"],filled=True)


# In[ ]:




