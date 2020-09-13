#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[20]:


dataset = pd.read_csv('hadoop.csv')
X =  pd.DataFrame(dataset, columns = ['ns','nd','nf','entropy','la','ld','lt','fix','ndev','age','nuc','exp','rexp','sexp'])
y = dataset.iloc[:,9:10].values


# In[21]:


from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
y = le1.fit_transform(y)
X['fix'] = le2.fit_transform(X['fix'])


# In[37]:


print(X)
print(y)


# In[23]:


from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
imputer.fit(X)
X = imputer.transform(X)


# In[24]:


print(X)


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[26]:


print(X_train)


# In[27]:


print(y_train)


# In[30]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[31]:


print(X_train)


# In[32]:


print(X_test)


# In[33]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[34]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[35]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[36]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




