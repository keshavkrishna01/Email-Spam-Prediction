#!/usr/bin/env python
# coding: utf-8

# # Email Spam Prediction Using Machine Learning Alorithms 

# ### Step 01:Importing the Dependencies

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ### Step 02: Data Collection & Pre-Processing**

# In[4]:


# loading the data from csv file to a pandas Dataframe
raw_mail_data = pd.read_csv('mail_data.csv')


# In[5]:


print(raw_mail_data)


# In[6]:


# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[7]:


# printing the first 5 rows of the dataframe
mail_data.head()


# In[8]:


# checking the number of rows and columns in the dataframe
mail_data.shape


# In[28]:


# label spam mail as 0;  ham mail as 1;

mail_data['Category'].replace({'spam':'0','ham':'1'},inplace=True)
# other method
# mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
# mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# In[29]:


# separating the data as texts and label

X = mail_data['Message']

Y = mail_data['Category']


# In[30]:


print(X)


# In[12]:


print(Y)


# ### Step 03: Data Modeling

# **a) Spliting data into test and Train section**

# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[14]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[18]:


# transform the text data to feature vectors that can be used as input to the Logistic regression

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[19]:


print(X_train)


# In[20]:


print(X_train_features)


# **b)Data Modeling: Selecting a machine learning model**

# In[21]:


model = LogisticRegression()


# ### Step 04: Model Trining

# In[22]:


# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)


# ### Step 05: Model Testing

# In[23]:


# prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[24]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[25]:


# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[26]:


print('Accuracy on test data : ', accuracy_on_test_data)


# ### Step 06: Model Deployment

# In[27]:


input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')


# In[ ]:




