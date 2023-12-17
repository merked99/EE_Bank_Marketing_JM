#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.version_info


# In[ ]:


get_ipython().system('pip3 install pycaret')
import pycaret
pycaret.__version__


# In[2]:


import pandas as pd
import numpy as np
# fetch dataset 
from ucimlrepo import fetch_ucirepo

# Fetch dataset 
try:
    bank_marketing = fetch_ucirepo(id=222)
except Exception as e:
    print(f"Error: {e}")


# In[3]:


# data (as pandas dataframes) 
X = bank_marketing.data.features 
Y = bank_marketing.data.targets 
  
# metadata 
print(bank_marketing.metadata) 
  
# variable information 
print(bank_marketing.variables) 

from pycaret.classification import *

# Assuming 'data_dict' is the dictionary containing the data
features = bank_marketing['data']['features']
targets = bank_marketing['data']['targets']

# Combine features and targets into a single DataFrame
df = pd.concat([features, targets], axis=1)

# Initialize PyCaret
s = setup(df, target='y', train_size=0.8, session_id=222, log_experiment=True)


# In[4]:


# import ClassificationExperiment and init the class
from pycaret.classification import ClassificationExperiment
exp = ClassificationExperiment()

# check the type of exp
type(exp)

# init setup on exp
exp.setup(df, target = 'y', session_id = 222)


# In[5]:


# Compare models and select the best ones
best_model = exp.compare_models()

# Choose best model from compare_models to create and tune final model
model = exp.create_model('lightgbm')
tuned_model = exp.tune_model(model)

# Finalize the model
final_model = exp.finalize_model(tuned_model)

# plot confusion matrix
plot_model(final_model, plot = 'confusion_matrix')

# plot AUC
plot_model(final_model, plot = 'auc')

# plot feature importance
plot_model(final_model, plot = 'feature')

# check docstring to see available plots 
help(plot_model)

exp.evaluate_model(final_model)

# predict on test set
holdout_pred = exp.predict_model(final_model)

# show predictions df
holdout_pred.head()


# In[6]:


# Copy the 'original' DataFrame
new_data = bank_marketing['data']['original'].copy()

# Drop multiple columns (e.g., 'column1' and 'column2')

new_data.drop('y', axis=1, inplace=True)

# Display the first few rows of the new DataFrame
new_data.head()


# In[10]:


# predict model on new_data
predictions = predict_model(final_model, data = new_data)
exp.evaluate_model(final_model)


# In[11]:


# predict on the test set
holdout_pred = exp.predict_model(final_model)

# show predictions df
holdout_pred.head()


# In[12]:


# save pipeline
exp.save_model(final_model, 'my_first_pipeline')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_true = df['y']
y_pred = predictions['prediction_label']
accuracy = accuracy_score(y_true, y_pred)
print('Accuracy is: ', accuracy)


precision = precision_score(y_true, y_pred, average='weighted') 
recall = recall_score(y_true, y_pred, average='weighted') 
f1 = f1_score(y_true, y_pred, average='weighted')

print('Precision is: ', precision) 
print('Recall is: ', recall) 
print('F1 score is: ', f1)


# In[13]:


# Identify feature types
feature_types = df.dtypes

# Print feature types
print(feature_types)


# In[14]:


import matplotlib.pyplot as plt

# Example: Visualize the distribution of the target variable 'y'
df['y'].value_counts().plot(kind='bar')
plt.show()


# In[ ]:




