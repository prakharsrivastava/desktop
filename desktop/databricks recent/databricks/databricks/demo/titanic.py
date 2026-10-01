# Databricks notebook source
# DBTITLE 1,Cell 1
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Download Titanic dataset from public source
import urllib.request
import os

# Create directory for data
data_dir = '/tmp/titanic'
os.makedirs(data_dir, exist_ok=True)

# Download train.csv
train_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
train_path = f'{data_dir}/train.csv'
urllib.request.urlretrieve(train_url, train_path)

# Download test.csv
test_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
test_path = f'{data_dir}/test.csv'
urllib.request.urlretrieve(test_url, test_path)

print('Data download complete.')
print(f'{train_path}')
print(f'{test_path}')

# COMMAND ----------

# DBTITLE 1,Load training data
train_data = pd.read_csv(f'{data_dir}/train.csv')
train_data.head()

# COMMAND ----------

# DBTITLE 1,Load test data
# For this tutorial, we'll split the training data to create a test set
# In actual Kaggle submission, you'd use the separate test.csv file
from sklearn.model_selection import train_test_split

# Split data: 80% train, 20% test
full_data = train_data.copy()
train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)

print(f"Training set: {len(train_data)} passengers")
print(f"Test set: {len(test_data)} passengers")
test_data.head()

# COMMAND ----------

# DBTITLE 1,Explore pattern - women survival rate
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

# COMMAND ----------

# DBTITLE 1,Explore pattern - men survival rate
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

# COMMAND ----------

# DBTITLE 1,Train RandomForest model and generate predictions
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
print(output)
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

# COMMAND ----------

# Simple example using RandomForestClassifier from sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier

# Dummy data
import pandas as pd

X_dummy = [[0, 1], [1, 0], [0, 0], [1, 1]]
y_dummy = [0, 1, 0, 1]

df_dummy = pd.DataFrame(X_dummy, columns=['col1', 'col2'])
df_dummy['y'] = y_dummy
print(df_dummy)

model_dummy = RandomForestClassifier(n_estimators=10, random_state=42)
model_dummy.fit(X_dummy, y_dummy)
pred_dummy = model_dummy.predict([[0, 2]])

print("Prediction for [[0, 2]]:", pred_dummy)