# Databricks notebook source
import mlflow

# COMMAND ----------

import os
os.getcwd()

# COMMAND ----------

mlflow.set_experiment('/Workspace/Users/rahul.jha.stats2@gmail.com/Demo Experiment')

# COMMAND ----------

with mlflow.start_run(run_name = "first test run"):
    mlflow.log_param("theta", 360)
    mlflow.log_params({
        "alpha" : 0.33,
        "beta" : 12
    })
    mlflow.log_metric("accuracy", 0.9)
    mlflow.log_artifact("sample.py")
    

# COMMAND ----------

# 1. Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# COMMAND ----------

mlflow.sklearn.autolog()

# COMMAND ----------

# 2. Load and prepare the data
iris = load_iris()
X = iris.data  # Features (data)
y = iris.target  # Target labels


# COMMAND ----------


# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# The test_size=0.2 means 20% of the data will be used for testing, and 80% for training.


# COMMAND ----------

# 4. Choose a model and initialize it
model = KNeighborsClassifier(n_neighbors=3) # n_neighbors is a hyperparameter

# 5. Train (fit) the model
# The model learns the relationship between X_train and y_train
model.fit(X_train, y_train)


# COMMAND ----------

# 6. Make predictions on the test set
predictions = model.predict(X_test)

# 7. Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")


# COMMAND ----------

with mlflow.start_run(run_name = "parent_run") as parent:
    with mlflow.start_run(run_name = "child_one", parent_run_id = parent.info.run_id, nested=True) as child1:
        mlflow.log_params({"param1": 100})
    with mlflow.start_run(run_name = "child_two", parent_run_id = parent.info.run_id, nested=True) as child2:
        mlflow.log_params({"param1": 100})
    with mlflow.start_run(run_name = "child_three", parent_run_id = parent.info.run_id, nested=True) as child3:
        mlflow.log_params({"param1": 100})

# COMMAND ----------

