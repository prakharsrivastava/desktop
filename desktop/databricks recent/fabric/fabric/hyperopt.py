# Databricks notebook source
# MAGIC %md
# MAGIC # Hyperparameter tuning using Hyperopt
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installing Hyperopt in our Python Environment

# COMMAND ----------

# MAGIC %pip install hyperopt

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading dataset into a spark Dataframe

# COMMAND ----------

df = spark.read.format("csv").option("header","true").load("Files/diabetes.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataframe Manipulation

# COMMAND ----------

from pyspark.sql.functions import *
   

data = df.dropna().select(col("Pregnancies").astype("int"),
                           col("Glucose").astype("int"),
                          col("BloodPressure").astype("int"),
                          col("SkinThickness").astype("int"),
                          col("Insulin").astype("int"),
                          col("BMI").astype("float"),
                          col("DiabetesPedigreeFunction").astype("float"),
                          col("Age").astype("int"),
                          col("Outcome").astype("int")
                          )

   
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
print ("Training Rows:", train.count(), " Testing Rows:", test.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Defining the Hyperopt Objective Function

# COMMAND ----------

from hyperopt import STATUS_OK
import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
   
def objective(params):
    # Train a model using the provided hyperparameter value
    numFeatures = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
    numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
    featureVector = VectorAssembler(inputCols=["normalizedFeatures"], outputCol="Features")
    mlAlgo = DecisionTreeClassifier(labelCol="Outcome",    
                                    featuresCol="Features",
                                    maxDepth=params['MaxDepth'], maxBins=params['MaxBins'])
    pipeline = Pipeline(stages=[numVector, numScaler, featureVector, mlAlgo])
    model = pipeline.fit(train)
       
    # Evaluate the model to get the target metric
    prediction = model.transform(test)
    eval = MulticlassClassificationEvaluator(labelCol="Outcome", predictionCol="prediction", metricName="accuracy")
    accuracy = eval.evaluate(prediction)
       
    # Hyperopt tries to minimize the objective function, so you must return the negative accuracy.
    return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using the Objective Function for Hyperparameter Tuning
# MAGIC

# COMMAND ----------

from hyperopt import Trials
from hyperopt import fmin, tpe, hp
  
search_space = {
    'MaxDepth': hp.randint('MaxDepth', 10),
    'MaxBins': hp.choice('MaxBins', [10, 20, 30])
}

algo=tpe.suggest

# Create a Trials object to track each run
trial_runs = Trials()
   
argmin = fmin(
  fn=objective,
  space=search_space,
  algo=algo,
  max_evals=3,
  trials=trial_runs)
   
print("Best param values: ", argmin)
   
# Get details from each trial run
print ("trials:")
for trial in trial_runs.trials:
    print ("\n", trial)