# Databricks notebook source
# MAGIC %md
# MAGIC # Training Diabetes Regression Predictor using MLflow and Fabric
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading Diabetes csv dataset into a Spark Dataframe

# COMMAND ----------

df = spark.read.format("csv").option("header", "true").load("Files/diabetes_Training_Dataset.csv")
display(df) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Casting Dataframe Columns with DataTypes

# COMMAND ----------

from pyspark.sql.types import *
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
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Splitting Dataframe into Testing and Training Dataset

# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
print ("Training Rows:", train.count(), " Testing Rows:", test.count())
     

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC #### Arranging Features in Matrix format

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, MinMaxScaler

numericFeatures = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]
numericColVector = VectorAssembler(inputCols=numericFeatures, outputCol = "numericFeatures")
vectorizedData = numericColVector.transform(train)

minMax = MinMaxScaler(inputCol= numericColVector.getOutputCol(), outputCol="normalizedFeatures")
scaledData = minMax.fit(vectorizedData).transform(vectorizedData)

compareNumerics = scaledData.select("numericFeatures", "normalizedFeatures")
display(compareNumerics)
     

# COMMAND ----------

# MAGIC %md
# MAGIC #### Scaling/Normalising our Features

# COMMAND ----------

preppedData = scaledData[col("normalizedFeatures").alias("features"), col("Outcome").alias("label")]
display(preppedData)
     

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training our Logisitic Regression Model

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.3)
model = lr.fit(preppedData)
print ("Model trained!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inferencing Model with Testing Dataset

# COMMAND ----------

# Prepare the test data
vectorizedTestData = numericColVector.transform(test)
scaledTestData = minMax.fit(vectorizedTestData).transform(vectorizedTestData)
preppedTestData = scaledTestData[col("normalizedFeatures").alias("features"), col("Outcome").alias("label")]
   
# Get predictions
prediction = model.transform(preppedTestData)
predicted = prediction.select("features", "probability", col("prediction").astype("Int"), col("label").alias("trueLabel"))
display(predicted)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluating Our Model with Various Metrics

# COMMAND ----------


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
   
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
   
# Simple accuracy
accuracy = evaluator.evaluate(prediction, {evaluator.metricName:"accuracy"})
print("Accuracy:", accuracy)
   
# Individual class metrics
labels = [0,1]
print("\nIndividual class metrics:")
for label in sorted(labels):
    print ("Class %s" % (label))
   
    # Precision
    precision = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                                evaluator.metricName:"precisionByLabel"})
    print("\tPrecision:", precision)
   
    # Recall
    recall = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                             evaluator.metricName:"recallByLabel"})
    print("\tRecall:", recall)
   
    # F1 score
    f1 = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                         evaluator.metricName:"fMeasureByLabel"})
    print("\tF1 Score:", f1)
   
# Weighted (overall) metrics
overallPrecision = evaluator.evaluate(prediction, {evaluator.metricName:"weightedPrecision"})
print("Overall Precision:", overallPrecision)
overallRecall = evaluator.evaluate(prediction, {evaluator.metricName:"weightedRecall"})
print("Overall Recall:", overallRecall)
overallF1 = evaluator.evaluate(prediction, {evaluator.metricName:"weightedFMeasure"})
print("Overall F1 Score:", overallF1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Encapsulation Our Model as a Pipeline

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
   

numFeatures = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
   
# Define the feature engineering and model training algorithm steps
numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
featureVector = VectorAssembler(inputCols=["normalizedFeatures"], outputCol="Features")
algo = LogisticRegression(labelCol="Outcome", featuresCol="Features", maxIter=10, regParam=0.3)
   
# Chain the steps as stages in a pipeline
pipeline = Pipeline(stages=[ numVector, numScaler, featureVector, algo])
   
# Use the pipeline to prepare data and fit the model algorithm
model = pipeline.fit(train)
print ("Model trained!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inferencing Predictions from Test Data using Model Pipeline

# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("Features", "probability", col("prediction").astype("Int"), col("Outcome").alias("trueLabel"))
display(predicted)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving Our Model in Lakehouse "Files"

# COMMAND ----------


model.save("Files/diabetes.model")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inferencing our Locally Stored Model

# COMMAND ----------

from pyspark.ml.pipeline import PipelineModel

persistedModel = PipelineModel.load("Files/diabetes.model")
   
newData = spark.createDataFrame ([{"Pregnancies": 8,
                                  "Glucose": 85,
                                  "BloodPressure": 65,
                                  "SkinThickness": 29,
                                  "Insulin": 0,
                                  "BMI": 26.6,
                                  "DiabetesPedigreeFunction": 0.672,
                                  "Age": 34
                                  }])
   
   
predictions = persistedModel.transform(newData)
display(predictions.select("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",  col("prediction").alias("PredictedOutcome")))
     

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logging the Model using MLflow

# COMMAND ----------

def train_diabetes_model(training_data, test_data):
    import mlflow
    import mlflow.spark
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    import time
    from mlflow.models.signature import infer_signature
    
    # Start an MLflow run  
    with mlflow.start_run():
        numFeatures = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

        # define feature engineering and model steps

        numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
        numScaler = MinMaxScaler(inputCol=numVector.getOutputCol(), outputCol="normalizedFeatures")
        featureVector = VectorAssembler(inputCols=["normalizedFeatures"], outputCol="features")
        algo = LogisticRegression(labelCol="Outcome", featuresCol="features")

        # chain the steps as stages in a Pipeline
        Pipeline = Pipeline(stages=[numVector, numScaler,featureVector,algo])

        
        print ("Training Logistic Regression model...")
        model = Pipeline.fit(training_data)
   
        # Evaluate the model and log metrics
        prediction = model.transform(test_data)
        metrics = ["accuracy", "weightedRecall", "weightedPrecision"]
        for metric in metrics:
            evaluator = MulticlassClassificationEvaluator(labelCol="Outcome", predictionCol="prediction", metricName=metric)
            metricValue = evaluator.evaluate(prediction)
            print("%s: %s" % (metric, metricValue))
            mlflow.log_metric(metric, metricValue)
        
        input_example = training_data.select(*numFeatures).limit(5).toPandas()
        prediction_example = prediction.select("prediction").limit(5).toPandas()

        # Infer the signature
        signature = infer_signature(input_example, prediction_example)
   
        # Log the model itself
        unique_model_name = "classifier-" + str(time.time())
        mlflow.spark.log_model(
            model, 
            unique_model_name, 
            signature=signature, 
            input_example=input_example, 
            conda_env=mlflow.spark.get_default_conda_env()
        )
        
   
        print("Experiment run complete.")

# COMMAND ----------

train_diabetes_model(train, test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Code for Pipeline Notebook
# MAGIC

# COMMAND ----------

import mlflow
from synapse.ml.predict import MLFlowTransformer
    
df = spark.read.format("delta").load("Tables/Training_Dataset")
    
model = MLFlowTransformer(
        inputCols=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"], # Your input columns here
        outputCol="predictions", # Your new column name here
        modelName="YOUR_MODEL_NAME", # Your model name here
        modelVersion="YOUR_MODEL_VERSION" # Your model version here
    )
df = model.transform(df)
    
df.write.format('delta').saveAsTable("Diabetes_Final_Table")