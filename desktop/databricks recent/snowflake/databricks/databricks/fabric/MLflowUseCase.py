# Databricks notebook source
# MAGIC %md
# MAGIC # Using MLflow for Model Training, Tracking and Versioning

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading Dataset into a Spark Dataframe

# COMMAND ----------

data = spark.read.format("csv").option("header","true").load("Files/diabetes.csv")
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataframe API Manipulations

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

data = data.dropna().select(col("Pregnancies").astype("int"),
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
# MAGIC ### Building Our Model with Logging of Metrics and Pipeline Encapsulation

# COMMAND ----------

def train_diabetes_model(training_data, test_data, maxIterations, regularization):
    import mlflow
    import mlflow.spark
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from mlflow.models.signature import infer_signature
    import time
    
    # Start an MLflow run  
    with mlflow.start_run():
        numFeatures = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

        # define feature engineering and model steps

        numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
        numScaler = MinMaxScaler(inputCol=numVector.getOutputCol(), outputCol="normalizedFeatures")
        featureVector = VectorAssembler(inputCols=["normalizedFeatures"], outputCol="features")
        algo = LogisticRegression(labelCol="Outcome", featuresCol="features", maxIter=maxIterations, regParam=regularization)

        # chain the steps as stages in a Pipeline
        Pipeline = Pipeline(stages=[numVector, numScaler,featureVector,algo])

        # Log training parameter values
        print ("Training Logistic Regression model...")
        mlflow.log_param('maxIter', algo.getMaxIter())
        mlflow.log_param('regParam', algo.getRegParam())
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

# MAGIC %md
# MAGIC ### Training Our Model with Specific Parameters

# COMMAND ----------

train_diabetes_model(train, test, 5, 0.5)

# COMMAND ----------

train_diabetes_model(train, test, 10, 0.2)