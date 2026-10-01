# Databricks notebook source
# MAGIC %md
# MAGIC # Sentiment Analysis Notebook
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading the dataset into a spark dataframe

# COMMAND ----------

df = spark.read.format("delta").load("Tables/ecommerce_table")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installing azure-ai-text-analytics SDK

# COMMAND ----------

# MAGIC %pip install azure-ai-textanalytics==5.2.0

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting variables for our language resource

# COMMAND ----------

language_key="YOUR_LANGUAGE_RESOURCE_KEY"
language_endpoint="YOUR_LANGUAGE_RESOURCE_ENDPOINT"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating a Language Analysis Client

# COMMAND ----------

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

def authenticate_client():
    ta_credential = AzureKeyCredential(language_key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=language_endpoint, 
            credential=ta_credential)
    return text_analytics_client

client = authenticate_client()
print(client)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Defining the Sentiment Analysis Function

# COMMAND ----------

def sentiment_analysis(client, documents):
    result = client.analyze_sentiment(documents, show_opinion_mining=True)
    doc_result = [doc for doc in result if not doc.is_error]
    
    reviews=[]

    for doc in doc_result:
        reviews.append(doc.sentiment)
    
    return reviews

# COMMAND ----------

# MAGIC %md
# MAGIC ### Defining a function to call sentiment analysis function in a Batch Size of 10

# COMMAND ----------

# Function to process reviews in batches
def process_reviews_in_batches(reviews, batch_size=10):
    results = []
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i+batch_size]
        sentiment_batch = sentiment_analysis(client, batch)  # Replace `None` with your client
        results.extend(sentiment_batch)
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataframe Extraction and Manipulation

# COMMAND ----------

# Extract reviews from the Spark DataFrame as a list
reviews_list = [row["Review"] for row in df.collect()]



# COMMAND ----------

# Invoke the sentiment analysis function by the process_reviews_in_batches function
sentiments = process_reviews_in_batches(reviews_list, batch_size=10)

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

# Map sentiment results back to a DataFrame
sentiments_rdd = spark.sparkContext.parallelize(zip(reviews_list, sentiments))
sentiments_df = sentiments_rdd.toDF(["review", "sentiment"])

# Join the sentiment DataFrame back to the original DataFrame
df_with_sentiment = df.join(sentiments_df, "review")

# Add "positive" and "negative" columns based on the sentiment
df_with_sentiment = (
    df_with_sentiment
    .withColumn("positive", (col("sentiment") == "positive").cast(IntegerType()))
    .withColumn("negative", (col("sentiment") == "negative").cast(IntegerType()))
)

# Show the final DataFrame with additional columns
display(df_with_sentiment)

# COMMAND ----------

# drop the duplicates from the dataframe
df_with_sentiment.dropDuplicates()
display(df_with_sentiment)

# COMMAND ----------

# Saving the DataFrame as a Parquet file
df_with_sentiment.write.mode("overwrite").parquet("Files/gold_layers")


# COMMAND ----------

# Saving the DataFrame as a table
df_with_sentiment.write.mode("overwrite").saveAsTable("gold_table")
