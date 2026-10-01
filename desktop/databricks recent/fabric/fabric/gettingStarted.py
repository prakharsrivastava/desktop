# Databricks notebook source
# MAGIC %md
# MAGIC # Running Complete Spark Statements
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC reading "2019.csv"
# MAGIC

# COMMAND ----------

df = spark.read.csv("Files/2019.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC displaying 2019.csv
# MAGIC

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC defining schema for 2019.csv

# COMMAND ----------

from pyspark.sql.types import *

orderSchema = StructType([
    StructField("SalesOrderNumber", StringType()),
    StructField("SalesOrderLineNumber", IntegerType()),
    StructField("OrderDate", DateType()),
    StructField("CustomerName", StringType()),
    StructField("Email", StringType()),
    StructField("Item", StringType()),
    StructField("Quantity", IntegerType()),
    StructField("UnitPrice", FloatType()),
    StructField("Tax", FloatType())
])

df = spark.read.format("csv").schema(orderSchema).load("Files/2019.csv")

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC defining schema and then reading the files altogether

# COMMAND ----------

from pyspark.sql.types import *

orderSchema = StructType([
    StructField("SalesOrderNumber", StringType()),
    StructField("SalesOrderLineNumber", IntegerType()),
    StructField("OrderDate", DateType()),
    StructField("CustomerName", StringType()),
    StructField("Email", StringType()),
    StructField("Item", StringType()),
    StructField("Quantity", IntegerType()),
    StructField("UnitPrice", FloatType()),
    StructField("Tax", FloatType())
    ])

df = spark.read.format("csv").schema(orderSchema).load("Files/*.csv")

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC playing with dataframe operations

# COMMAND ----------

customers = df['CustomerName', 'Email']

print(customers.count())
print(customers.distinct().count())

display(customers.distinct())

# COMMAND ----------

customers = df.select("CustomerName", "Email").where(df['Item']=='Road-250 Red, 52')
print(customers.count())
print(customers.distinct().count())

display(customers.distinct())

# COMMAND ----------

productSales = df.select("Item", "Quantity").groupBy("Item").sum()

display(productSales)

# COMMAND ----------

from pyspark.sql.functions import split

# Select only "CustomerName" and "Email" columns (assuming they exist in your original DataFrame 'df')
customers_with_names = df.select("CustomerName", "Email")

# Split the "CustomerName" column using a space delimiter
customers_with_names = customers_with_names.withColumn("firstName", split(customers_with_names["CustomerName"], " ").getItem(0))
customers_with_names = customers_with_names.withColumn("lastName", split(customers_with_names["CustomerName"], " ").getItem(1))

# Display the resulting DataFrame
display(customers_with_names)

# COMMAND ----------

from pyspark.sql.functions import *

yearlySales = df.select(year(col("OrderDate")).alias("Year")).groupBy("Year").count().orderBy("Year")

display(yearlySales)

# COMMAND ----------

# MAGIC %md
# MAGIC transforming the data to be saved in parquet file format
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import *

# Create Year and Month columns
transformed_df = df.withColumn("Year", year(col("OrderDate"))).withColumn("Month", month(col("OrderDate")))

# Create the new FirstName and LastName fields
transformed_df = transformed_df.withColumn("FirstName", split(col("CustomerName"), " ").getItem(0)).withColumn("LastName", split(col("CustomerName"), " ").getItem(1))

# Filter and reorder columns
transformed_df = transformed_df["SalesOrderNumber", "SalesOrderLineNumber", "OrderDate", "Year", "Month", "FirstName", "LastName", "Email", "Item", "Quantity", "UnitPrice", "Tax"]

# Display the first five orders
display(transformed_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC saving the transformed data in parquet formatted file
# MAGIC

# COMMAND ----------

transformed_df.write.mode("overwrite").parquet('Files/transformed_data/orders')

print ("Transformed data saved!")

# COMMAND ----------

# MAGIC %md
# MAGIC loading the parquet formatted file in a dataframe

# COMMAND ----------

orders_df = spark.read.format("parquet").load("Files/transformed_data/orders")
display(orders_df)

# COMMAND ----------

# MAGIC %md
# MAGIC saving as a delta table

# COMMAND ----------

orders_df.write.format("delta").saveAsTable("salesOrders")

# COMMAND ----------

# MAGIC %md
# MAGIC using spark.sql API support to read the delta table
# MAGIC

# COMMAND ----------

sql_df = spark.sql("SELECT * FROM salesorders")
display(sql_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Running complete SQL statements
# MAGIC

# COMMAND ----------

# MAGIC %%sql
# MAGIC SELECT YEAR(OrderDate) AS OrderYear,
# MAGIC        SUM((UnitPrice * Quantity) + Tax) AS GrossRevenue
# MAGIC FROM salesorders
# MAGIC GROUP BY YEAR(OrderDate)
# MAGIC ORDER BY OrderYear;