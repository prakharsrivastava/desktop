# Databricks notebook source
# MAGIC %md
# MAGIC ### Storage account connection

# COMMAND ----------

# spark.conf.set("fs.azure.account.auth.type.stglbfdnprod001.dfs.core.windows.net", "OAuth")
# spark.conf.set("fs.azure.account.oauth.provider.type.stglbfdnprod001.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
# spark.conf.set("fs.azure.account.oauth2.client.id.stglbfdnprod001.dfs.core.windows.net", "72cbbc73-54da-4a59-a70b-91b9728401f8")
# spark.conf.set("fs.azure.account.oauth2.client.secret.stglbfdnprod001.dfs.core.windows.net", "adls-auth-secret")
# spark.conf.set("fs.azure.account.oauth2.client.endpoint.stglbfdnprod001.dfs.core.windows.net", "https://login.microsoftonline.com/105b2061-b669-4b31-92ac-24d304d195dc/oauth2/token")


# COMMAND ----------

# MAGIC %run /DataFabric/HPE-Azure-CDO-GLB/utils/utilities

# COMMAND ----------

# MAGIC %md
# MAGIC ### Variable and Parameter declarations
# MAGIC

# COMMAND ----------



# COMMAND ----------

from datetime import datetime

current_date = datetime.now().strftime('%Y-%m-%d')

# Parameters
dbutils.widgets.text("p_sourceContainer", "")
dbutils.widgets.text("p_sourceFileLocation", "")
dbutils.widgets.text("p_storageAccountName", "")
dbutils.widgets.text("p_catalogTableName", "")

source_container = dbutils.widgets.get("p_sourceContainer")
source_file_location = dbutils.widgets.get("p_sourceFileLocation")
storage_account_name = dbutils.widgets.get("p_storageAccountName")
catalog_table_name = dbutils.widgets.get("p_catalogTableName")



inputFilePathCommand = f"abfss://{source_container}@{storage_account_name}.dfs.core.windows.net/{source_file_location}/"
print(inputFilePathCommand)
archive_location = f"abfss://{source_container}@{storage_account_name}.dfs.core.windows.net/{source_file_location}/Archive/"
print(archive_location)


# COMMAND ----------

df = spark.read.parquet("abfss://spark-egi-ingest@stglbfdnprod001.dfs.core.windows.net/ea_common_sales_order_netchange_kpi_agg_hw_fact/01072026/")
display(df.count())

# COMMAND ----------

dbutils.fs.ls(inputFilePathCommand)

# COMMAND ----------

# # Check if the path exists
# try:
#     dbutils.fs.ls(archive_location)
#     print(f"Path '{archive_location}' already exists.")
# except Exception:
#     print(f"Path '{archive_location}' does not exist. Creating it now...")
#     dbutils.fs.mkdirs(archive_location)
#     print(f"Path '{archive_location}' created successfully.")

check_and_create_path(archive_location)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Functions

# COMMAND ----------

# from pyspark.sql.functions import to_date, col, lower

# def target_schema_conversion(input_df):
#     target_catalog_df = spark.table(catalog_table_name)
#     target_schema = {field.name.lower(): field.dataType for field in target_catalog_df.schema.fields}
#     print(target_schema)
#     input_df = input_df.select([col(c).cast(target_schema[c]) if c in target_schema else col(c) for c in input_df.columns])
#     return input_df

# COMMAND ----------

input_df = spark.read.parquet(inputFilePathCommand)
input_df.printSchema()

# COMMAND ----------

display(input_df.count())

# COMMAND ----------

catalog_table_name

# COMMAND ----------

from pyspark.sql.functions import collect_list, col, struct, current_timestamp, lit, when, to_date, regexp_extract
import json

# Read parquet file
#input_df = spark.read.parquet(inputFilePathCommand)


spark.conf.set("spark.sql.adaptive.enabled", True)                     
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", True)
#spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")#not allowed
#spark.conf.set("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
#spark.conf.set("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", True)
spark.conf.set("spark.databricks.delta.autoCompact.enabled", True)

# Adding this to convert the column from datetime to date column. By default adf is converting the vertica db DATE data type into DateTime

# if  len(date_Conversion_Column_Names_List) > 0 and all(cols.strip().lower() for cols in date_Conversion_Column_Names_List if cols.strip().lower() in [col.lower() for col in input_df.columns]):
#     input_df = dateTypeConversion(input_df, date_Conversion_Column_Names_List)

# Configuring the input_df column data type from target catalog table data type
# input_df = spark.read.parquet(inputFilePathCommand)
renaming_columns = [col.split(".")[-1] if "."  in col else col  for col in input_df.columns]
input_df = input_df.toDF(*renaming_columns)
input_df.cache()
input_df = input_df.toDF(*[c.lower() for c in input_df.columns])
#---------------------------------------------------------


# input_df = target_schema_conversion(input_df)
input_df = target_schema_conversion(input_df,catalog_table_name)
#------------------------------------------------------------
# Creating audit columns and source_file_name
# input_df = input_df.withColumn("ins_dtm", current_timestamp()).withColumn("updt_dtm", current_timestamp()).withColumn("parquet_source_file_name",regexp_extract(input_df["_metadata.file_path"], r"([^/]+$)", 1))

input_df = add_audit_columns(input_df)

#Loading data into Table by creating the raw data as delta format -- bronze layer
if input_df.first() is not None :
    try:
        input_df.printSchema()
        #display(input_df)
        input_df.write.option("mergeSchema", "true").format("delta").mode("overwrite").saveAsTable(catalog_table_name)

        # files = dbutils.fs.ls(inputFilePathCommand)
        # for file in files:
        #     if file.name.endswith('.parquet'):
        #         dbutils.fs.mv(file.path, archive_location, True)

        #move_files_to_archive(inputFilePathCommand, archive_location)
                
        # # Create JSON format
        # json_df = spark.sql(
        #     f"""
        #         SELECT * FROM (DESCRIBE HISTORY {catalog_table_name})
        #         ORDER BY version DESC
        #         LIMIT 1
        #         """
        # )
        # operation_metrics = json_df.select(col("operationMetrics")).collect()[0][0]
        # num_output_rows = operation_metrics.get("numOutputRows", 0)
        
        # success_message = json.loads(json.dumps({"total_rows": num_output_rows, "output_msg":"Data loaded into table"}))

        success_message = data_load_status(catalog_table_name)
        
        dbutils.notebook.exit(success_message)
        input_df.unpersist()
    except Exception as err:
        input_df.unpersist()
        failure_message = json.loads(json.dumps({"total_rows": 0, "error_output_msg":str(err)}))
        raise Exception(failure_message)
        dbutils.notebook.exit(failure_message)
else:
    success_message = json.loads(json.dumps({"total_rows": 0, "output_msg":"Data count is 0"}))
    dbutils.notebook.exit(success_message)

