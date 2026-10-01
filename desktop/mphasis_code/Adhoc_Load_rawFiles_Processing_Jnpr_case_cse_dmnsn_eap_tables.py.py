# Databricks notebook source
# MAGIC %md
# MAGIC ### Storage account connection

# COMMAND ----------

# MAGIC %run /DataFabric/HPE-Azure-CDO-GLB/utils/utilities

# COMMAND ----------

secret_value = Secrets()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Variable and Parameter declarations
# MAGIC

# COMMAND ----------

from datetime import datetime

current_date = datetime.now().strftime('%Y-%m-%d')
print(current_date)
# Parameters
dbutils.widgets.text("p_sourceContainer", "")
dbutils.widgets.text("p_sourceFileLocation", "")
dbutils.widgets.text("p_storageAccountName", "")
dbutils.widgets.text("p_catalogTableName", "")
dbutils.widgets.text("p_dt","")

source_container = dbutils.widgets.get("p_sourceContainer")
source_file_location = dbutils.widgets.get("p_sourceFileLocation")
storage_account_name = dbutils.widgets.get("p_storageAccountName")
catalog_table_name = dbutils.widgets.get("p_catalogTableName")
print(catalog_table_name)
p_dt = dbutils.widgets.get("p_dt")


inputFilePathCommand = f"abfss://{source_container}@{storage_account_name}.dfs.core.windows.net/{source_file_location}/"
print(inputFilePathCommand)
archive_location = f"abfss://{source_container}@{storage_account_name}.dfs.core.windows.net/{source_file_location}/Archive/{p_dt}"
print(archive_location)


# COMMAND ----------

dbutils.fs.ls(inputFilePathCommand)

# COMMAND ----------

# Check if the path exists
try:
    dbutils.fs.ls(archive_location)
    print(f"Path '{archive_location}' already exists.")
except Exception:
    print(f"Path '{archive_location}' does not exist. Creating it now...")
    dbutils.fs.mkdirs(archive_location)
    print(f"Path '{archive_location}' created successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Functions

# COMMAND ----------

# DBTITLE 1,Cell 9
from pyspark.sql.functions import to_date, col, lower

def target_schema_conversion(input_df):
    target_catalog_df = spark.table(catalog_table_name)
    target_schema = {field.name.lower(): field.dataType for field in target_catalog_df.schema.fields}
    print(target_schema)
    input_df = input_df.select([col(c).cast(target_schema[c]) if c in target_schema else col(c) for c in input_df.columns])
    return input_df

# COMMAND ----------

input_df = spark.read.format("orc").option("mergeSchema", "true").load(inputFilePathCommand)
# .option("recursiveFileLookup", "true")
input_df.printSchema()

# COMMAND ----------

# print(catalog_table_name)
# # print(len(input_df.columns))

input_df = target_schema_conversion(input_df)
print(catalog_table_name)
print(input_df.count())

# COMMAND ----------

display(input_df.dtypes)

# COMMAND ----------

input_df.show(1,False)

# COMMAND ----------

from pyspark.sql.functions import collect_list, col, struct, current_timestamp, lit, when, to_date, regexp_extract
import json

spark.conf.set("spark.sql.adaptive.enabled", True)                     
# spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", True)
#spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")#not allowed
#spark.conf.set("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
#spark.conf.set("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
# spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", True)
# spark.conf.set("spark.databricks.delta.autoCompact.enabled", True)
spark.conf.set("fs.azure.enable.concurrent.append", "true")
spark.conf.set("fs.azure.without_padding", "true")
spark.conf.set("fs.azure.read.optimize.concurrent", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# input_df = target_schema_conversion(input_df)
# print(catalog_table_name)
#------------------------------------------------------------
# Creating audit columns and source_file_name
input_df = input_df.withColumn("create_dtm", current_timestamp()).withColumn("update_dtm", current_timestamp()).withColumn("parquet_source_file_name",regexp_extract(input_df["_metadata.file_path"], r"([^/]+$)", 1))

#Loading data into Table by creating the raw data as delta format -- bronze layer
if input_df.count() > 0 :
    try:
        input_df.write.option("mergeSchema", "true").format("delta").mode("overwrite").saveAsTable(catalog_table_name)

        files = dbutils.fs.ls(inputFilePathCommand)
        # for file in files:
        #     if file.name.endswith('.parquet'):
        #         dbutils.fs.mv(file.path, archive_location, True)
                
        # Create JSON format
        json_df = spark.sql(
            f"""
                SELECT * FROM (DESCRIBE HISTORY {catalog_table_name})
                ORDER BY version DESC
                LIMIT 1
                """
        )
        operation_metrics = json_df.select(col("operationMetrics")).collect()[0][0]
        num_output_rows = operation_metrics.get("numOutputRows", 0)

        success_message = json.loads(json.dumps({"total_rows": num_output_rows, "output_msg":"Data loaded into table"}))
        
        dbutils.notebook.exit(success_message)
    except Exception as err:

        failure_message = json.loads(json.dumps({"total_rows": 0, "error_output_msg":str(err)}))
        raise Exception(failure_message)
        dbutils.notebook.exit(failure_message)
else:
    success_message = json.loads(json.dumps({"total_rows": 0, "output_msg":"Data count is 0"}))
    dbutils.notebook.exit(success_message)
