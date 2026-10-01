# Databricks notebook source
# MAGIC %md
# MAGIC ### Storage account connection

# COMMAND ----------

spark.conf.set("fs.azure.account.auth.type.stglbfdnprod001.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.stglbfdnprod001.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.stglbfdnprod001.dfs.core.windows.net", "72cbbc73-54da-4a59-a70b-91b9728401f8")
spark.conf.set("fs.azure.account.oauth2.client.secret.stglbfdnprod001.dfs.core.windows.net", "adls-auth-secret")
spark.conf.set("fs.azure.account.oauth2.client.endpoint.stglbfdnprod001.dfs.core.windows.net", "https://login.microsoftonline.com/105b2061-b669-4b31-92ac-24d304d195dc/oauth2/token")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Variable and Parameter declarations
# MAGIC

# COMMAND ----------

# Parameters
dbutils.widgets.text("p_sourceContainer", "")
dbutils.widgets.text("p_sourceFileLocation", "")
dbutils.widgets.text("p_storageAccountName", "")
dbutils.widgets.text("p_catalogTableName", "")
dbutils.widgets.text("p_dt","")
dbutils.widgets.text("p_pk_columns","")
dbutils.widgets.text("p_incremental_column","")

source_container = dbutils.widgets.get("p_sourceContainer")
source_file_location = dbutils.widgets.get("p_sourceFileLocation")
storage_account_name = dbutils.widgets.get("p_storageAccountName")
catalog_table_name = dbutils.widgets.get("p_catalogTableName")
dt = dbutils.widgets.get("p_dt")
pk_cols_str = dbutils.widgets.get("p_pk_columns")
incremental_column = dbutils.widgets.get("p_incremental_column")

# Variables
pk_cols = [c.strip() for c in pk_cols_str.split(",")]

inputFilePathCommand = f"abfss://{source_container}@{storage_account_name}.dfs.core.windows.net/{source_file_location}/{dt}"
archive_location = f"abfss://{source_container}@{storage_account_name}.dfs.core.windows.net/{source_file_location}/Archive/{dt}/"


# COMMAND ----------

inputFilePathCommand

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

from pyspark.sql.functions import to_date, col, lower
 
def target_schema_conversion(input_df):
    target_catalog_df = spark.table(catalog_table_name)
    target_schema = {field.name.lower(): field.dataType for field in target_catalog_df.schema.fields}
    print(target_schema)
    input_df = input_df.select([col(c).cast(target_schema[c]) if c in target_schema else col(c) for c in input_df.columns])
    return input_df

# COMMAND ----------

from pyspark.sql.functions import collect_list, col, struct, current_timestamp, lit, when, to_date, regexp_extract
import json
from pyspark.sql.functions import to_date, col

# Read parquet file
print(inputFilePathCommand)
input_df = spark.read.parquet(inputFilePathCommand)

new_columns = [col_name.split(".")[-1] for col_name in input_df.columns]
input_df = input_df.toDF(*new_columns)
input_df = target_schema_conversion(input_df)
input_df = input_df.withColumn("create_dtm", current_timestamp()).withColumn("update_dtm", current_timestamp()).withColumn("parquet_source_file_name",regexp_extract(input_df["_metadata.file_path"], r"([^/]+$)", 1))
input_df.cache()
input_df.count()

# COMMAND ----------

# DBTITLE 1,Bronze Table Name
parts = catalog_table_name.split('.')
bronze_table_name = f"{parts[0]}.sch_brnz.{parts[2]}"
print(bronze_table_name)

# COMMAND ----------

#Loading data into Table by creating the raw data as delta format -- bronze layer

if input_df.count() > 0:
    try:
        input_df.write.option("mergeSchema", "true").format("delta").mode("overwrite").saveAsTable(f"{bronze_table_name}")
        files = dbutils.fs.ls(inputFilePathCommand)
        for file in files:
            if file.name.endswith('.parquet'):
                dbutils.fs.mv(file.path, archive_location, True)
        
        # Create JSON format
        json_df = spark.sql(
            f"""
                SELECT * FROM (DESCRIBE HISTORY {bronze_table_name})
                ORDER BY version DESC
                LIMIT 1
                """
        )        
        operation_metrics = json_df.select(col("operationMetrics")).collect()[0][0]
        num_output_rows = operation_metrics.get("numOutputRows", 0)
        max_ins_ts = spark.sql(f"""select date_format( max({incremental_column}), 'yyyy-MM-dd HH:mm:ss' ) as max_ins_ts from {bronze_table_name} """).collect()[0][0]
        success_message = json.loads(json.dumps({"total_rows": num_output_rows, "output_msg":"Data loaded into table", "max_ins_ts":max_ins_ts}))
        # success_message = json.loads(json.dumps({"total_rows": num_output_rows, "output_msg":"Data loaded into table"}))
    except Exception as err:
        failure_message = json.loads(json.dumps({"total_rows": 0, "error_output_msg":str(err)}))
        raise Exception(failure_message)
        dbutils.notebook.exit(failure_message)
else:
    # Moving files to archive location
    # dbutils.fs.mv(inputFilePathCommand, archive_location)
    dbutils.notebook.exit("Data is empty")

# COMMAND ----------

success_message

# COMMAND ----------

# table_name = catalog_table_name.split(".")[-1]
# print("table_name : "+table_name)
# silver_catalog_name = catalog_table_name.split(".")[0]
# print("silver_catalog_name : "+silver_catalog_name)

# COMMAND ----------

max_slvr_create_dtm = spark.sql(f"""select max(create_dtm) as max_create_dtm from {catalog_table_name} """).collect()[0][0]
print("max_slvr_create_dtm :"+str(max_slvr_create_dtm))

# COMMAND ----------

print(f"""SELECT * FROM {bronze_table_name} WHERE create_dtm > '{max_slvr_create_dtm}' """)

# COMMAND ----------


print(f"SELECT * FROM {bronze_table_name} WHERE create_dtm > '{max_slvr_create_dtm}'")

# COMMAND ----------


df = spark.sql(f"""SELECT * FROM {bronze_table_name} WHERE create_dtm > '{max_slvr_create_dtm}' """)

df.count()

# COMMAND ----------

distinct_keys_df = df.select(*pk_cols).dropDuplicates()

distinct_keys_df.createOrReplaceTempView("distinct_keys")

# distinct_keys_df.display()

# COMMAND ----------

from pyspark.sql import functions as F

duplicates_df = (
    distinct_keys_df
    .groupBy(pk_cols)
    .count()
    .filter(F.col("count") > 1)
)

print("Number of duplicated key combinations:", duplicates_df.count())
display(duplicates_df)


# COMMAND ----------

join_condition = "AND ".join([f"{catalog_table_name}.{pk} = b.{pk}" for pk in pk_cols])
print(join_condition)

# COMMAND ----------


deleting_keys = spark.sql(f"""
DELETE FROM {catalog_table_name}
WHERE EXISTS (
    SELECT 1
    FROM distinct_keys b
    WHERE {join_condition}
)
""")
deleting_keys.display()

# COMMAND ----------

catalog_table_name

# COMMAND ----------

df.write.option("mergeSchema", "true").format("delta").mode("overwrite").saveAsTable(f"{catalog_table_name}")

silver_count = spark.sql(f"SELECT count(*) as cnt FROM {catalog_table_name}").collect()[0]["cnt"]

success_message["silver_table_count"] = silver_count

dbutils.notebook.exit(success_message)