# Databricks notebook source
def get_watermark(pipeline_name, driver_table, control_table):
    try:
        df = spark.table(control_table).filter((col("pipeline_name") == pipeline_name) & (col("source_table") == driver_table))
        row = df.select("last_processed_value").limit(1).collect()
        return row[0][0] if row else None
    except:
        raise Exception("Failed reading control table\n" + traceback.format_exc())

# COMMAND ----------

def read_incremental(driver_table, watermark_col, watermark):
    try:
        df = spark.table(driver_table)

        if watermark:
            df = df.filter(col(watermark_col) > lit(watermark))

        return df.select("*", lit(False).alias("from_retry"))

    except:
        raise Exception("Failed reading driver table\n" + traceback.format_exc())

# COMMAND ----------

def update_watermark(incremental_df, control_table, pipeline_name, driver_table, watermark_col):

    max_ts = incremental_df.agg(max(watermark_col)).collect()[0][0]

    if max_ts:
        spark.sql(f"""
        MERGE INTO {control_table} t
        USING (
            SELECT '{pipeline_name}' AS pipeline_name,
                   '{driver_table}' AS source_table,
                   '{watermark_col}' AS watermark_column,
                   TIMESTAMP('{max_ts}') AS last_processed_value,
                   current_timestamp() AS created_ts,
                   current_timestamp() AS updated_ts
        ) s
        ON t.pipeline_name = s.pipeline_name AND t.source_table = s.source_table 

        WHEN MATCHED THEN UPDATE SET
            t.last_processed_value = s.last_processed_value,
            t.updated_ts = s.updated_ts

        WHEN NOT MATCHED THEN INSERT (
            pipeline_name,
            source_table,
            watermark_column,
            last_processed_value,
            created_ts,
            updated_ts
        )
        VALUES (
            s.pipeline_name,
            s.source_table,
            s.watermark_column,
            s.last_processed_value,
            s.created_ts,
            s.updated_ts
        )
        """)
        print(f"Watermark column{watermark_col} value is updated with {max_ts} in the control table {control_table}")

# COMMAND ----------

def handle_retry(final_input, failed_keys, retry_table, driver_pk, watermark_col):
    try:
        if failed_keys.limit(1).count() == 0:
            return

        failed_keys.createOrReplaceTempView("failed_view")

        spark.sql(f"""
        MERGE INTO {retry_table} t
        USING failed_view s
        ON t.primary_key = s.{driver_pk}

        WHEN MATCHED THEN
          UPDATE SET
            t.retry_count = t.retry_count + 1,
            t.status      = 'PENDING',
            t.updated_ts  = current_timestamp()

        WHEN NOT MATCHED THEN
          INSERT (
            primary_key,
            retry_count,
            status,
            created_ts,
            updated_ts
          )
          VALUES (
            s.{driver_pk},
            1,
            'PENDING',
            current_timestamp(),
            current_timestamp()
          )
        """)

    except:
        raise Exception("Failed handling retry\n" + traceback.format_exc())

# COMMAND ----------

from pyspark.sql.functions import lit

def read_retry(driver_df, incremental_df, retry_table, driver_pk):
    try:
        # Step 1: Read retry keys
        retry_keys_df = (
            spark.table(retry_table)
            .filter("status = 'PENDING' AND retry_count <= 5")
              .select(col("primary_key").alias(driver_pk))
        )           

        if retry_keys_df.limit(1).count() == 0:
            return incremental_df.limit(0).withColumn("from_retry", lit(True))

        # Step 2: Cache incremental keys (used twice)
        incr_keys = incremental_df.select(driver_pk).distinct().cache()
        incr_keys.count()

        # Step 3: Remove overlapping keys
        filtered_retry_keys = retry_keys_df.join(
            incr_keys,
            driver_pk,
            "left_anti"
        )

        if filtered_retry_keys.limit(1).count() == 0:
            return incremental_df.limit(0).withColumn("from_retry", lit(True))

        # Step 4: Read driver ONLY for required retry keys
        retry_df = (
            driver_df.join(
                filtered_retry_keys,
                driver_pk,
                "inner"
            )
            .select("*", lit(True).alias("from_retry"))
        )

        return retry_df

    except:
        raise Exception("Failed reading retry data\n" + traceback.format_exc())

# COMMAND ----------

def upsert_retry_tracker(failed_keys_df, retry_tracker_table, src_pk_col):
    # If there are no failed keys, exit early
    if failed_keys_df.limit(1).count() == 0:
        return

    # Create a temp view for failed keys
    failed_keys_df.createOrReplaceTempView("failed_retry_keys")
    #print("from the upsert_retry_tracker function")
    #ailed_keys_df.printSchema()

    #print(f"retry table {retry_tracker_table}")

    # Merge failed keys into retry tracker table
    spark.sql(f"""
        MERGE INTO {retry_tracker_table} t
        USING (
            SELECT {src_pk_col} AS primary_key,
                   1 AS retry_count,
                   'PENDING' AS status,
                   current_timestamp() AS created_ts,
                   current_timestamp() AS updated_ts
            FROM failed_retry_keys
        ) s
        ON t.primary_key = s.primary_key
        WHEN MATCHED THEN UPDATE SET
            t.retry_count = t.retry_count + 1,
            t.status = 'PENDING',
            t.updated_ts = current_timestamp()
        WHEN NOT MATCHED THEN INSERT *
    """)
    print(f"Required Re-process keys are updated into {retry_tracker_table} ")
    # Display failed keys for debugging
    #spark.sql("SELECT * from failed_retry_keys").show(10, False)
    #print("merge is done")

# COMMAND ----------

from pyspark.sql.functions import collect_list, col, struct, current_timestamp, lit, when, to_date
import json

def target_schema_conversion(input_df, catalog_table_name):
    target_catalog_df = spark.table(catalog_table_name)
    # Create lowercase mapping of target schema
    target_schema = {field.name.lower(): field.dataType for field in target_catalog_df.schema.fields}
    # Apply casting by matching lowercase column names
    input_df = input_df.select([
        col(c).cast(target_schema[c.lower()]).alias(c) if c.lower() in target_schema else col(c)
        for c in input_df.columns])
    target_columns = target_catalog_df.columns
    input_order_df = input_df.select([col(c).alias(c.lower()) for c in target_columns])
    return input_order_df
