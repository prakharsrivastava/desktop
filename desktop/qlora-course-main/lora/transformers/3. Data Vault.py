# Databricks notebook source
# DBTITLE 1,Data Vault Introduction
# MAGIC %md
# MAGIC # Data Vault - Hybrid Approach
# MAGIC
# MAGIC ## Ye hybrid approach hai
# MAGIC
# MAGIC **Data Vault** combines best of:
# MAGIC * 3NF (Third Normal Form)
# MAGIC * Dimensional Modeling (Star Schema)
# MAGIC
# MAGIC ## Purpose:
# MAGIC
# MAGIC ✅ **Flexible warehouse** - Easily adapt to new requirements
# MAGIC
# MAGIC ✅ **Scalable** - Parallel loading, fast queries
# MAGIC
# MAGIC ✅ **Schema change friendly** - Add sources without breaking existing structure
# MAGIC
# MAGIC ✅ **Historical tracking** - Complete audit trail
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Ye especially modern enterprise DW me use hota hai
# MAGIC
# MAGIC Big companies like banks, telecom, retail use Data Vault because:
# MAGIC * Multiple data sources constantly changing
# MAGIC * Need to track history
# MAGIC * Schema evolution without disruption
# MAGIC * Regulatory compliance (audit trail)

# COMMAND ----------

# DBTITLE 1,Setup - Import Libraries
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, md5, concat_ws, current_timestamp, desc, row_number
from pyspark.sql.window import Window
from datetime import datetime

print("Data Vault Setup Complete")

# COMMAND ----------

# DBTITLE 1,Data Vault ke 3 Components
# MAGIC %md
# MAGIC # Data Vault ke 3 Components
# MAGIC
# MAGIC ## 1. Hub - Business Key
# MAGIC
# MAGIC Core business entities (unique identifiers)
# MAGIC
# MAGIC **Example:**
# MAGIC * Customer_ID
# MAGIC * Employee_ID
# MAGIC * Product_ID
# MAGIC * Order_ID
# MAGIC
# MAGIC **Structure:**
# MAGIC ```
# MAGIC HUB_CUSTOMER
# MAGIC ├── customer_hk (hash key - primary key)
# MAGIC ├── customer_id (business key)
# MAGIC ├── load_dts (load timestamp)
# MAGIC └── record_source (source system)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Link - Relationship between hubs
# MAGIC
# MAGIC Connects multiple hubs together
# MAGIC
# MAGIC **Example:**
# MAGIC * Customer **buys** Product
# MAGIC * Employee **works in** Department
# MAGIC * Order **contains** Product
# MAGIC
# MAGIC **Structure:**
# MAGIC ```
# MAGIC LINK_CUSTOMER_ORDER
# MAGIC ├── customer_order_hk (link hash key)
# MAGIC ├── customer_hk (foreign key to Hub)
# MAGIC ├── order_hk (foreign key to Hub)
# MAGIC ├── load_dts
# MAGIC └── record_source
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Satellite - Descriptive attributes
# MAGIC
# MAGIC All the details and attributes
# MAGIC
# MAGIC **Example:**
# MAGIC * Customer: name, address, DOB, email, phone
# MAGIC * Employee: salary, designation, joining_date
# MAGIC * Product: name, price, category, description
# MAGIC
# MAGIC **Structure:**
# MAGIC ```
# MAGIC SAT_CUSTOMER
# MAGIC ├── customer_hk (foreign key to Hub)
# MAGIC ├── load_dts (when this version was loaded)
# MAGIC ├── load_end_dts (when this version expired)
# MAGIC ├── name
# MAGIC ├── address
# MAGIC ├── dob
# MAGIC ├── email
# MAGIC ├── hash_diff (detect changes)
# MAGIC └── record_source
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Example 1 - Create Sample Data (Day 1)
# Aaj sirf customer details aa rahi hai
# Day 1: Only customer data arrives

print("=== Day 1: Customer Data Arrives ===")

customer_data_day1 = [
    (1001, 'Rahul Kumar', 'Delhi', '1990-05-15', 'rahul@email.com'),
    (1002, 'Priya Sharma', 'Mumbai', '1992-08-20', 'priya@email.com'),
    (1003, 'Amit Patel', 'Bangalore', '1988-03-10', 'amit@email.com'),
]

customers_df = spark.createDataFrame(
    customer_data_day1,
    ['customer_id', 'name', 'city', 'dob', 'email']
)

print("Customer data loaded:")
display(customers_df)

# COMMAND ----------

# DBTITLE 1,Create Hub - Customer_ID
# Hub: Business key (Customer_ID)

HUB_CUSTOMER = customers_df.select(
    md5(concat_ws('|', col('customer_id').cast('string'))).alias('customer_hk'),  # Hash key
    col('customer_id').alias('customer_bk'),  # Business key
    current_timestamp().alias('load_dts'),
    lit('CRM_SYSTEM').alias('record_source')
).distinct()

print("HUB_CUSTOMER - Business Keys:")
print("Sirf customer_id store hoti hai, attributes nahi")
display(HUB_CUSTOMER)

# COMMAND ----------

# DBTITLE 1,Create Satellite - Customer Attributes
# Satellite: Descriptive attributes (name, address, DOB, email)

SAT_CUSTOMER = customers_df.select(
    md5(concat_ws('|', col('customer_id').cast('string'))).alias('customer_hk'),
    current_timestamp().alias('load_dts'),
    lit(None).cast('timestamp').alias('load_end_dts'),  # NULL = current/active
    col('name'),
    col('city'),
    col('dob'),
    col('email'),
    md5(concat_ws('|', col('name'), col('city'), col('dob'), col('email'))).alias('hash_diff'),
    lit('CRM_SYSTEM').alias('record_source')
)

print("SAT_CUSTOMER - Descriptive Attributes:")
print("Saare details yaha store hote hai")
display(SAT_CUSTOMER)

# COMMAND ----------

# DBTITLE 1,Why Companies Use Data Vault
# MAGIC %md
# MAGIC # Why Companies Use Data Vault
# MAGIC
# MAGIC ## Real World Scenario:
# MAGIC
# MAGIC ### Suppose:
# MAGIC **Aaj** sirf customer details aa rahi thi (name, city, DOB, email)
# MAGIC
# MAGIC ### Kal:
# MAGIC **Tax data aa gaya** (PAN number, tax bracket, filing status)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Traditional Approaches:
# MAGIC
# MAGIC ### 3NF me:
# MAGIC ```sql
# MAGIC -- Bahut schema change karna padega
# MAGIC ALTER TABLE customer ADD COLUMN pan_number VARCHAR(10);
# MAGIC ALTER TABLE customer ADD COLUMN tax_bracket VARCHAR(20);
# MAGIC ALTER TABLE customer ADD COLUMN filing_status VARCHAR(20);
# MAGIC
# MAGIC -- Existing queries break ho sakte hai
# MAGIC -- ETL pipelines update karne padenge
# MAGIC ```
# MAGIC
# MAGIC ### Dimensional (Star Schema) me:
# MAGIC ```sql
# MAGIC -- Fact table redesign
# MAGIC -- Dimension table restructure
# MAGIC -- Historical data migration
# MAGIC -- Aggregations recalculate
# MAGIC ```
# MAGIC
# MAGIC ### Data Vault me:
# MAGIC ```sql
# MAGIC -- Bas new satellite add karo. Done!
# MAGIC CREATE TABLE SAT_CUSTOMER_TAX (
# MAGIC     customer_hk STRING,
# MAGIC     load_dts TIMESTAMP,
# MAGIC     pan_number STRING,
# MAGIC     tax_bracket STRING,
# MAGIC     filing_status STRING,
# MAGIC     record_source STRING
# MAGIC );
# MAGIC
# MAGIC -- Hub aur existing satellite unchanged!
# MAGIC -- Purani queries same chalti rahegi
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Isi liye:
# MAGIC
# MAGIC ✅ **Agile** - Quick adaptation to new sources
# MAGIC
# MAGIC ✅ **Scalable** - Add tables without touching existing ones
# MAGIC
# MAGIC ✅ **Historical** - Complete audit trail maintained
# MAGIC
# MAGIC ✅ **Schema evolution friendly** - No breaking changes

# COMMAND ----------

# DBTITLE 1,Day 2 - Tax Data Arrives
# Kal: Tax data aa gaya
# Day 2: New data source - Tax information

print("=== Day 2: Tax Data Arrives ===")

tax_data = [
    (1001, 'ABCDE1234F', '30% bracket', 'Filed'),
    (1002, 'XYZAB5678G', '20% bracket', 'Filed'),
    (1003, 'PQRST9012H', '30% bracket', 'Pending'),
]

tax_df = spark.createDataFrame(
    tax_data,
    ['customer_id', 'pan_number', 'tax_bracket', 'filing_status']
)

print("Tax data loaded from Tax Department:")
display(tax_df)

# COMMAND ----------

# DBTITLE 1,Create New Satellite - Tax Data
# Data Vault me: Bas new satellite add karo!
# No changes to Hub or existing Satellite

SAT_CUSTOMER_TAX = tax_df.select(
    md5(concat_ws('|', col('customer_id').cast('string'))).alias('customer_hk'),
    current_timestamp().alias('load_dts'),
    lit(None).cast('timestamp').alias('load_end_dts'),
    col('pan_number'),
    col('tax_bracket'),
    col('filing_status'),
    md5(concat_ws('|', col('pan_number'), col('tax_bracket'), col('filing_status'))).alias('hash_diff'),
    lit('TAX_DEPT').alias('record_source')
)

print("SAT_CUSTOMER_TAX - New Satellite Added:")
print("Hub aur SAT_CUSTOMER unchanged!")
display(SAT_CUSTOMER_TAX)

# COMMAND ----------

# DBTITLE 1,Business View - Join All Data
# Business View: Hub + Multiple Satellites
# Easily join karo jab chahiye

customer_complete_view = HUB_CUSTOMER \
    .join(SAT_CUSTOMER, 'customer_hk', 'left') \
    .join(SAT_CUSTOMER_TAX, 'customer_hk', 'left') \
    .select(
        col('customer_bk').alias('customer_id'),
        SAT_CUSTOMER['name'],
        SAT_CUSTOMER['city'],
        SAT_CUSTOMER['email'],
        SAT_CUSTOMER_TAX['pan_number'],
        SAT_CUSTOMER_TAX['tax_bracket'],
        SAT_CUSTOMER_TAX['filing_status']
    )

print("Complete Customer View (CRM + Tax Data):")
print("Dono sources ka data ek saath!")
display(customer_complete_view)

# COMMAND ----------

# DBTITLE 1,Historical Tracking Example
# Suppose customer address change ho gaya
# Data Vault automatically tracks history

print("=== Day 3: Rahul moved from Delhi to Pune ===")

# New record arrives with updated city
updated_customer = [
    (1001, 'Rahul Kumar', 'Pune', '1990-05-15', 'rahul@email.com'),  # City changed
]

updated_df = spark.createDataFrame(
    updated_customer,
    ['customer_id', 'name', 'city', 'dob', 'email']
)

# Create new satellite record (insert, not update!)
SAT_CUSTOMER_NEW = updated_df.select(
    md5(concat_ws('|', col('customer_id').cast('string'))).alias('customer_hk'),
    current_timestamp().alias('load_dts'),
    lit(None).cast('timestamp').alias('load_end_dts'),
    col('name'),
    col('city'),
    col('dob'),
    col('email'),
    md5(concat_ws('|', col('name'), col('city'), col('dob'), col('email'))).alias('hash_diff'),
    lit('CRM_SYSTEM').alias('record_source')
)

# Union old and new records
SAT_CUSTOMER_HISTORY = SAT_CUSTOMER.union(SAT_CUSTOMER_NEW)

print("Satellite with History:")
print("Purana record: Delhi")
print("Naya record: Pune")
print("Dono preserved!")

# Show history for customer 1001
rahul_history = SAT_CUSTOMER_HISTORY.filter(
    col('customer_hk') == md5(concat_ws('|', lit('1001')))
).orderBy('load_dts')

display(rahul_history)

# COMMAND ----------

# DBTITLE 1,Point-in-Time Query
# Historical query: Customer ka address kya tha specific date pe?
# Example: What was the address on Day 1?

print("Point-in-Time Query: Customer details as of Day 1")
print("Rahul ka address Day 1 pe: Delhi")
print("Rahul ka address Day 3 pe: Pune")

# Get latest record per customer
window_spec = Window.partitionBy('customer_hk').orderBy(desc('load_dts'))

current_state = SAT_CUSTOMER_HISTORY \
    .withColumn('row_num', row_number().over(window_spec)) \
    .filter(col('row_num') == 1) \
    .drop('row_num')

current_view = HUB_CUSTOMER \
    .join(current_state, 'customer_hk', 'inner') \
    .select(
        col('customer_bk').alias('customer_id'),
        col('name'),
        col('city').alias('current_city'),
        col('email')
    )

print("\nCurrent State (Latest):")
display(current_view)

# COMMAND ----------

# DBTITLE 1,Example - Order Data with Links
# Link example: Customer buys Product
# Multiple hubs connect karte hai through Link

print("=== Adding Order Data with Links ===")

# Order data
orders = [
    (2001, 1001, '2024-05-01', 5000.00),
    (2002, 1002, '2024-05-02', 3500.00),
    (2003, 1001, '2024-05-10', 8000.00),
]

orders_df = spark.createDataFrame(
    orders,
    ['order_id', 'customer_id', 'order_date', 'amount']
)

# Hub Order
HUB_ORDER = orders_df.select(
    md5(concat_ws('|', col('order_id').cast('string'))).alias('order_hk'),
    col('order_id').alias('order_bk'),
    current_timestamp().alias('load_dts'),
    lit('ORDER_SYSTEM').alias('record_source')
).distinct()

# Link: Customer -> Order
LINK_CUSTOMER_ORDER = orders_df.select(
    md5(concat_ws('|',
        md5(concat_ws('|', col('customer_id').cast('string'))),  # customer_hk
        md5(concat_ws('|', col('order_id').cast('string')))      # order_hk
    )).alias('customer_order_hk'),
    md5(concat_ws('|', col('customer_id').cast('string'))).alias('customer_hk'),
    md5(concat_ws('|', col('order_id').cast('string'))).alias('order_hk'),
    current_timestamp().alias('load_dts'),
    lit('ORDER_SYSTEM').alias('record_source')
).distinct()

# Satellite Order
SAT_ORDER = orders_df.select(
    md5(concat_ws('|', col('order_id').cast('string'))).alias('order_hk'),
    current_timestamp().alias('load_dts'),
    lit(None).cast('timestamp').alias('load_end_dts'),
    col('order_date'),
    col('amount'),
    md5(concat_ws('|', col('order_date').cast('string'), col('amount').cast('string'))).alias('hash_diff'),
    lit('ORDER_SYSTEM').alias('record_source')
)

print("HUB_ORDER created")
print("LINK_CUSTOMER_ORDER created")
print("SAT_ORDER created")
print("\nLink Table:")
display(LINK_CUSTOMER_ORDER)

# COMMAND ----------

# DBTITLE 1,Complete Business View - Customer Orders
# Complete view: Customer details + Orders
# Multiple hubs and satellites joined

# Get current customer info
window_spec = Window.partitionBy('customer_hk').orderBy(desc('load_dts'))
current_customers = SAT_CUSTOMER_HISTORY \
    .withColumn('row_num', row_number().over(window_spec)) \
    .filter(col('row_num') == 1) \
    .drop('row_num')

# Join everything
customer_orders_view = HUB_CUSTOMER \
    .join(current_customers, 'customer_hk', 'inner') \
    .join(LINK_CUSTOMER_ORDER, 'customer_hk', 'inner') \
    .join(HUB_ORDER, 'order_hk', 'inner') \
    .join(SAT_ORDER, 'order_hk', 'inner') \
    .select(
        col('customer_bk').alias('customer_id'),
        current_customers['name'].alias('customer_name'),
        current_customers['city'],
        col('order_bk').alias('order_id'),
        SAT_ORDER['order_date'],
        SAT_ORDER['amount']
    ) \
    .orderBy('customer_id', 'order_date')

print("Complete Business View:")
print("Hub + Link + Satellite data combined!")
display(customer_orders_view)

# COMMAND ----------

# DBTITLE 1,Summary - Data Vault Benefits
# MAGIC %md
# MAGIC # Data Vault Benefits Summary
# MAGIC
# MAGIC ## ✅ Schema Change Friendly
# MAGIC
# MAGIC **Traditional:**
# MAGIC ```
# MAGIC Day 1: Customer data arrives
# MAGIC Day 2: Tax data arrives → ALTER TABLE, migrations, ETL changes
# MAGIC Day 3: Address changes → Complex UPDATE logic
# MAGIC ```
# MAGIC
# MAGIC **Data Vault:**
# MAGIC ```
# MAGIC Day 1: HUB_CUSTOMER + SAT_CUSTOMER
# MAGIC Day 2: Add SAT_CUSTOMER_TAX → No changes to existing tables!
# MAGIC Day 3: New record in SAT_CUSTOMER → History automatically maintained!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Historical Tracking
# MAGIC
# MAGIC * Every change creates a new row (INSERT, not UPDATE)
# MAGIC * Point-in-time queries easy
# MAGIC * Full audit trail
# MAGIC * No data loss
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Scalable
# MAGIC
# MAGIC * Hubs loaded independently
# MAGIC * Satellites loaded in parallel
# MAGIC * Links loaded separately
# MAGIC * Hash keys for fast joins
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Flexible
# MAGIC
# MAGIC * Multiple satellites per hub
# MAGIC * Different sources → Different satellites
# MAGIC * Business views combine as needed
# MAGIC * Add/remove without breaking
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Real World Usage
# MAGIC
# MAGIC **Banking:**
# MAGIC * HUB_CUSTOMER, SAT_KYC, SAT_CREDIT, SAT_TRANSACTIONS
# MAGIC
# MAGIC **Retail:**
# MAGIC * HUB_PRODUCT, SAT_INVENTORY, SAT_PRICING, SAT_SUPPLIER
# MAGIC
# MAGIC **Healthcare:**
# MAGIC * HUB_PATIENT, SAT_DEMOGRAPHICS, SAT_MEDICAL_HISTORY, SAT_INSURANCE
# MAGIC
# MAGIC **Telecom:**
# MAGIC * HUB_SUBSCRIBER, SAT_PLAN, SAT_USAGE, SAT_BILLING
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Implementation in Databricks
# MAGIC
# MAGIC ```python
# MAGIC # Use Delta tables for ACID
# MAGIC HUB_CUSTOMER.write.format('delta').mode('append').saveAsTable('hub_customer')
# MAGIC
# MAGIC # Use Lakeflow Pipelines for incremental loading
# MAGIC # Materialized views for business queries
# MAGIC # Unity Catalog for governance
# MAGIC ```