# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Simple Databricks Examples
# MAGIC
# MAGIC This notebook demonstrates:
# MAGIC 1. **Measures vs Dimensions** - Core concepts with working SQL
# MAGIC 2. **ABAC (Attribute-Based Access Control)** - Row filtering and column masking examples

# COMMAND ----------

# DBTITLE 1,Sales Data Example
# MAGIC %md
# MAGIC ## Part 1: Measures vs Dimensions
# MAGIC
# MAGIC Let's create a simple sales table and demonstrate measure/dimension concepts.

# COMMAND ----------

# DBTITLE 1,Create sales table
# MAGIC %sql
# MAGIC -- Create sample sales data
# MAGIC CREATE OR REPLACE TABLE demo_sales (
# MAGIC   date DATE,
# MAGIC   region STRING,
# MAGIC   product STRING,
# MAGIC   amount DOUBLE,
# MAGIC   cost DOUBLE
# MAGIC );
# MAGIC
# MAGIC INSERT INTO demo_sales VALUES
# MAGIC   ('2026-01-01', 'North', 'Laptop', 1000, 700),
# MAGIC   ('2026-01-01', 'South', 'Mobile', 500, 300),
# MAGIC   ('2026-01-02', 'North', 'Mobile', 600, 350),
# MAGIC   ('2026-01-02', 'South', 'Laptop', 1200, 800),
# MAGIC   ('2026-01-03', 'North', 'Laptop', 1100, 750),
# MAGIC   ('2026-01-03', 'South', 'Mobile', 550, 320);
# MAGIC
# MAGIC SELECT * FROM demo_sales;

# COMMAND ----------

# DBTITLE 1,Measures Explanation
# MAGIC %md
# MAGIC ### Measures (Aggregations)
# MAGIC
# MAGIC Measures answer **"How much?"** or **"How many?"**
# MAGIC
# MAGIC - Revenue = SUM(amount)
# MAGIC - Profit = SUM(amount - cost)
# MAGIC - Order Count = COUNT(*)

# COMMAND ----------

# DBTITLE 1,Revenue by Region (Dimension)
# MAGIC %sql
# MAGIC -- Revenue (MEASURE) by Region (DIMENSION)
# MAGIC SELECT 
# MAGIC   region,
# MAGIC   SUM(amount) as revenue,
# MAGIC   SUM(amount - cost) as profit,
# MAGIC   COUNT(*) as orders
# MAGIC FROM demo_sales
# MAGIC GROUP BY region;

# COMMAND ----------

# DBTITLE 1,Revenue by Product (Dimension)
# MAGIC %sql
# MAGIC -- Same MEASURES, different DIMENSION (Product)
# MAGIC SELECT 
# MAGIC   product,
# MAGIC   SUM(amount) as revenue,
# MAGIC   SUM(amount - cost) as profit,
# MAGIC   COUNT(*) as orders
# MAGIC FROM demo_sales
# MAGIC GROUP BY product;

# COMMAND ----------

# DBTITLE 1,Revenue by Date (Dimension)
# MAGIC %sql
# MAGIC -- Same MEASURES, different DIMENSION (Date)
# MAGIC SELECT 
# MAGIC   date,
# MAGIC   SUM(amount) as revenue,
# MAGIC   SUM(amount - cost) as profit,
# MAGIC   COUNT(*) as orders
# MAGIC FROM demo_sales
# MAGIC GROUP BY date
# MAGIC ORDER BY date;

# COMMAND ----------

# DBTITLE 1,Key Insight
# MAGIC %md
# MAGIC ### 💡 Key Insight
# MAGIC
# MAGIC The **same measures** (Revenue, Profit, Orders) are reused across different **dimensions** (Region, Product, Date).
# MAGIC
# MAGIC This is exactly what a **Metric View** formalizes:
# MAGIC - Define measures ONCE
# MAGIC - Slice by ANY dimension
# MAGIC - Consistent KPIs everywhere

# COMMAND ----------

# DBTITLE 1,ABAC Section
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Part 2: ABAC (Row Filter + Column Mask)
# MAGIC
# MAGIC Let's demonstrate row-level and column-level security concepts.

# COMMAND ----------

# DBTITLE 1,Create employee table
# MAGIC %sql
# MAGIC -- Create employee table with sensitive data
# MAGIC CREATE OR REPLACE TABLE demo_employees (
# MAGIC   emp_id INT,
# MAGIC   name STRING,
# MAGIC   department STRING,
# MAGIC   salary DOUBLE,
# MAGIC   ssn STRING
# MAGIC );
# MAGIC
# MAGIC INSERT INTO demo_employees VALUES
# MAGIC   (1, 'Alice Johnson', 'Finance', 95000, '123-45-6789'),
# MAGIC   (2, 'Bob Smith', 'Sales', 75000, '987-65-4321'),
# MAGIC   (3, 'Carol Lee', 'Finance', 105000, '555-12-3456'),
# MAGIC   (4, 'David Kim', 'Engineering', 120000, '111-22-3333');
# MAGIC
# MAGIC SELECT * FROM demo_employees;

# COMMAND ----------

# DBTITLE 1,Row Filter Concept
# MAGIC %md
# MAGIC ### Row Filter Example
# MAGIC
# MAGIC Simulating: **Users only see employees in their own department**

# COMMAND ----------

# DBTITLE 1,Row filter simulation
# MAGIC %sql
# MAGIC -- Simulating Alice (Finance) view
# MAGIC -- In real ABAC, this filter applies automatically based on current_user()
# MAGIC SELECT * FROM demo_employees
# MAGIC WHERE department = 'Finance';  -- Alice only sees Finance employees

# COMMAND ----------

# DBTITLE 1,Column Mask Concept
# MAGIC %md
# MAGIC ### Column Mask Example
# MAGIC
# MAGIC Simulating: **Only HR sees full SSN, others see masked**

# COMMAND ----------

# DBTITLE 1,Column mask simulation
# MAGIC %sql
# MAGIC -- Simulating non-HR user view (masked SSN)
# MAGIC SELECT 
# MAGIC   emp_id,
# MAGIC   name,
# MAGIC   department,
# MAGIC   salary,
# MAGIC   concat('XXX-XX-', substring(ssn, 8, 4)) as ssn  -- Masked SSN
# MAGIC FROM demo_employees
# MAGIC WHERE department = 'Finance';

# COMMAND ----------

# DBTITLE 1,HR view (full access)
# MAGIC %sql
# MAGIC -- Simulating HR user view (full SSN visible)
# MAGIC SELECT 
# MAGIC   emp_id,
# MAGIC   name,
# MAGIC   department,
# MAGIC   salary,
# MAGIC   ssn  -- Full SSN for HR
# MAGIC FROM demo_employees;  -- HR sees all departments

# COMMAND ----------

# DBTITLE 1,ABAC Summary
# MAGIC %md
# MAGIC ### 💡 ABAC in Real Implementation
# MAGIC
# MAGIC Instead of manually writing WHERE clauses and CASE statements in every query:
# MAGIC
# MAGIC ```sql
# MAGIC CREATE POLICY department_filter
# MAGIC FOR ROW FILTER
# MAGIC USING (department = current_user_department());
# MAGIC
# MAGIC CREATE POLICY ssn_mask
# MAGIC FOR COLUMN MASK
# MAGIC USING (
# MAGIC   CASE
# MAGIC     WHEN is_account_group_member('hr_group') THEN ssn
# MAGIC     ELSE concat('XXX-XX-', substring(ssn, 8, 4))
# MAGIC   END
# MAGIC );
# MAGIC ```
# MAGIC
# MAGIC ✅ Policy applies **automatically** to every query
# MAGIC
# MAGIC ✅ No code changes needed in applications
# MAGIC
# MAGIC ✅ Centralized security governance

# COMMAND ----------

# DBTITLE 1,Cleanup
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Cleanup (Optional)

# COMMAND ----------

# DBTITLE 1,Drop demo tables
# MAGIC %sql
# MAGIC -- Uncomment to clean up demo tables
# MAGIC -- DROP TABLE IF EXISTS demo_sales;
# MAGIC -- DROP TABLE IF EXISTS demo_employees;

# COMMAND ----------

