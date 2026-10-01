# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Databricks Metric View - Interview Prep Guide
# MAGIC
# MAGIC ## What is a Metric View?
# MAGIC
# MAGIC A **Databricks Metric View** is a semantic layer in Unity Catalog where we centrally define reusable business **measures** (metrics) and **dimensions** for consistent KPI analysis across SQL, dashboards, and AI/BI workloads.
# MAGIC
# MAGIC ### Core Components
# MAGIC
# MAGIC ```
# MAGIC                 Metric View
# MAGIC                      |
# MAGIC         ---------------------------
# MAGIC         |                         |
# MAGIC     Measures                  Dimensions
# MAGIC         |                         |
# MAGIC     Revenue                     Region
# MAGIC     Profit                      Product
# MAGIC     Orders                      Date
# MAGIC         |
# MAGIC         ↓
# MAGIC  Central business logic
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Measures vs Dimensions
# MAGIC %md
# MAGIC ## Measures vs Dimensions
# MAGIC
# MAGIC | Aspect | Measures | Dimensions |
# MAGIC |--------|----------|------------|
# MAGIC | **Definition** | Quantitative metrics that can be aggregated | Qualitative attributes used to group/filter data |
# MAGIC | **Type** | Numeric (calculated/aggregated) | Categorical or temporal |
# MAGIC | **Examples** | Revenue, Profit, Order Count, Avg Cost | Region, Product, Date, Customer Type |
# MAGIC | **SQL Operation** | `SUM()`, `AVG()`, `COUNT()`, `MAX()` | `GROUP BY`, `WHERE`, `FILTER` |
# MAGIC | **Question** | "How much?" or "How many?" | "By what?" or "When?" or "Where?" |
# MAGIC
# MAGIC ### Example:
# MAGIC - **Measure**: `Revenue = SUM(amount)` → Answers "How much?"
# MAGIC - **Dimension**: `Region` → Answers "By where?"
# MAGIC - **Query**: "Show me **Revenue** (measure) **by Region** (dimension)"

# COMMAND ----------

# DBTITLE 1,Concrete Example
# MAGIC %md
# MAGIC ## Concrete Example
# MAGIC
# MAGIC ### Source Table: `sales`
# MAGIC
# MAGIC | date       | region | product | amount | cost |
# MAGIC |------------|--------|---------|--------|------|
# MAGIC | 2026-01-01 | North  | Laptop  | 1000   | 700  |
# MAGIC | 2026-01-01 | South  | Mobile  | 500    | 300  |
# MAGIC | 2026-01-02 | North  | Mobile  | 600    | 350  |
# MAGIC | 2026-01-02 | South  | Laptop  | 1200   | 800  |
# MAGIC
# MAGIC ### Metric View Definition
# MAGIC
# MAGIC **Measures:**
# MAGIC - `Revenue = SUM(amount)`
# MAGIC - `Profit = SUM(amount - cost)`
# MAGIC - `Orders = COUNT(*)`
# MAGIC
# MAGIC **Dimensions:**
# MAGIC - `Region`
# MAGIC - `Product`
# MAGIC - `Date`
# MAGIC
# MAGIC ### Reusable Queries
# MAGIC
# MAGIC Now the same `Revenue` definition can be sliced by different dimensions:
# MAGIC
# MAGIC ```sql
# MAGIC -- Revenue by Region
# MAGIC SELECT Region, Revenue FROM metric_view
# MAGIC
# MAGIC -- Revenue by Product
# MAGIC SELECT Product, Revenue FROM metric_view
# MAGIC
# MAGIC -- Revenue by Region + Product
# MAGIC SELECT Region, Product, Revenue FROM metric_view
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Problem It Solves
# MAGIC %md
# MAGIC ## Why Do We Need Metric Views?
# MAGIC
# MAGIC ### ❌ Without Metric View
# MAGIC
# MAGIC **Dashboard A:**
# MAGIC ```sql
# MAGIC SELECT region, SUM(amount) as revenue
# MAGIC FROM sales
# MAGIC GROUP BY region;
# MAGIC ```
# MAGIC
# MAGIC **Dashboard B:**
# MAGIC ```sql
# MAGIC SELECT region, 
# MAGIC        SUM(amount) - SUM(discount) as revenue
# MAGIC FROM sales
# MAGIC GROUP BY region;
# MAGIC ```
# MAGIC
# MAGIC **Problem:** Both call it "Revenue" but use different formulas → **Inconsistent KPIs** 🚨
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ With Metric View
# MAGIC
# MAGIC **Centrally defined once:**
# MAGIC ```
# MAGIC Revenue = SUM(amount) - SUM(discount)
# MAGIC ```
# MAGIC
# MAGIC **Used everywhere consistently:**
# MAGIC - Dashboards
# MAGIC - Genie / AI-BI
# MAGIC - SQL queries
# MAGIC - BI tools
# MAGIC
# MAGIC All use the **same Revenue definition** → **Consistent KPIs** ✓

# COMMAND ----------

# DBTITLE 1,Interview Q&A
# MAGIC %md
# MAGIC ## Interview Q&A
# MAGIC
# MAGIC ### Q1: What is a Databricks Metric View?
# MAGIC
# MAGIC **Answer:**
# MAGIC > "A Databricks Metric View is a semantic layer in Unity Catalog where we centrally define reusable business measures such as revenue, profit, or customer count, along with dimensions such as region, product, and date. This ensures the same KPI definition is reused consistently across SQL, dashboards, and AI/BI workloads."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q2: What's the difference between a Metric View and a Normal View?
# MAGIC
# MAGIC **Answer:**
# MAGIC > "A normal View stores SQL query logic and returns result sets. A Metric View stores semantic business logic—measures, dimensions, joins, and metric definitions—so KPIs can be consistently analyzed across different dimensions without rewriting the business logic."
# MAGIC
# MAGIC **Simple way to remember:**
# MAGIC - **Table** → Raw data
# MAGIC - **View** → SQL logic (query definition)
# MAGIC - **Metric View** → Business/KPI logic (semantic layer)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q3: Give an example of measures vs dimensions
# MAGIC
# MAGIC **Answer:**
# MAGIC > "Measures are quantitative metrics that answer 'how much' or 'how many'—like Revenue = SUM(amount) or Order Count = COUNT(*). Dimensions are qualitative attributes that answer 'by what' or 'where' or 'when'—like Region, Product, or Date. Together, they enable slice-and-dice analysis: for example, 'Show me Revenue (measure) by Region (dimension).' The same measure definition can be grouped by any dimension without redefining the metric."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q4: What components can a Metric View model?
# MAGIC
# MAGIC **Answer:**
# MAGIC > "A Databricks Metric View can model:
# MAGIC > - **Sources**: Base tables or views
# MAGIC > - **Joins**: Relationships between tables
# MAGIC > - **Dimensions**: Categorical/temporal fields for grouping
# MAGIC > - **Measures**: Aggregated metrics with formulas
# MAGIC > - **Filters**: Pre-defined filter logic
# MAGIC > 
# MAGIC > The implementation supports YAML-based semantic definitions for declarative modeling."

# COMMAND ----------

# DBTITLE 1,Creating a Metric View Example
# MAGIC %md
# MAGIC ## How to Create a Metric View in Databricks
# MAGIC
# MAGIC ### Step 1: Create Source Table (Example)

# COMMAND ----------

# DBTITLE 1,Create sample sales table
# MAGIC %sql
# MAGIC -- Create a sample sales table
# MAGIC CREATE OR REPLACE TABLE sales (
# MAGIC   date DATE,
# MAGIC   region STRING,
# MAGIC   product STRING,
# MAGIC   amount DOUBLE,
# MAGIC   cost DOUBLE
# MAGIC );
# MAGIC
# MAGIC INSERT INTO sales VALUES
# MAGIC   ('2026-01-01', 'North', 'Laptop', 1000, 700),
# MAGIC   ('2026-01-01', 'South', 'Mobile', 500, 300),
# MAGIC   ('2026-01-02', 'North', 'Mobile', 600, 350),
# MAGIC   ('2026-01-02', 'South', 'Laptop', 1200, 800),
# MAGIC   ('2026-01-03', 'North', 'Laptop', 1100, 750),
# MAGIC   ('2026-01-03', 'South', 'Mobile', 550, 320);

# COMMAND ----------

# DBTITLE 1,Metric View Creation Note
# MAGIC %md
# MAGIC ### Step 2: Define Metric View
# MAGIC
# MAGIC Metric Views in Databricks are typically created through:
# MAGIC 1. **Unity Catalog UI** - Visual interface for defining measures and dimensions
# MAGIC 2. **YAML definitions** - Declarative semantic model files
# MAGIC 3. **SQL (if supported)** - Using `CREATE METRIC VIEW` syntax
# MAGIC
# MAGIC **Example Conceptual Definition:**
# MAGIC
# MAGIC ```yaml
# MAGIC metric_view: sales_metrics
# MAGIC sources:
# MAGIC   - table: sales
# MAGIC measures:
# MAGIC   - name: Revenue
# MAGIC     expression: SUM(amount)
# MAGIC   - name: Profit
# MAGIC     expression: SUM(amount - cost)
# MAGIC   - name: Orders
# MAGIC     expression: COUNT(*)
# MAGIC dimensions:
# MAGIC   - name: Region
# MAGIC     column: region
# MAGIC   - name: Product
# MAGIC     column: product
# MAGIC   - name: Date
# MAGIC     column: date
# MAGIC ```
# MAGIC
# MAGIC ### Step 3: Query the Metric View
# MAGIC
# MAGIC Once created, you can query it like a regular table:

# COMMAND ----------

# DBTITLE 1,Query examples using metric view
# MAGIC %sql
# MAGIC -- Example queries (assuming metric view 'sales_metrics' exists)
# MAGIC
# MAGIC -- Revenue by Region
# MAGIC SELECT region, SUM(amount) as revenue
# MAGIC FROM sales
# MAGIC GROUP BY region;
# MAGIC
# MAGIC -- Profit by Product
# MAGIC SELECT product, SUM(amount - cost) as profit
# MAGIC FROM sales
# MAGIC GROUP BY product;
# MAGIC
# MAGIC -- Orders by Date
# MAGIC SELECT date, COUNT(*) as orders
# MAGIC FROM sales
# MAGIC GROUP BY date
# MAGIC ORDER BY date;

# COMMAND ----------

# DBTITLE 1,Key Takeaways
# MAGIC %md
# MAGIC ## Key Takeaways for Interview
# MAGIC
# MAGIC ✅ **Metric View = Semantic Layer** for centralized KPI definitions
# MAGIC
# MAGIC ✅ **Measures** (quantitative, aggregated) + **Dimensions** (categorical, grouping)
# MAGIC
# MAGIC ✅ **Single source of truth** → Consistent metrics across all tools
# MAGIC
# MAGIC ✅ **Reusability** → Define once, slice by any dimension
# MAGIC
# MAGIC ✅ **Part of Unity Catalog** → Governed, versioned, discoverable
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Quick Comparison
# MAGIC
# MAGIC | Feature | Normal View | Metric View |
# MAGIC |---------|-------------|-------------|
# MAGIC | Purpose | SQL abstraction | Business logic layer |
# MAGIC | Contains | Query definition | Measures + Dimensions |
# MAGIC | Reusability | Limited to that query | Measures reusable across dimensions |
# MAGIC | Consistency | Can vary | Enforces single definition |
# MAGIC | Use Case | Simplify complex queries | Standardize KPIs |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Remember This Line
# MAGIC
# MAGIC > "Metric Views provide a semantic layer where **measures** define *what* to calculate (like Revenue), and **dimensions** define *how* to slice it (by Region, Product, or Date)—ensuring consistent KPIs across the organization."

# COMMAND ----------

# DBTITLE 1,ABAC Overview
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Unity Catalog ABAC (Attribute-Based Access Control)
# MAGIC
# MAGIC ## What is ABAC?
# MAGIC
# MAGIC **ABAC** (Attribute-Based Access Control) is Unity Catalog's centralized, policy-driven data access control mechanism. Instead of granting permissions table-by-table, you create **policies** that automatically apply **row filters** and **column masks** based on **governed tags** and user attributes.
# MAGIC
# MAGIC ### Core Components
# MAGIC
# MAGIC ```
# MAGIC         ABAC Policy
# MAGIC              |
# MAGIC     -------------------
# MAGIC     |                 |
# MAGIC Row Filters    Column Masks
# MAGIC     |                 |
# MAGIC   Filter rows    Hide/mask values
# MAGIC   based on       based on
# MAGIC   user role      sensitivity tags
# MAGIC ```
# MAGIC
# MAGIC ### Three Building Blocks:
# MAGIC
# MAGIC 1. **Governed Tags** - Metadata attributes applied to tables/columns (e.g., `pii.level=high`, `department=finance`)
# MAGIC 2. **Row Filters** - Policy that controls which rows a user can see
# MAGIC 3. **Column Masks** - Policy that controls whether column values are visible, masked, or hidden

# COMMAND ----------

# DBTITLE 1,ABAC Concrete Example
# MAGIC %md
# MAGIC ## Concrete ABAC Example
# MAGIC
# MAGIC ### Scenario: Employee Data Access Control
# MAGIC
# MAGIC You have an `employees` table with sensitive salary information:
# MAGIC
# MAGIC | emp_id | name          | department | salary  | ssn         |
# MAGIC |--------|---------------|------------|---------|-------------|
# MAGIC | 1      | Alice Johnson | Finance    | 95000   | 123-45-6789 |
# MAGIC | 2      | Bob Smith     | Sales      | 75000   | 987-65-4321 |
# MAGIC | 3      | Carol Lee     | Finance    | 105000  | 555-12-3456 |
# MAGIC | 4      | David Kim     | Engineering| 120000  | 111-22-3333 |
# MAGIC
# MAGIC ### Requirements:
# MAGIC
# MAGIC 1. **Row Filter**: Users can only see employees in their own department
# MAGIC 2. **Column Mask**: Only HR can see full SSN; others see masked values
# MAGIC
# MAGIC ### Step 1: Apply Governed Tags
# MAGIC
# MAGIC ```sql
# MAGIC -- Tag the SSN column as PII
# MAGIC ALTER TABLE employees SET TAGS ('pii.level' = 'high');
# MAGIC ALTER TABLE employees ALTER COLUMN ssn SET TAGS ('pii.type' = 'ssn');
# MAGIC
# MAGIC -- Tag the department column
# MAGIC ALTER TABLE employees ALTER COLUMN department SET TAGS ('classification' = 'department');
# MAGIC ```
# MAGIC
# MAGIC ### Step 2: Create Row Filter Policy
# MAGIC
# MAGIC ```sql
# MAGIC CREATE POLICY department_filter
# MAGIC ON catalog.schema.employees
# MAGIC FOR ROW FILTER
# MAGIC USING (
# MAGIC   -- Users only see rows from their own department
# MAGIC   department = current_user_department() OR
# MAGIC   is_account_group_member('admin_group')
# MAGIC );
# MAGIC ```
# MAGIC
# MAGIC ### Step 3: Create Column Mask Policy
# MAGIC
# MAGIC ```sql
# MAGIC CREATE POLICY ssn_mask
# MAGIC ON catalog.schema.employees
# MAGIC FOR COLUMN MASK
# MAGIC USING (
# MAGIC   CASE
# MAGIC     -- HR sees full SSN
# MAGIC     WHEN is_account_group_member('hr_group') THEN ssn
# MAGIC     -- Others see masked SSN (XXX-XX-1234)
# MAGIC     ELSE concat('XXX-XX-', substring(ssn, 8, 4))
# MAGIC   END
# MAGIC );
# MAGIC ```
# MAGIC
# MAGIC ### Result:
# MAGIC
# MAGIC **Alice (Finance department) queries:**
# MAGIC ```sql
# MAGIC SELECT * FROM employees;
# MAGIC ```
# MAGIC
# MAGIC | emp_id | name          | department | salary | ssn         |
# MAGIC |--------|---------------|------------|--------|-------------|
# MAGIC | 1      | Alice Johnson | Finance    | 95000  | XXX-XX-6789 |
# MAGIC | 3      | Carol Lee     | Finance    | 105000 | XXX-XX-3456 |
# MAGIC
# MAGIC ✅ Only sees Finance employees (row filter)
# MAGIC ✅ SSN is masked (column mask)
# MAGIC
# MAGIC **HR member queries:**
# MAGIC ```sql
# MAGIC SELECT * FROM employees;
# MAGIC ```
# MAGIC
# MAGIC | emp_id | name          | department | salary  | ssn         |
# MAGIC |--------|---------------|------------|---------|-------------|
# MAGIC | 1      | Alice Johnson | Finance    | 95000   | 123-45-6789 |
# MAGIC | 2      | Bob Smith     | Sales      | 75000   | 987-65-4321 |
# MAGIC | 3      | Carol Lee     | Finance    | 105000  | 555-12-3456 |
# MAGIC | 4      | David Kim     | Engineering| 120000  | 111-22-3333 |
# MAGIC
# MAGIC ✅ Sees all employees (admin override)
# MAGIC ✅ Full SSN visible (HR group privilege)

# COMMAND ----------

# DBTITLE 1,ABAC Key Functions
# MAGIC %md
# MAGIC ## ABAC Policy Functions
# MAGIC
# MAGIC ### Tag Condition Functions
# MAGIC
# MAGIC ```sql
# MAGIC -- Check if table has a specific tag key
# MAGIC has_tag('pii.level')
# MAGIC
# MAGIC -- Check if table has a tag with specific value
# MAGIC has_tag_value('pii.level', 'high')
# MAGIC
# MAGIC -- Check if a column has a specific tag
# MAGIC has_column_tag('ssn', 'pii.type')
# MAGIC ```
# MAGIC
# MAGIC ### User Context Functions
# MAGIC
# MAGIC ```sql
# MAGIC -- Current user's email
# MAGIC current_user()
# MAGIC
# MAGIC -- Check group membership
# MAGIC is_account_group_member('hr_group')
# MAGIC
# MAGIC -- Custom UDF for department lookup
# MAGIC current_user_department()  -- Returns user's department
# MAGIC ```
# MAGIC
# MAGIC ### Policy Management
# MAGIC
# MAGIC ```sql
# MAGIC -- Show all policies
# MAGIC SHOW POLICIES ON catalog.schema.table;
# MAGIC
# MAGIC -- Describe specific policy
# MAGIC DESCRIBE POLICY policy_name ON catalog.schema.table;
# MAGIC
# MAGIC -- Drop policy
# MAGIC DROP POLICY policy_name ON catalog.schema.table;
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,ABAC vs Traditional Access Control
# MAGIC %md
# MAGIC ## ABAC vs Traditional Access Control
# MAGIC
# MAGIC | Aspect | Traditional (Table-Level) | ABAC (Attribute-Based) |
# MAGIC |--------|---------------------------|------------------------|
# MAGIC | **Granularity** | Table or schema level | Row and column level |
# MAGIC | **Scope** | Per-object grants | Centralized policies |
# MAGIC | **Maintenance** | Grant per table/user | Policy applies automatically |
# MAGIC | **Scalability** | Manual, doesn't scale | Automatic, scales |
# MAGIC | **Flexibility** | Rigid | Dynamic based on attributes |
# MAGIC | **Example** | `GRANT SELECT ON table TO user` | `CREATE POLICY ... USING (department = user_dept)` |
# MAGIC
# MAGIC ### Why ABAC?
# MAGIC
# MAGIC **Traditional approach:**
# MAGIC ```sql
# MAGIC -- Need separate view per department
# MAGIC CREATE VIEW finance_employees AS SELECT * FROM employees WHERE department = 'Finance';
# MAGIC CREATE VIEW sales_employees AS SELECT * FROM employees WHERE department = 'Sales';
# MAGIC ...
# MAGIC GRANT SELECT ON finance_employees TO finance_group;
# MAGIC GRANT SELECT ON sales_employees TO sales_group;
# MAGIC ```
# MAGIC ❌ **Problem**: One view per department/role → maintenance nightmare
# MAGIC
# MAGIC **ABAC approach:**
# MAGIC ```sql
# MAGIC -- Single policy on the source table
# MAGIC CREATE POLICY department_filter
# MAGIC FOR ROW FILTER
# MAGIC USING (department = current_user_department());
# MAGIC ```
# MAGIC ✅ **Solution**: One policy applies dynamically to all users

# COMMAND ----------

# DBTITLE 1,ABAC Interview Q&A
# MAGIC %md
# MAGIC ## ABAC Interview Q&A
# MAGIC
# MAGIC ### Q1: What is ABAC in Unity Catalog?
# MAGIC
# MAGIC **Answer:**
# MAGIC > "ABAC (Attribute-Based Access Control) is Unity Catalog's centralized, policy-driven data access control mechanism. Instead of granting permissions table-by-table, you create policies that automatically apply row filters and column masks based on governed tags and user attributes. For example, a row filter policy can ensure users only see data from their own department, and a column mask policy can hide or redact sensitive columns like SSN based on the user's role."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q2: What are the key components of ABAC?
# MAGIC
# MAGIC **Answer:**
# MAGIC > "ABAC has three key components:
# MAGIC > 1. **Governed Tags** - Metadata attributes applied to tables and columns to classify data sensitivity or ownership
# MAGIC > 2. **Row Filters** - Policies that control which rows a user can see based on conditions like department, region, or role
# MAGIC > 3. **Column Masks** - Policies that control whether sensitive column values are visible, masked (partially hidden), or completely redacted
# MAGIC > 
# MAGIC > Together, these enable fine-grained, dynamic access control without creating multiple views or manual grants."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q3: How does ABAC differ from traditional table-level permissions?
# MAGIC
# MAGIC **Answer:**
# MAGIC > "Traditional access control grants permissions at the table or schema level—either you have access to the whole table or you don't. ABAC provides row-level and column-level security through centralized policies. For example, instead of creating separate views per department and granting access to each, you write one row filter policy that dynamically filters data based on the user's department attribute. This scales much better and reduces maintenance overhead."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q4: Give a real-world ABAC use case
# MAGIC
# MAGIC **Answer:**
# MAGIC > "Consider an employees table with salary and SSN data. With ABAC, I can create a row filter policy so users only see employees in their own department, and a column mask policy so only the HR group sees full SSN values—others see masked values like 'XXX-XX-1234'. The policies apply automatically based on the querying user's attributes, so there's no need to create separate department-specific views or manage per-user grants. The same table serves all users with appropriate, dynamic access control."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Remember This Line
# MAGIC
# MAGIC > "ABAC enables **row-level and column-level security** through **centralized policies** that apply dynamically based on **governed tags** and **user attributes**—eliminating the need for multiple views and manual per-user grants."

# COMMAND ----------

