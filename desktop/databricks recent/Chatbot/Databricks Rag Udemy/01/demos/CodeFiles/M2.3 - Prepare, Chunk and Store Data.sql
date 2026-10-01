-- Databricks notebook source
-- MAGIC %md
-- MAGIC ### A. Setup environment
-- MAGIC
-- MAGIC Install libraries and check files

-- COMMAND ----------

-- DBTITLE 1,Install dependencies - langchain
-- MAGIC %python
-- MAGIC
-- MAGIC %pip install langchain langchain-text-splitters
-- MAGIC
-- MAGIC dbutils.library.restartPython()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B. Get list of unprocessed files

-- COMMAND ----------

-- DBTITLE 1,Check files in volume
-- MAGIC %python
-- MAGIC
-- MAGIC display(
-- MAGIC           dbutils.fs.ls("/Volumes/insurance/rag/raw/")
-- MAGIC        )

-- COMMAND ----------

-- DBTITLE 1,Join files in volume to processed files to find unprocessed files
-- Create view to find files that are not yet processed
CREATE TEMPORARY VIEW vwFilesToProcess
AS

    SELECT allFiles.path,
           allFiles._metadata.file_name AS filename

    FROM READ_FILES('/Volumes/insurance/rag/raw/', format => 'binaryFile') AS allFiles

        LEFT ANTI JOIN insurance.rag.ProcessedFiles AS processedFiles      ON allFiles.path = processedFiles.FilePath;


-- Query view to check which files to process
SELECT * FROM vwFilesToProcess;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### C. Extract file (PDF) content

-- COMMAND ----------

-- DBTITLE 1,Use ai_parse_document method
-- Create view to get parsed content of those files that are not yet processed
CREATE TEMPORARY VIEW vwParseDocuments
AS
    SELECT ai_parse_document(content) AS parsed_document

        , filesToProcess.path
        , filesToProcess.filename

    FROM READ_FILES('/Volumes/insurance/rag/raw/', format => 'binaryFile') AS allFiles
    
        JOIN vwFilesToProcess AS filesToProcess      ON allFiles.path = filesToProcess.path;


-- Query view to check parsed content of files
SELECT * FROM vwParseDocuments;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### D. Create data chunks

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### D.1 Format-specific chunking

-- COMMAND ----------

-- DBTITLE 1,Extract elements from content array
SELECT path
     , item.content AS content
     , item.bbox.page_id[0] AS pageId

FROM 
(
    -- Extract content and page ID from parsed content of documents
    SELECT transform(
                         parsed_document:document.elements
                             ::ARRAY<STRUCT<content:STRING, bbox:ARRAY<STRUCT<page_id:INT>>>>
                        
                       , x -> x
                    ) AS content

         , path

    FROM vwParseDocuments    
)

-- Put array items of content into separate rows
LATERAL VIEW explode(content) exploded_table AS item

LIMIT 10 -- Only used to execute query faster and show limited results. Remove in production

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### D.2 Recursive chunking (using Langchain's recursive character text splitter)

-- COMMAND ----------

-- DBTITLE 1,Create split method and register as UDF
-- MAGIC %python
-- MAGIC
-- MAGIC # Import libraries
-- MAGIC from pyspark.sql.types import *
-- MAGIC from pyspark.sql.functions import *
-- MAGIC
-- MAGIC from langchain_text_splitters import RecursiveCharacterTextSplitter
-- MAGIC
-- MAGIC # Define properties for text splitter
-- MAGIC textSplitter = RecursiveCharacterTextSplitter(
-- MAGIC                                                 chunk_size      = 1000,
-- MAGIC                                                 chunk_overlap   = 200,
-- MAGIC                                                 
-- MAGIC                                                 separators      = ["\n\n", "\n", " ", ""],
-- MAGIC                                                 length_function = len
-- MAGIC                                              )
-- MAGIC
-- MAGIC # Define text splitting function
-- MAGIC def textSplitRecursive(text):
-- MAGIC     if text is None:
-- MAGIC         return []
-- MAGIC     return textSplitter.split_text(text)
-- MAGIC
-- MAGIC # Register for use in SQL
-- MAGIC spark.udf.register("textSplitRecursiveSql", textSplitRecursive, ArrayType(StringType()))
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### E. Store chunks in DocumentChunks table

-- COMMAND ----------

-- DBTITLE 1,Append chunks to DocumentChunks table
INSERT INTO insurance.rag.DocumentChunks (Text, Source, PolicyName, UpdatedOn)

SELECT  explode( textSplitRecursiveSql(content) ) AS Text     -- Split content using recursive text splitter and put them on separate rows

      , path AS Source
      , filename AS PolicyName
      , CURRENT_TIMESTAMP AS UpdatedOn
    
FROM
(
    -- Combine all elements of a document into a single string
    SELECT array_join(
                          transform(
                                        parsed_document:document.elements
                                            ::ARRAY<STRUCT<content:STRING>>, x -> x.content
                                   ),                
                          '\n'
                     ) AS content        
         , path
         , filename

    FROM vwParseDocuments
)

-- COMMAND ----------

-- DBTITLE 1,Verify chunks in table
SELECT *
FROM insurance.rag.DocumentChunks

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### F. Store processed files in ProcessedFiles table

-- COMMAND ----------

-- DBTITLE 1,Update ProcessedFiles table
-- Insert list of processed files in the table
INSERT INTO insurance.rag.ProcessedFiles (FilePath, UpdatedOn)
SELECT path, CURRENT_TIMESTAMP FROM vwFilesToProcess;


-- Check files that are already processed
SELECT * FROM insurance.rag.ProcessedFiles

-- COMMAND ----------

