# Merged Section-wise Interview Question Bank

Source: 31 interview transcripts / 1,781 interviewer question-prompt turns. Similar questions were merged by concept; distinct scenario/follow-up questions were retained.

**Merged core questions: 320**

## 1. Profile, Role, Project & Architecture Experience

1. Tell me about yourself and summarize your career journey.
2. What is your current role and what are your day-to-day responsibilities?
3. What is your role in the current team: individual contributor, lead, manager, or architect?
4. Describe your current/recent project and its end-to-end architecture.
5. Which part of the architecture did you personally design or own?
6. Describe the most complex project you have worked on and the challenges you faced.
7. Have you built any project or component from scratch? Walk me through it.
8. If you rebuilt one of your existing solutions from scratch, what would you change and why?
9. What is the largest data volume you have worked with?
10. Which cloud platforms and data platforms have you worked on?
11. How much experience do you have in Data Engineering, ML, GenAI, and Agentic AI?
12. How do you convert a vague business requirement into a technical solution?
13. How do you select technologies for a solution instead of simply using familiar tools?
14. How do you estimate cost, resources, and delivery timelines for a new solution?
15. How do you handle client-facing architecture discussions and defend your design decisions?

## 2. Enterprise Data Architecture & Data Platform Design

1. Design an end-to-end data platform that receives data from multiple heterogeneous sources and serves downstream consumers.
2. How would you design a scalable platform when source systems and requirements may change in the future?
3. What layers would you include before Bronze, Silver, and Gold in a medallion architecture?
4. What is the purpose of Bronze, Silver, and Gold layers?
5. How would you ingest structured, semi-structured, and unstructured data in the same platform?
6. Which tools/patterns would you use for batch, near-real-time, and streaming ingestion?
7. How would you design a metadata-driven ingestion/transformation framework?
8. What is metadata-driven architecture and how have you implemented it?
9. How would you design a reusable configuration-driven data-quality framework?
10. How would you expose curated data to APIs, external customers, BI, and data-science workloads?
11. How would you design data retention, archival, and storage-tiering policies?
12. How do you design a platform for changing schemas and evolving source systems?
13. How would you choose between 3NF, Data Vault, dimensional/Kimball modelling, and other enterprise modelling approaches?
14. What is Data Mesh and when would you use it?
15. How would you design a platform that supports both batch and streaming (Lambda architecture)?

## 3. ETL/ELT, Ingestion, CDC & Data Quality

1. What is the difference between full load, incremental load, and CDC?
2. How does CDC work internally and what source-system prerequisites are required?
3. How would you design a CDC pipeline for inserts, updates, and deletes?
4. In Bronze, would you append or overwrite CDC data, and why?
5. How would you merge CDC data into Silver/Gold?
6. What is the difference between SCD Type 0, Type 1, Type 2, and Type 3?
7. How do you handle schema drift and schema evolution?
8. What happens when an incoming file does not match the expected schema?
9. How do you make a pipeline self-healing when schema changes occur?
10. How do you ensure data completeness and correctness during ingestion?
11. What data-quality dimensions/checks do you normally implement?
12. How do you validate primary-key uniqueness, nulls, duplicates, data types, and business rules?
13. How do you reconcile source and target data after migration or ingestion?
14. How do you make pipelines idempotent so retries do not create duplicates?
15. How do you recover when only part of a batch is successfully loaded?
16. How would you migrate hundreds or thousands of tables using a reusable metadata-driven approach?
17. How would you validate a large migration before cutover?
18. How would you plan parallel run, reconciliation, and final cutover?

## 4. Databricks Architecture & Platform

1. Explain Databricks architecture and the responsibility of each major component.
2. What is the difference between classic clusters, serverless SQL warehouses, and Photon-enabled compute?
3. How do you choose the appropriate Databricks compute for different personas/workloads?
4. What is Photon and what advantage does it provide?
5. What are the major differences between older Spark/Databricks runtimes and newer runtimes?
6. What are Databricks cluster policies and why are they needed?
7. What controls would you put into a cluster policy?
8. How do you optimize Databricks costs while meeting SLAs?
9. What are Lakeflow Connect/Lakeflow pipelines and where would you use them?
10. What are Spark Declarative Pipelines / DLT?
11. How do DLT expectations work for data-quality enforcement?
12. What is Databricks Genie / Genie Space / Genie Code and where have you used them?
13. What newer Databricks features have you explored recently?
14. How do Databricks workspaces, catalogs, and environments relate to each other?
15. How do you promote engineering and AI workloads from dev to test to production in Databricks?

## 5. Spark Performance, Troubleshooting & Capacity Planning

1. A Spark job that normally runs quickly is suddenly 5x slower with no code change. How do you troubleshoot it?
2. A Databricks job takes 8+ hours and misses its SLA. Walk me through your troubleshooting steps.
3. Why can a job processing 100 GB take longer than another job processing multiple TB?
4. What Spark UI metrics/stages would you inspect first?
5. What causes data skew and how do you fix it?
6. When would you use broadcast joins?
7. What is AQE and what optimizations does it perform?
8. How do you choose the number of partitions?
9. What is the difference between repartition and coalesce?
10. What causes shuffle and how do you reduce it?
11. How do you handle out-of-memory errors in Spark?
12. What is the difference between cache and persist?
13. What is the difference between MEMORY_ONLY and MEMORY_AND_DISK persistence?
14. How do you solve the small-files problem?
15. What is bucketing and when would you use it?
16. How do you size a Spark cluster for a known data volume and SLA?
17. How do you prioritize an SLA-critical Spark job?
18. What proactive dashboards/alerts would you build so users do not discover performance problems first?
19. Can a Spark job succeed technically but produce wrong data? What would you investigate?

## 6. Delta Lake, Lakehouse & Table Optimization

1. What is Delta Lake and what is a Delta table?
2. Can Delta tables be stored on AWS S3 and how?
3. What file format is used underneath Delta Lake?
4. What advantages does Delta provide over plain Parquet?
5. How does a DELETE work internally on a Delta table?
6. What are deletion vectors and why are they useful?
7. What is checkpointing in Delta and why is it needed?
8. What is VACUUM and how does it affect time travel?
9. What is the default/defined retention policy and how do you manage it?
10. What is OPTIMIZE and why is it needed?
11. What is Auto Optimize and how is it different from OPTIMIZE?
12. What is Z-ORDER and when would you use it?
13. What is liquid clustering and how is it different from partitioning/Z-ORDER?
14. What is the difference between clustering and partitioning?
15. How would you roll back or recover a Delta table?
16. What is Iceberg and how does it compare conceptually with Delta Lake?
17. Why would you choose Parquet over CSV?

## 7. Unity Catalog, Governance, PII & Security

1. What is Unity Catalog and what are its main governance features?
2. How would you implement row-level security and column-level security?
3. How do row filters and column masks work?
4. How would you protect PII across Bronze, Silver, and Gold?
5. What is the difference between masking and tokenization?
6. When would you tokenize rather than mask sensitive data?
7. How would you implement RBAC for an enterprise data/AI platform?
8. How do you implement auditing and lineage so you can explain how a metric was produced months ago?
9. How would you answer an auditor asking which source fields and logic produced a historical risk score?
10. What is Delta Sharing and when would you use it?
11. How do you securely share data with external customers?
12. How do you classify/tag sensitive data as part of governance?
13. What controls should exist before an AI agent can execute SQL against production data?
14. What access level should an agent have to production databases?
15. What would you do if PII was accidentally embedded and indexed in a vector database?
16. What design changes prevent PII from reaching an LLM or vector index again?
17. How would you handle GDPR/security requirements in a GenAI application?

## 8. Streaming, Kafka, Event Processing & Checkpointing

1. Design a near-real-time ingestion architecture for continuously arriving events/logs.
2. When would you choose Kafka/Event Hubs/Kinesis versus simpler event-triggered services?
3. How do Kafka partitions and consumer groups work?
4. How do you guarantee ordering for events belonging to the same customer/key?
5. How do you avoid message loss and duplicate processing?
6. What is exactly-once or effectively-once processing and how do you design for it?
7. What is a streaming checkpoint and what does it store?
8. What happens if a Structured Streaming checkpoint becomes corrupted?
9. How would you recover from a corrupted checkpoint with very high data volume?
10. How do you handle late-arriving events?
11. What are watermarks and when are they needed?
12. How do you safely replay the last seven days of events after finding a transformation defect?
13. How do you ensure replay does not corrupt downstream systems?
14. How would you build event-driven processing on AWS/Azure?

## 9. AWS, Azure & Cloud Architecture

1. How familiar are you with AWS/Azure services and which ones have you used directly?
2. Design a low-cost near-real-time ingestion solution on AWS.
3. When would you use Lambda, Step Functions, EventBridge, SNS, SQS, or Kinesis?
4. Why would you split a workflow into multiple Lambda functions instead of one large function?
5. How do you decide whether Step Functions are actually needed?
6. How do you manage continuously arriving S3 files without storage cost growing indefinitely?
7. How would you choose storage classes/retention tiers for historical data?
8. What datastore would you choose when one year of historical data must stay immediately available?
9. How do you design event-triggered processing from file or time-based events?
10. How do you use Azure Data Factory in your architecture?
11. Would ADF be the orchestration layer or Databricks, and why?
12. How do you use Azure Event Hubs for streaming?
13. How do you configure email/notification services in Azure?
14. How do you reduce cloud spend before scaling a data/AI platform?

## 10. Snowflake & Data Warehouse

1. How comfortable are you with Snowflake and where have you used it?
2. How would you load data from S3 into Snowflake?
3. What are Snowflake stages and what types exist?
4. When would you use Snowpipe versus bulk COPY?
5. What limitations/trade-offs does Snowpipe have for TB-scale loads?
6. How would you apply weekly change-log files to an existing Snowflake table instead of full refresh?
7. Can you repartition/cluster an existing table and how would you approach it?
8. How does Snowflake data sharing work with external consumers?
9. Why would you move from Redshift to Snowflake or vice versa?
10. How do Snowflake and Databricks integrate in a Lakehouse/warehouse architecture?

## 11. SQL & Data Modelling

1. Write/explain a query to find the top N salaries from an employee table.
2. How would you remove duplicates from a table when there is no primary key?
3. How do you identify duplicate records when all columns may be the same?
4. How would you recursively traverse an employee-manager hierarchy?
5. How would you calculate sales for a manager including all direct and indirect reports?
6. What SQL technique would you use for hierarchical data: recursive CTE or another approach?
7. How do you perform incremental comparison between yesterday's and today's data?
8. What is a fact table and what is a dimension table?
9. What is the difference between 3NF and dimensional/Kimball modelling?
10. How would you choose the data model for the Silver enterprise layer?
11. How comfortable are you writing complex SQL and how would you rate yourself?

## 12. Python & General Coding

1. How do you convert a Python list into a Pandas DataFrame?
2. How do you build a rolling one-minute API-call count from timestamps?
3. How do you define a Python class and constructor using __init__?
4. How do you initialize object attributes in Python?
5. Is a Python dictionary mutable or immutable?
6. What is the difference between a list, tuple, set, and dictionary?
7. What is the difference between multithreading and multiprocessing?
8. How do you implement threading in Python?
9. How do you optimize Python code performance?
10. How do you implement dependency injection in Python?
11. How do you send email/notifications from Python or PySpark jobs?
12. How would you structure reusable ingestion/transformation/data-quality code using classes and functions?

## 13. Traditional Machine Learning

1. Explain bias versus variance and the bias-variance trade-off.
2. What is overfitting and underfitting, and how do you detect/fix them?
3. What is data leakage during training and how do you prevent it?
4. When would you choose Random Forest versus Gradient Boosting?
5. How does Random Forest work internally?
6. If n_estimators=5 in Random Forest, how are rows/features used across trees?
7. How does Gradient Boosting work?
8. What is the difference between XGBoost, LightGBM, CatBoost, and Random Forest?
9. When would you choose deep learning instead of traditional ML?
10. What is Logistic Regression and why is it a classification algorithm?
11. What loss/objective does Logistic Regression use?
12. How can Logistic Regression be extended to multiclass classification?
13. What would you use when the classes are not linearly separable?
14. How do you handle an imbalanced classification dataset?
15. When do you use accuracy, precision, recall, F1, and ROC-AUC?
16. What is model drift and what causes it?
17. How do you detect and respond to model drift?
18. What metrics and validation strategy do you use before putting a model into production?

## 14. Deep Learning, Transformers & Fine-Tuning

1. What is the difference between RNN/LSTM and Transformers?
2. What is self-attention and multi-head attention?
3. What is the difference between CNNs and Vision Transformers?
4. Does backpropagation happen in CNNs/Transformers and how?
5. What is an embedding and how does an embedding model work?
6. What determines embedding dimension?
7. What is fine-tuning and when would you use it?
8. What fine-tuning strategy have you used?
9. What are LoRA/adapters and how do you choose adapter/rank dimensions?
10. What is the difference between GPT/OpenAI models and open-source models such as Llama?
11. What changes operationally when you use a hosted GPT model versus self-hosted Llama?
12. How would you containerize and deploy a foundation model?

## 15. GenAI / LLM Application Design

1. Describe an enterprise LLM application you built end to end.
2. Which foundation/LLM models have you used and why did you choose them?
3. What prompting techniques have you used?
4. How do zero-shot, few-shot, and other prompting approaches differ?
5. How do you manage token usage and context-window limits?
6. How do you reduce input tokens before sending a large document to the model?
7. How do you select the right prompt version for a specific user request?
8. How do you manage prompt templates and prompt versions?
9. How do you reduce hallucinations in an LLM application?
10. How do you evaluate whether an LLM answer is trustworthy before production?
11. How do you monitor token consumption, latency, failures, and answer quality?
12. How do you handle timeouts/network failures when calling an LLM API?
13. How do you make LLM workflows resumable rather than restarting from the beginning?
14. How do you protect an LLM application against prompt injection?
15. How would you detect/mask sensitive information without relying only on regex?
16. How do you decide whether an LLM is actually necessary for a business problem?

## 16. RAG Architecture, Chunking, Embeddings & Retrieval

1. Explain the complete RAG architecture from document ingestion to final answer.
2. Why did you choose RAG instead of fine-tuning?
3. What chunking strategies have you used and why?
4. What are chunk size and chunk overlap?
5. What is the problem with chunks that are too large?
6. What is the problem with chunks that are too small?
7. How do you choose chunk size and overlap for a use case?
8. Which embedding model did you use and why?
9. How do you choose an embedding model based on quality, dimension, latency, and deployment constraints?
10. How does cosine similarity work?
11. Which vector databases/vector stores have you used?
12. How is a vector database different from a traditional database?
13. What is ANN search and why is an index required?
14. What are HNSW/IVF/PQ-style indexing approaches conceptually?
15. How would you improve poor retrieval quality?
16. What is hybrid search and when would you combine dense retrieval with lexical/BM25 retrieval?
17. What is reranking and when would you add a reranker?
18. What is MMR and how does it balance relevance and diversity?
19. What is MRR and how is it different from MMR?
20. What are Precision@K and Recall@K?
21. If Recall@K is high but Precision@K is low, what is likely wrong and what would you tune first?
22. How do you evaluate retrieval separately from final-answer generation?
23. How do you handle information spread across multiple documents when normal vector RAG performs poorly?
24. How would you scale a RAG pipeline from hundreds to millions/billions of requests while controlling latency?
25. What metrics would you monitor when RAG latency and hallucination suddenly increase?
Production challenges , fallback
Retry mechanism
text2sql
Multimodal rag
Psedieo
From document
## 17. Agentic AI, Tool Calling, State & Memory

1. Explain tool-call implementation in an agentic system.
2. How does the LLM decide whether to answer directly or call a tool?
3. What should you consider when defining a custom tool/function?
4. How do you know whether a tool call actually occurred?
5. How would you observe tool calls without third-party observability products?
6. What is the difference between token streaming and agent/tool tracing?
7. How do you implement a multi-agent system?
8. What are the main components of an agentic system?
9. How do multiple agents communicate with each other?
10. What is a supervisor agent and when would you use one?
11. How do you orchestrate nodes/agents in LangGraph?
12. What is synchronous versus asynchronous agent execution?
13. How do you maintain agent state across steps?
14. What is the difference between state and context?
15. How do you maintain state for multiple concurrent users/sessions?
16. What is short-term/working memory versus long-term memory?
17. What is semantic memory?
18. What is episodic memory?
19. What is buffer memory and where does it fit?
20. How do you persist and retrieve long-term agent memory?
21. How do you prevent an agent from repeatedly calling the same tool forever?
22. What limits/safeguards do you put around autonomous agents?
23. Which actions can an agent perform autonomously and which should require human approval?
24. How do you log why an agent stopped or failed?
25. How do you make an agent restart a failed pipeline safely without reprocessing old data?
26. How would you automate requirement-document analysis and use-case/test-case extraction using agents?
27. How would you build reusable agents for data migration, governance, ingestion, or code conversion?
28. How do you manage context engineering and the knowledge layer in an agentic solution?

## 18. MLOps, CI/CD, Deployment & Observability

1. Explain the CI/CD architecture you have implemented for data/ML projects.
2. What happens from code commit through build, test, deployment, and production?
3. What artifact does your build process produce?
4. How do you promote artifacts from dev to test to production?
5. How do GitHub Actions/Jenkins pipelines fit into your deployment process?
6. How do you use Terraform/IaC for environment provisioning?
7. How do you deploy models using Docker/Kubernetes/AKS?
8. How do you deploy and serve models in Databricks?
9. How do you version models and decide which model is production/staging/archived?
10. How does MLflow experiment tracking relate to model registry/versioning?
11. What information do you store in an MLflow experiment?
12. How do you monitor models/agents in production?
13. Which metrics/KPIs do you monitor for LLM applications?
14. How do you test an AI application before production?
15. How do you implement observability for LangGraph/agent workflows?
16. How do you monitor pipelines, retries, errors, latency, and data quality?

## 19. Architecture Scenarios, Reliability & Cost

1. How would you design a self-healing pipeline when upstream schemas break?
2. How would you handle a pipeline that restarts and accidentally reprocesses old files?
3. How would you make streaming replay safe and idempotent?
4. How would you make a system resilient to partial failures and retries?
5. How would you scale a system when traffic suddenly increases by orders of magnitude?
6. How would you reduce latency without sacrificing answer quality?
7. How do you decide between a simple solution and a complex managed service?
8. How would you reduce cloud cost while preserving reliability and SLA?
9. What architecture decision did you make that later turned out to be wrong, and what did you learn?
10. How do you decide whether to refactor an existing brownfield system or leave it as-is?
11. How do you validate that a target/staging system stays synchronized with its source?
12. How do you detect and recover from partially loaded data?
13. How would you support historical reproducibility and auditing months after a result was produced?

## 20. Leadership, Estimation, Agile & Behavioral

1. How do you estimate a project when the work contains research/POCs and uncertainty?
2. How do you create a WBS and derive the project end date?
3. How do you decide sprint scope for research-oriented work?
4. How do you allocate tasks across team members?
5. How do you handle tight deadlines?
6. How do you communicate risks and trade-offs to clients/stakeholders?
7. How do you manage requirements that change during development?
8. How do you decide what can and cannot be delivered in a sprint?
9. How do you plan a multi-month migration/delivery program?
10. How do you estimate team size and required skills?
11. How do you lead engineers while remaining technically hands-on?
12. How do you handle ownership of a feature/product from requirement through production support?
13. Why are you looking for a change?
14. What is your preferred location / notice period / joining availability?
15. What questions do you have for us?

LLM fine-tuning	Conceptual	High
LoRA / QLoRA / PEFT	Some theory	High
PyTorch	Limited hands-on discussed	High
TensorFlow / Keras	Little/no hands-on discussed	High
GAN	Not prepared deeply	High
VAE	Not prepared deeply	High
Multimodal models	Not prepared	High
Computer Vision	Limited	High
Synthetic data generation	Limited	High
Data augmentation	Limited	High
Training foundation models	Very limited	High
Docker	Conceptual/basic exposure	Medium
Kubernetes / AKS	Conceptual	Medium–High
Azure Cognitive Services	Limited	High
Python backend/FastAPI	Some exposure	Medium
.NET backend	Not discussed	High, but optional if Python accepted
Production model serving	Conceptual–Good	Medium
Cloud AI deployment	Partial	


5 wrkflow pattern
prompyt chaining
routing
paralleization with :aggregation
orchestration worker: synthesizer
evaluator optimizer


capgemini data engineering 
Your consumer job has already read data from a Kafka topic and committed the offset. Now you need to reread that data. How will you do it?
If the data has already been consumed and the offset committed, is there another way to read it again from the Kafka topic?
If corrupted or incorrect data is discovered later in Bronze/Raw, how will you reprocess the data from the source?
If data has been archived after processing, how will your pipeline automatically pick it from the archive and reprocess it?
Can the reprocessing mechanism be automated so that no manual upload or manual intervention is required?
Apart from validation, how would you design automated reprocessing in an ETL pipeline?
What scenarios can require reprocessing—for example deletion, duplicate records, source corruption, etc.?
Can reprocessing be parameter-driven instead of requiring manual effort?
Can you change/update the watermark to an earlier value and rerun the job to re-extract data?
How would you update a watermark column when your data is stored on S3, ADLS, GCS, or another data lake?
How should an ETL framework support historical load, incremental load, retry, and reprocessing?
If you need to reprocess data, can you restart from any layer, or do you need to start again from ingestion?
You mentioned CDC. How are you handling CDC?
Once data lands in ADLS and you read it through Spark/Databricks, how exactly are you reading it—file-based or table-based?
For every source, are you following the same approach of processing the file and then moving it to an archive?
If historical files remain in the data lake, how will your Spark job read only the newly arrived files instead of rereading all historical files?
If your job has been stopped for the last two days, how will your “current timestamp/current date” logic still identify the correct files to process?
If there are only files in the landing/raw area, where are you storing the watermark state? Do you need a table for it?
How does the Silver layer know which files it needs to read from the landing/raw layer?
Have you created a table on top of the Raw/Bronze layer?
Where is the timestamp column coming from? Are you creating it while loading data into Bronze?
When a new file arrives in ADLS, how is the metadata/catalog information updated? Does anything need to run for that?
If your source file had 10 columns earlier and now has 12 columns, how will your Unity Catalog/table metadata be updated?
Have you implemented schema evolution / mergeSchema logic in your jobs? Do you enable it for every file/job?
When you submit a Spark job to a Spark cluster, explain the Spark architecture and what happens in the background.
When is a Spark job created, when are stages created, and when are tasks created?
Who assigns work to worker nodes/executors? Does the driver allocate resources?
If the driver allocates resources, then what is the role of the Resource Manager / Cluster Manager?
Suppose Spark reads a CSV into df1, creates df2 using a filter, and creates df3 using groupBy. How many stages will be created?
Are you comfortable with SQL?
Have you worked on AWS cloud? Which AWS services have you used?
What is the cold start problem in AWS Lambda?
Given an Employees table with IDs 1, 2, 4, and 7, how would you find the missing IDs 3, 5, and 6? Assume there can be millions of records.
How would you implement the missing-ID solution using recursion / a recursive query?
Given a Logins table with user_id and login_date, find users who logged in for 3 consecutive days.
Are you familiar with Slowly Changing Dimensions (SCD)? Explain the different SCD types.
Explain SCD Type 0, Type 1, Type 2, and Type 3.
How would you implement SCD Type 2 without using a MERGE statement?
If the analytics layer must keep both historical and current versions of a record, how will you maintain them in the same table?
If one business ID receives multiple records/versions, how do you determine which one is the latest/current record?
Have you actually designed or implemented SCD Type 2 in any project?
Do you have any questions for me / about the role



A. Questions actually asked in your previous interviews
Your Kafka consumer has already consumed a record and committed the offset. How do you read that record again?
Does committing an offset delete the Kafka message?
If data is still retained in Kafka, how can you replay it?
Can you reset a consumer group's offset?
Can you reread Kafka data using another consumer group?
If downstream Bronze data is corrupted/deleted, how do you reprocess it from Kafka?
What is the difference between retrying a failed message and replaying already successfully processed messages?
When should a message go to a DLQ?
Is DLQ the right mechanism for replaying already successfully consumed data?
What exactly are you handling on the Kafka consumer side?
Can replay logic be built into the consumer rather than requiring an administrator?
How would you automate Kafka reprocessing?
What streaming frameworks have you worked with?
How would you process Kafka directly versus through Spark Structured Streaming?
When would you use direct Kafka consumers versus a streaming framework?
How do consumer groups work?
How do partitions affect consumer parallelism?
How do you guarantee ordering for events for the same customer?
How do you prevent message loss?
How do you prevent duplicate processing?
What is exactly-once/effectively-once processing?
What does a streaming checkpoint store?
What happens if a Structured Streaming checkpoint is corrupted?
How do you recover from a corrupted checkpoint with huge amounts of Kafka data?
How do you handle late-arriving events?
What is a watermark?
How would you replay the last seven days after discovering a transformation bug?
How do you ensure replay doesn't corrupt downstream systems?
How do you handle schema evolution in a streaming pipeline?
How would you design Kafka → Databricks → Bronze → Silver processing?

Those replay and committed-offset questions were particularly central in the Capgemini discussion.

Your earlier preparation material also explicitly covered per-key ordering, duplicates/loss, effectively-once processing, checkpoint recovery, late data, seven-day replay and safe downstream replay.

B. Kafka fundamentals
What is Apache Kafka?
Why is Kafka called a distributed event-streaming platform?
Kafka vs traditional message queue: what's the difference?
Kafka vs RabbitMQ?
Kafka vs SQS?
Kafka vs Kinesis?
Kafka vs Azure Event Hubs?
When would you choose Kafka?
When would you not choose Kafka?
What is a Kafka cluster?
What is a broker?
What is a topic?
What is a partition?
What is an offset?
Are offsets unique across a topic or only within a partition?
What is a Kafka record/message?
What fields does a Kafka record contain?
What is a message key?
What is a consumer?
What is a producer?
What is a consumer group?
How does Kafka provide scalability?
How does Kafka provide fault tolerance?
Why can multiple applications read the same Kafka records?
Why doesn't consuming a Kafka message remove it?
C. Kafka storage and retention
Where does Kafka actually store records?
What is a Kafka log?
What is a log segment?
How are records physically organized inside partitions?
What is retention?
What is retention.ms?
What is retention.bytes?
Does Kafka delete a record when a consumer reads it?
Does Kafka delete a record when its offset is committed?
What happens when retention expires?
Can you replay a message after its retention period has expired?
How do you decide the appropriate Kafka retention period?
What is log compaction?
Retention vs log compaction?
What is a tombstone record?
How does compaction work with keyed events?
Can a compacted topic preserve every historical event?
When would you use a compacted topic?
When would you use delete-based retention?
Can retention and compaction be used together?
D. Partitions
Why does Kafka have partitions?
How does Kafka decide which partition receives a record?
What happens when the producer sends a key?
What happens if the producer doesn't provide a key?
Why does partition count control Kafka parallelism?
Can multiple consumers in the same consumer group read one partition simultaneously?
What happens when you have 10 partitions and 5 consumers?
What happens with 5 partitions and 10 consumers?
What happens to the extra five consumers?
Can you increase partition count?
Can you decrease partition count?
What happens to key distribution after increasing partitions?
Can increasing partitions affect ordering?
What is a hot partition?
How do you identify partition skew?
How would you fix a hot partition?
What makes a good Kafka partition key?
Customer ID or timestamp—which would generally be a better partition key and why?
What happens if one customer produces 50% of all events?
How do you choose number of partitions?
How does partition count affect throughput?
How does partition count affect consumer scaling?
What are the disadvantages of having too many partitions?
E. Ordering
Does Kafka guarantee ordering?
Is ordering guaranteed across an entire topic?
Where exactly does Kafka guarantee ordering?
How do you guarantee ordering for one customer?
Why should related records have the same key?
What can break ordering?
Can retries result in ordering problems?
What happens to ordering if partition count changes?
Can you guarantee global ordering?
Why is global ordering expensive?
How do you handle out-of-order CDC records?
If update event 3 arrives before update event 2, what would you do?
How can sequence number/version/LSN help?
What if strict ordering is required but throughput is very high?
F. Producer internals
How does a Kafka producer work internally?
What happens from producer.send() until the record reaches Kafka?
What is producer batching?
What does batch.size do?
What does linger.ms do?
What does buffer.memory do?
What is compression in Kafka?
gzip vs snappy vs lz4 vs zstd?
Why does batching improve throughput?
Throughput vs latency trade-off in producers?
What are Kafka acknowledgements?
Explain acks=0.
Explain acks=1.
Explain acks=all.
Which acknowledgement setting would you use for critical financial data?
What happens when the producer times out?
What is producer retry?
Can producer retries create duplicates?
What is an idempotent producer?
How does idempotent production prevent duplicates?
What are producer IDs and sequence numbers conceptually?
What is max.in.flight.requests.per.connection?
How can it affect ordering?
What is a Kafka producer transaction?
When would you use Kafka transactions?
G. Brokers, replication and availability
How does Kafka replicate partitions?
What is a partition leader?
What is a replica?
What is a follower replica?
What is an ISR?
What does ISR stand for?
What happens if the partition leader fails?
How is a new leader selected?
What is replication factor?
If replication factor is 3, what does it mean?
Can replication factor be greater than broker count?
What is min.insync.replicas?
How does acks=all interact with min.insync.replicas?
What happens if ISR drops below minimum?
Would you prefer rejecting writes or accepting potentially unsafe writes?
What happens when a broker goes down?
What happens when multiple brokers fail?
What is an under-replicated partition?
What is an offline partition?
What causes replica lag?
What is unclean leader election?
What risk does unclean leader election create?
How do you design Kafka for high availability?
H. Kafka metadata / cluster management
What is KRaft?
Why did Kafka historically use ZooKeeper?
What responsibilities does Kafka's metadata/controller plane have?
What is the Kafka controller?
What happens when a controller fails?
How are brokers discovered?
What information does the cluster metadata contain?
What happens when brokers join or leave a cluster?

For modern Kafka study, concentrate on KRaft, while still knowing ZooKeeper because older enterprise environments/interviewers may mention it.

I. Consumer fundamentals
How does a Kafka consumer work?
What does poll() do?
What is consumer position?
What is committed offset?
What's the difference between current position and committed offset?
Where are consumer-group offsets stored?
What is __consumer_offsets?
What happens if the consumer crashes before committing?
What happens if it commits before processing?
What happens if processing succeeds but commit fails?
Why can consumers receive duplicates?
How do you make the consumer idempotent?
Auto commit vs manual commit?
When would you disable auto commit?
Synchronous vs asynchronous commit?
When should the consumer commit an offset?
Commit before processing vs after processing?
How do you handle slow processing?
What is max.poll.records?
What is max.poll.interval.ms?
What is session.timeout.ms?
What is heartbeat interval?
What happens if processing takes longer than max.poll.interval.ms?
J. Consumer groups
Why do consumer groups exist?
How are partitions assigned to group members?
Can two consumer groups read the same topic independently?
Can two consumers inside the same group process the same partition at the same time?
What happens when a consumer joins the group?
What happens when a consumer leaves?
What is consumer-group rebalance?
Why are rebalances expensive?
What events trigger a rebalance?
What happens to processing during rebalance?
How do you reduce unnecessary rebalances?
What is static group membership?
What is cooperative rebalancing?
Eager vs cooperative rebalance?
What is a partition assignment strategy?
Range assignor?
Round-robin assignor?
Sticky/cooperative sticky assignment?
subscribe() vs manually assign() partitions?
When would manual partition assignment make sense?
K. OFFSET QUESTIONS — your most important area
What exactly is a Kafka offset?
Who creates offsets?
Does Kafka maintain an offset per message?
Does every consumer group have its own committed offsets?
If group A commits offset 100, does group B start at 100?
What happens when no committed offset exists?
What is auto.offset.reset=earliest?
What is latest?
What is none?
earliest vs latest?
Does auto.offset.reset=earliest automatically replay data when a valid committed offset already exists?
How do you reread data after committing the offset?
How do you reset offsets?
How do you reset to earliest?
How do you reset to a specific offset?
How do you reset by timestamp?
How do you replay only one partition?
How do you replay only a specific time period?
How do you replay between start and end offsets?
What is seek()?
seek() vs consumer-group reset?
New consumer group vs resetting the existing group?
Which is safer for a production replay?
How do you determine the offset corresponding to a timestamp?
What happens if the desired offset has already been deleted by retention?
How would you keep production consumption running while performing historical replay?

If there is one chapter you need to master deeply, it is this one.

L. Replay / reprocessing
What is replay?
Retry vs replay?
Backfill vs replay?
Recovery vs replay?
Why is Kafka suitable for replay?
How would you reprocess yesterday's events?
How would you replay seven days?
How would you replay one specific customer?
How would you replay only failed partitions?
Would you reset the production consumer group's offsets?
Why might resetting production offsets be dangerous?
Would you create a dedicated replay consumer group?
How do you track a replay run?
Would you add a replay_run_id?
How do you avoid writing duplicates during replay?
How do you prevent emails/APIs/payments from firing again during replay?
How do you reconcile replay output?
What if Bronze has been deleted but Kafka still retains the events?
What if both Kafka retention and Bronze data are gone?
What should your RPO/recovery strategy be?
Should Kafka be your permanent historical archive?
Why keep immutable Bronze if Kafka has retention?
What happens when the code defect affected only Silver?
Would you replay Kafka or rebuild from Bronze?
How do you choose the lowest safe layer to restart from?

Your earlier architecture material emphasizes keeping an immutable/replayable Bronze layer and using controlled replay rather than blindly moving production offsets backwards.

M. Delivery semantics
Explain at-most-once.
Explain at-least-once.
Explain exactly-once.
Which one does a normal Kafka consumer typically behave like?
What causes at-least-once duplicates?
How do you achieve effectively-once processing?
What does exactly-once mean across Kafka only?
What does exactly-once mean end-to-end?
Does Kafka exactly-once automatically guarantee exactly-once writes into an external database?
How would you guarantee effectively-once into Delta?
How would you guarantee effectively-once into PostgreSQL?
What is an idempotent sink?
What is an idempotency key?
Can MERGE help?
How would event ID + business key help?
How do Kafka transactions help read-process-write Kafka workflows?
What are transactional producers?
What is read_committed isolation?
N. Duplicate processing
Why can Kafka records be processed more than once?
Consumer crashes after DB write but before offset commit—what happens?
Offset committed before DB write and the application crashes—what happens?
Which failure causes duplicates?
Which failure causes data loss?
How do you deduplicate messages?
Event ID vs offset for deduplication?
Can topic + partition + offset be used as a unique source identifier?
What if the producer itself sends the same business event twice?
How would you distinguish technical duplicates from legitimate repeat business events?
Where should deduplication occur: Bronze, Silver or consumer?
How long should you keep deduplication state?
O. Retry and DLQ
What kinds of errors should be retried?
What errors should not be retried?
What is a transient error?
What is a permanent/data error?
What is a poison pill message?
How do you handle a poison message?
What is a DLQ?
When should a message move to a DLQ?
Should every failed Kafka message immediately go to DLQ?
How many retries should you perform?
What is exponential backoff?
Why should you add jitter?
What is a retry topic?
DLQ vs retry topic?
Should DLQ preserve original Kafka metadata?
What fields should a DLQ event contain?
Should you include original topic?
Partition?
Offset?
Exception?
Retry count?
Event ID?
Failure timestamp?
How do you replay DLQ records after correcting the issue?
Should replaying DLQ require Kafka offset rewinding?
How do you prevent an infinite DLQ loop?

A good architecture separates DLQ replay from rewinding the original Kafka consumer, which was also part of your earlier preparation.

P. Backpressure and consumer lag
What is consumer lag?
How do you calculate lag?
What does increasing lag indicate?
What are common causes of lag?
Slow consumer vs insufficient partitions?
What if producer rate is 100k events/sec but consumers handle 50k/sec?
How do you reduce lag?
Add consumers?
Add partitions?
Optimize processing?
Batch writes?
Increase consumer fetch size?
Scale Spark executors?
What if adding consumers doesn't improve performance?
What happens if number of consumers already equals number of partitions?
What is backpressure?
How do you protect downstream systems from Kafka traffic spikes?
How do you rate-limit consumers?
When would you pause/resume Kafka partitions?
What metrics would tell you the pipeline cannot keep up?
Q. Late and out-of-order data
Processing time vs event time?
What is late-arriving data?
Why can Kafka messages arrive late?
Why can events become out of order?
How do you handle late events in Spark Structured Streaming?
What is watermarking?
Is a watermark the same thing as a Kafka offset?
What state does a watermark help clean?
What happens to events arriving later than the watermark?
How would you handle events that are too late but still business-critical?
How would you update previously computed aggregates?
How do windows work?
Tumbling window?
Sliding window?
Session window?
How do you join two Kafka streams?
What happens with late records during stream-stream joins?
Why can stream joins create huge state?

Late events and watermarks have already appeared in your previous interview preparation.

R. Serialization and schema
Why does Kafka store bytes rather than understand your business schema?
What is serialization?
What is deserialization?
JSON vs Avro?
Avro vs Protobuf?
JSON Schema?
Why would you use Avro?
What is Schema Registry?
Does Schema Registry store messages?
How does a producer use Schema Registry?
How does a consumer determine the schema?
What is schema evolution?
Backward compatibility?
Forward compatibility?
Full compatibility?
What happens when a producer adds a column/field?
What if a producer removes a required field?
What if data type changes from integer to string?
How do you stop incompatible schema changes?
What is a data contract?
Who owns schema changes?
What happens when multiple consumers depend on different schema versions?
How would you quarantine incompatible messages?
How do you avoid breaking all consumers when a producer changes schema?
S. Kafka + CDC
What is CDC?
Polling vs CDC?
How does log-based CDC work?
How does Debezium work conceptually?
Source DB → Debezium → Kafka: explain the architecture.
What does an insert CDC event look like?
Update?
Delete?
What are before/after images?
What is LSN/SCN/binlog position?
Why do CDC events need ordering?
How do you handle duplicate CDC events?
How do you handle out-of-order CDC events?
How do you handle deletes?
Hard delete vs soft delete?
How do you perform SCD2 using Kafka CDC?
What happens if Debezium stops for six hours?
How does it resume?
How do snapshots work?
Initial snapshot vs incremental CDC?
How do you avoid duplicate rows between initial snapshot and CDC?
What happens when source schema changes?
How do you preserve CDC operation metadata in Bronze?

Your project/prep architecture already uses the pattern of a streaming bus feeding Bronze and then stateful validation/deduplication before Silver.

T. Kafka Connect
What is Kafka Connect?
Why use Kafka Connect instead of writing a custom producer/consumer?
Source connector vs sink connector?
Standalone vs distributed mode?
What is a connector?
What is a task?
How does Connect scale?
Where are connector offsets stored?
How are connector configurations stored?
How are connector statuses stored?
What is Debezium's relationship with Kafka Connect?
How do you handle connector failure?
How do you monitor connector lag/errors?
What is Single Message Transform (SMT)?
When should you avoid heavy transformation inside Kafka Connect?
U. Kafka + Spark Structured Streaming / Databricks
How do you read Kafka using Spark Structured Streaming?
What are bootstrap.servers?
subscribe vs assign vs subscribePattern?
What is startingOffsets?
earliest vs latest?
Can you specify explicit partition offsets?
What is maxOffsetsPerTrigger?
Why would you use it?
What is failOnDataLoss?
What happens if Kafka deleted offsets required by the stream?
How does Structured Streaming checkpoint Kafka progress?
Is a Spark checkpoint the same as the Kafka consumer group's committed offset?
Why should every streaming query have its own checkpoint?
What happens if two queries share a checkpoint directory?
What happens if you delete the checkpoint?
Can deleting a checkpoint cause the stream to replay data?
How do you safely restart from a new checkpoint?
How do you make the Delta target idempotent?
What is foreachBatch?
Why use foreachBatch for MERGE?
How do you deduplicate Kafka records in Spark?
How do you use topic, partition, and offset columns?
How do you preserve Kafka metadata in Bronze?
How do you limit streaming ingestion rate?
How do you handle a huge backlog?
How do you catch up after downtime?
How do you handle schema parsing errors?
Where do invalid events go?
How do you monitor streaming queries?
How do you recover from state-store/checkpoint corruption?

Current Databricks supports Kafka directly as a Structured Streaming source/sink.

V. Kafka vs Auto Loader — especially for your interviews
Kafka vs Auto Loader: what's the fundamental difference?
When should the source be Kafka?
When should the source be cloud files?
When would you use Auto Loader?
When would you use Structured Streaming with Kafka?
Can the same downstream Bronze/Silver architecture handle both?
How does checkpointing differ?
How does replay differ?
Kafka offset vs processed-file state?
How does file schema drift differ from event schema evolution?
How would you process continuously arriving files?
How would you process continuously arriving Kafka events?
Auto Loader vs COPY INTO?
Kafka vs Event Hubs?
Kafka vs Kinesis?
Streaming vs micro-batch vs normal batch?

Current Auto Loader is the cloudFiles Structured Streaming source for cloud object storage, including S3, ADLS and GCS.

W. Monitoring and observability
What Kafka metrics do you monitor?
Consumer lag?
Producer error rate?
Request latency?
Throughput?
Bytes in/out?
Under-replicated partitions?
Offline partitions?
ISR shrink/expand?
Broker disk utilization?
Network utilization?
CPU?
JVM GC?
Consumer rebalance frequency?
DLQ rate?
Retry rate?
Schema failure count?
Processing latency?
Event-time lag?
End-to-end data freshness?
How do you define Kafka SLA?
What alerts would you configure?
How do you identify a stuck consumer?
How do you distinguish producer problem from consumer problem?
How do you detect partition skew?
X. Kafka performance tuning
How do you improve producer throughput?
How do you reduce producer latency?
How do you tune batching?
How does compression affect CPU vs network?
How do you increase consumer throughput?
How does number of partitions affect performance?
How does consumer count affect performance?
How does message size affect Kafka performance?
What happens with very large messages?
Why are millions of very small messages inefficient?
Should you batch small events?
How do you size Kafka brokers?
Disk throughput vs network throughput?
SSD vs HDD considerations?
How do page cache and sequential I/O benefit Kafka?
How do you estimate required partitions from throughput?
How would you test Kafka capacity before production?
Y. Security
How do you secure Kafka?
Encryption in transit?
TLS/SSL?
Authentication vs authorization?
What is SASL?
SASL/PLAIN?
SCRAM?
Kerberos?
ACLs?
How do you allow producer-only access to a topic?
How do you allow consumer-only access?
How do you manage secrets?
How do you encrypt data at rest?
How do you protect PII in events?
Should sensitive fields even be put into Kafka?
How do you audit Kafka access?
How do you rotate credentials without stopping producers?
Z. Disaster recovery / multi-region
What happens if an entire Kafka cluster fails?
What is your RPO?
What is your RTO?
How do you replicate Kafka across regions?
What is MirrorMaker 2 conceptually?
Active-active vs active-passive?
How do consumer offsets behave during DR?
How would consumers resume after regional failover?
How do you avoid duplicate events during failover?
How do you test Kafka disaster recovery?
What if one region is isolated but still accepting writes?
How do you reconcile events after recovery?
AA. Architect scenario questions — most important after offsets

These are the questions I would expect at your experience level:

You consume Kafka data, write to Delta, then crash before committing progress. What happens?
You commit offset first and Delta write fails. What happens?
Which design gives data loss and which gives duplicates?
Kafka contains 7 days retention. A bug was discovered in the last 5 days. Design the replay.
Production consumption must not stop during replay. How do you design it?
100 consumers are needed, but the topic has only 20 partitions. What happens?
Consumer lag suddenly grows from zero to 20 million. How do you troubleshoot it?
One partition has 80% of traffic. What caused it and how do you fix it?
A consumer keeps crashing on one malformed message. How do you stop the entire group from getting stuck?
A consumer writes to an external payment API. How do you replay safely without charging twice?
A downstream database supports no transactions with Kafka. How do you achieve effectively-once?
A producer sends duplicate business events with different Kafka offsets. How do you detect them?
Kafka retained the event, but your Structured Streaming checkpoint is corrupted. Design recovery.
Kafka has already expired the offset required by the checkpoint. What now?
Consumer has been down longer than Kafka retention. What happens?
Your topic has 12 partitions and you increase it to 24. What can happen to keyed ordering?
You need all events for one account strictly ordered but millions of accounts in parallel. Design partitioning.
Business requests global ordering. How would you challenge/design this requirement?
Source changes schema unexpectedly. Should your stream stop, quarantine, rescue or evolve?
One field changes from int to string. What do you do?
A new nullable field appears. What do you do?
The application needs both real-time processing and historical replay. How would you design topic retention and Bronze?
You have three downstream teams. Should they use one consumer group or three?
One team's consumer is slow. Does it affect the other consumer groups?
A broker fails while a producer is writing. What happens?
Partition leader dies after receiving an event but before replication. What determines whether data is lost?
Explain the interaction of replication factor, ISR, acks=all, and min.insync.replicas.
Kafka throughput doubles overnight. What would you scale first?
Consumer processing is CPU-heavy. Would adding Kafka partitions solve everything?
A consumer calls an API taking 30 seconds per message and rebalances continuously. Why?
How would you decouple slow external APIs from Kafka consumption?
You receive out-of-order CDC updates. How do you decide the current version?
Delete arrives before update because of processing/replay issues. How do you handle it?
Your Kafka-to-Delta job succeeds technically but data counts are wrong. How do you debug it?
How would you reconcile source Kafka counts with Bronze and Silver?
How do you prove no records were lost?
How do you prove duplicates were handled?
What metadata should Bronze retain for every Kafka record?
What happens if someone manually resets production offsets?
What controls would you put around replay?
Should replay be operator-driven, API-driven or pipeline-parameter-driven?
How do you audit who initiated a replay and which offsets were processed?
How do you prevent two simultaneous replay jobs from corrupting the target?
How do you handle side effects differently from analytical sinks?
How would you design Kafka ingestion capable of full replay, partial replay and single-key replay?
Your highest-priority 40

You do not need to memorize all 580 equally. For the next interview, I would first master:

Offsets: 212–237
Replay: 238–262
Delivery semantics: 263–280
Duplicates: 281–292
Retry/DLQ: 293–318
Consumer lag: 319–338
Kafka + Spark: 419–448
Architect scenarios: 536–580

And especially be able to answer this chain without thinking:

Kafka message
   ↓
Topic
   ↓
Partition
   ↓
Offset
   ↓
Consumer Group
   ↓
poll()
   ↓
Process
   ↓
Durable sink
   ↓
Commit/checkpoint

Failure?
 ├─ transient → retry
 ├─ bad event → DLQ/quarantine
 ├─ crash before commit → possible duplicate → idempotency
 ├─ committed too early → possible data loss
 └─ historical defect → replay/reset/seek/new consumer group
                         ↓
                      idempotent sink

If you can answer the offset + replay + duplicate + consumer-group + partition + checkpoint + DLQ + ordering sections properly, you will be much better prepared for the exact type of follow-up questioning that exposed the gap in the Capgemini interview.




Best Practices and Concepts for Architecture and Dimensional Design


Here we go.Define a fact table. Define a dimension table. Why keep them separate? What is the grain of a fact table? What happens if grain is unclear? Why are facts mostly numeric? What are additive, semi-additive, and non-additive facts? What is a surrogate key? Why not use business keys? What is a degenerate dimension? What is a junk dimension? When would you use each? What is a factless fact table? Scenario. Design a simple sales data model. What is SCD? Define types 0 to 6. Which ones are most common, and why? Walk through SCD 2 step by step. Why are surrogate keys critical in type2? How do facts point to the right historical dimension? What is a late arriving dimension? How do you handle it? What's the trade-off between star and snowflake schemas? Performance scenario. If a query is slow, how do you diagnose it? Modeling question. Given customer address changes daily, but history matters, which SCD type and why? We'll go one by one, and I'll evaluate after each.


types of agents

Presidio <email_address>
HMac- tokenize for all email same thing returned


User
 ↓
Auth0 / SSO
 ↓
JWT
 ↓
API Gateway
 ↓
Token validation
 ↓
Agent Orchestrator
 ↓
Policy / RBAC check
 ↓
┌───────────────┬───────────────┬───────────────┐
│ Finance Agent │ HR Agent      │ IT Agent      │
│ allowed       │ denied        │ limited       │
└───────────────┴───────────────┴───────────────┘
 ↓
Allowed tool call
 ↓
Vector DB / SQL / APIs
 ↓
ACL metadata filter
 ↓
LLM
 ↓
Output guardrail
 ↓
User




Developer commit
    ↓
GitHub / Azure DevOps
    ↓
CI Pipeline
    ↓
1. Lint + unit tests
2. Agent/tool tests
3. RAG evaluation tests
4. Security scans
5. Build Docker image
6. Push image to registry
    ↓


CD Pipeline
    ↓
Dev
    ↓
Integration tests
    ↓
UAT / Staging
    ↓
Approval
    ↓
Production
    ↓
Monitoring + rollback

                    RAG ARCHITECTURE

              ┌──────────────────────┐
              │  Documents / Sources │
              │ PDF / DB / API / S3  │
              └──────────┬───────────┘
                         ↓
                 Document Loader
                         ↓
               Clean / Normalize
                         ↓
                    Chunking
                         ↓
              Add Metadata / ACL
                         ↓
                  Embeddings
                         ↓
                 Vector Database
                         │
                         │
User Query               │
    ↓                    │
Authentication           │
    ↓                    │
Query preprocessing      │
    ↓                    │
Query embedding          │
    ↓                    │
Vector Search ───────────┘
    ↓
Metadata / ACL Filter
    ↓
Top-K Chunks
    ↓
Optional Reranker
    ↓
Prompt Construction
    ↓
LLM
    ↓
Guardrails / Citations
    ↓
Final Answer


Pattern	Simple meaning	Example
ReAct	Think → Act → Observe → repeat	Agent searches DB, sees result, then decides next tool
Tool-Use Agent	LLM decides which tool/API to call	Weather API, SQL, calculator
Router	Request ko correct agent/tool tak bhejna	Finance query → Finance Agent
Sequential	Agents/tasks ek ke baad ek	Research → Analyze → Write
Parallel	Multiple agents same time kaam kare	3 agents research different sources
Planner–Executor	Planner breaks goal into tasks; executor performs them	“Plan trip” → flights, hotel, itinerary
Supervisor / Manager	One manager controls specialist agents	Supervisor → HR/Finance/IT agents
Hierarchical	Multi-level supervisors and agents	Enterprise → department → specialist agents
Reflection / Critic	Agent apna answer review/improve kare	Writer → Critic → revised answer
Evaluator–Optimizer	One generates, another scores and improves	SQL generator + SQL validator
Debate / Multi-agent voting	Multiple agents propose answers, judge picks best	3 analysts + judge
Swarm / Peer-to-peer	Agents hand off directly without central supervisor	Sales agent → Legal agent → Support agent
Agentic RAG	Agent decides when/how to retrieve	Search vector DB, rerank, retry
Memory-based Agent	Agent remembers past state/preferences	Customer-support history
Human-in-the-loop	Human approval before sensitive action	Refund > ₹50k needs manager approval
Event-driven Agent	Triggered by an event	New email → classify → create ticket
Workflow + Agent hybrid	Deterministic steps + LLM where reasoning needed	RBAC deterministic, LLM for routing/explanation
——————————————————————————
Suppose sentence hai:
“The cat drinks milk.”
Transformer mein har word ek vector hota hai:

cat → [0.2, 0.8, 0.4, ...]

Self-Attention kya karta hai?
drinks doosre words ko dekhta hai:

The     ← low attention
cat     ← high attention
drinks
milk    ← high attention

So attention ke baad drinks ka vector contextual ho gaya:

drinks
  ↓
"I know CAT is doing the action
 and MILK is related to this action."

Attention = tokens ke beech information exchange.
Ab Feed-Forward Network (FFN)
Attention se jo drinks vector mila, FFN sirf us vector ko individually process karta hai.
Conceptually:

Contextual vector for "drinks"
[0.3, 0.7, 0.1, 0.9]
          ↓
      Linear layer
          ↓
        GELU
          ↓
      Linear layer
          ↓
[0.8, 0.2, 0.6, 0.4]

Mathematically:
F
F
N
(
x
)
=
W
2
 
G
E
L
U
(
W
1
x
+
b
1
)
+
b
2
FFN(x) = W_2 \, GELU(W_1x+b_1)+b_2
Same FFN separately har token par:

"The"    → FFN → new "The" representation
"cat"    → FFN → new "cat" representation
"drinks" → FFN → new "drinks" representation
"milk"   → FFN → new "milk" representation

FFN ke andar drinks directly cat ko nahi dekhta. Wo cross-token mixing attention already kar chuka hai.
Easy memory:

Self-Attention
     ↓
"Other words se kya information leni hai?"
     ↓
FFN
     ↓
"Ab is information ko process/transform kaise karna hai?"

Interview line:
“Self-attention mixes information across tokens, whereas the feed-forward network independently transforms the features of each token representation.”
Isi sentence ko next mein main actual 3-dimensional vectors + matrix multiplication se dikha sakta hoon, jisse FFN bilkul clear ho jayega.

step with a hidden state. LSTM is a special RNN that handles long dependencies better. Transformers replace recurrence with self-attention so each token can directly look at any other token. Multi-head just means we do that several times in parallel to capture different relationships. And the feed-forward network is a separate per-token MLP. Not a cross token mixer. Attention handles the mixing. And, very important correction for interviews : transformers absolutely use backpropagation, just not backpropagation through time, since there's no recurrence. What you should learn next in order is basic neural nets and backprop, then CNN fundamentals, then RNN, LSTM, GRU, after that sequence to sequence and attention, and then full transformer architecture, and then LLM specifics, and then fine-tuning methods like LoRA. Once that’s clear, the RAG patterns you've been discussing will make much more sense.


Optimizer activation function


Binary Logistic Regression
→ Sigmoid
→ Binary Cross-Entropy

Multiclass Logistic Regression
→ Softmax
→ Categorical Cross-Entropy