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