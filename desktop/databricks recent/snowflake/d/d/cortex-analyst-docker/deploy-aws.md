# Deploy Cortex Analyst API to AWS

This guide deploys the Cortex Analyst API as a Docker container on AWS.

## Quick Start (one command)

```bash
# 1. Download all files from the Databricks workspace folder:
#    cortex-analyst-docker/
#
# 2. On your local machine, cd into the folder and run:
cp .env.example .env     # then edit .env with your Snowflake password
mv Dockerfile.py Dockerfile   # rename (Databricks adds .py)
chmod +x full-deploy.sh
./full-deploy.sh          # builds, pushes to ECR, deploys to App Runner, tests
```

The `full-deploy.sh` script runs all 3 steps automatically:
1. **Push** — Builds Docker image, creates ECR repo, pushes image
2. **Deploy** — Creates IAM role, stores password in SSM, deploys to App Runner
3. **Test** — Tests all 8 endpoints against the live URL

## Prerequisites

- AWS CLI configured (`aws configure`)
- Docker installed
- Snowflake account credentials

## 1. Environment Variables

Create a `.env` file (never commit this):

```env
SF_USER=PRAKHAR1207SRIVASTAVA
SF_PASSWORD=your_password_here
SF_ACCOUNT=EMXEKCM-PC15902
SF_WAREHOUSE=COMPUTE_WH
SF_DATABASE=PRAK_DB
SF_SCHEMA=PRAK_SCH
```

## 2. Build & Test Locally

```bash
# Build the Docker image
docker build -t cortex-analyst-copay .

# Run locally
docker run -p 8080:8080 --env-file .env cortex-analyst-copay

# Test endpoints
curl http://localhost:8080/health
curl http://localhost:8080/total-spend
curl http://localhost:8080/spend-by-brand
curl http://localhost:8080/patient-count
curl -X POST http://localhost:8080/query -H 'Content-Type: application/json' -d '{"question": "What is the total Copay spend?"}'
curl -X POST http://localhost:8080/sql -H 'Content-Type: application/json' -d '{"sql": "SELECT * FROM PRAK_DB.PRAK_SCH.COPAY_CLAIMS LIMIT 5"}'
```

## 3. Push to Amazon ECR

### Option A: Automated (recommended)

```bash
chmod +x push-to-ecr.sh
./push-to-ecr.sh
```

The script auto-detects your AWS account ID, creates the ECR repo, builds, tags,
and pushes the image. It also handles the `Dockerfile.py` → `Dockerfile` rename.

### Option B: Manual

```bash
# Create ECR repository
aws ecr create-repository --repository-name cortex-analyst-copay --region ap-southeast-7

# Get ECR login
aws ecr get-login-password --region ap-southeast-7 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.ap-southeast-7.amazonaws.com

# Tag & push
docker tag cortex-analyst-copay <ACCOUNT_ID>.dkr.ecr.ap-southeast-7.amazonaws.com/cortex-analyst-copay:latest
docker push <ACCOUNT_ID>.dkr.ecr.ap-southeast-7.amazonaws.com/cortex-analyst-copay:latest
```

## 4a. Deploy to AWS App Runner (Simplest)

### Option A: Automated (recommended)

```bash
chmod +x deploy-app-runner.sh
./deploy-app-runner.sh
```

The script stores `SF_PASSWORD` in SSM Parameter Store (secure), creates the
App Runner service with health checks, waits for it to start, and prints the URL.

### Prerequisite: App Runner ECR Access Role

App Runner needs an IAM role to pull images from ECR. Create it once:

```bash
aws iam create-role \
  --role-name AppRunnerECRAccessRole \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"build.apprunner.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy \
  --role-name AppRunnerECRAccessRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
```

### Option B: Manual

```bash
aws apprunner create-service \
  --service-name cortex-analyst-copay \
  --region ap-southeast-7 \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "<ACCOUNT_ID>.dkr.ecr.ap-southeast-7.amazonaws.com/cortex-analyst-copay:latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8080",
        "RuntimeEnvironmentVariables": [
          {"Name": "SF_USER", "Value": "PRAKHAR1207SRIVASTAVA"},
          {"Name": "SF_ACCOUNT", "Value": "EMXEKCM-PC15902"},
          {"Name": "SF_WAREHOUSE", "Value": "COMPUTE_WH"},
          {"Name": "SF_DATABASE", "Value": "PRAK_DB"},
          {"Name": "SF_SCHEMA", "Value": "PRAK_SCH"}
        ],
        "RuntimeEnvironmentSecrets": [
          {"Name": "SF_PASSWORD", "SecretArn": "arn:aws:ssm:ap-southeast-7:<ACCOUNT_ID>:parameter/cortex-analyst/sf-password"}
        ]
      },
      "ECRConfiguration": {
        "AccessRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/AppRunnerECRAccessRole"
      }
    },
    "AutoDeploymentsEnabled": false
  }' \
  --instance-configuration '{"CPU": "0.5 vCPU", "Memory": "1 GB"}' \
  --health-check-configuration '{"Protocol": "HTTP", "Path": "/health", "Interval": 10, "Timeout": 5, "HealthyThreshold": 2, "UnhealthyThreshold": 3}'

# Store password securely first
aws ssm put-parameter --name "/cortex-analyst/sf-password" --value "your_password" --type SecureString --region ap-southeast-7
```

## 4b. Deploy to ECS Fargate

```bash
# Create cluster
aws ecs create-cluster --cluster-name cortex-analyst --region ap-southeast-7

# Register task definition
aws ecs register-task-definition \
  --family cortex-analyst-copay \
  --network-mode awsvpc \
  --requires-compatibilities FARGATE \
  --cpu 512 --memory 1024 \
  --container-definitions '[
    {
      "name": "api",
      "image": "<ACCOUNT_ID>.dkr.ecr.ap-southeast-7.amazonaws.com/cortex-analyst-copay:latest",
      "portMappings": [{"containerPort": 8080, "protocol": "tcp"}],
      "environment": [
        {"name": "SF_USER", "value": "PRAKHAR1207SRIVASTAVA"},
        {"name": "SF_ACCOUNT", "value": "EMXEKCM-PC15902"},
        {"name": "SF_WAREHOUSE", "value": "COMPUTE_WH"},
        {"name": "SF_DATABASE", "value": "PRAK_DB"},
        {"name": "SF_SCHEMA", "value": "PRAK_SCH"}
      ],
      "secrets": [
        {"name": "SF_PASSWORD", "valueFrom": "arn:aws:ssm:ap-southeast-7:<ACCOUNT_ID>:parameter/sf-password"}
      ]
    }
  ]' \
  --region ap-southeast-7

# Store password in AWS SSM Parameter Store (recommended)
aws ssm put-parameter --name "/sf-password" --value "your_password" --type SecureString --region ap-southeast-7

# Run the service
aws ecs run-task \
  --cluster cortex-analyst \
  --task-definition cortex-analyst-copay \
  --count 1 \
  --launch-type FARGATE \
  --network-configuration 'awsvpcConfiguration={subnets=["subnet-xxx"],securityGroups=["sg-xxx"],assignPublicIp="ENABLED"}' \
  --region ap-southeast-7
```

## 5. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Snowflake connectivity check |
| POST | `/query` | NL → SQL via Cortex Analyst (`{"question": "..."}`) |
| POST | `/sql` | Direct SQL execution (`{"sql": "SELECT ..."}`) |
| GET | `/schema` | List tables and columns |
| GET | `/total-spend` | Total Copay spend |
| GET | `/spend-by-brand` | Spend by brand |
| GET | `/patient-count` | Unique patient count |

## Security Notes

- Store `SF_PASSWORD` in AWS Secrets Manager or SSM Parameter Store (never in plaintext env vars in production)
- Use an Application Load Balancer with HTTPS for the ECS deployment
- Restrict the security group to allow inbound only from trusted sources
- The `/query` endpoint requires Cortex Analyst REST API to be enabled on your Snowflake account (non-trial tier)

## 6. Test the Deployed API

### Automated (recommended)

```bash
chmod +x test-api.sh

# Test against localhost (before deploying)
./test-api.sh

# Test against deployed App Runner URL
./test-api.sh https://<service-id>.ap-southeast-7.awsapprunner.com
```

The script tests all 8 endpoints and prints pass/fail with results:
- Health check with Snowflake version
- Total Copay spend
- Spend by brand (all brands)
- Unique patient count
- NL query (Cortex Analyst or graceful error)
- Direct SQL execution
- Schema listing

### Manual

```bash
# Replace $URL with your App Runner URL or http://localhost:8080
URL=https://<service-id>.ap-southeast-7.awsapprunner.com

curl $URL/health
curl $URL/total-spend
curl $URL/spend-by-brand
curl $URL/patient-count
curl -X POST $URL/query -H 'Content-Type: application/json' -d '{"question": "What is the total Copay spend?"}'
curl -X POST $URL/sql -H 'Content-Type: application/json' -d '{"sql": "SELECT * FROM PRAK_DB.PRAK_SCH.COPAY_CLAIMS LIMIT 5"}'
curl $URL/schema
```

### Expected Results

| Endpoint | Expected Response |
|----------|-------------------|
| `/health` | `{"status": "healthy", "snowflake_version": "10.34.101", "account": "ZS62970"}` |
| `/total-spend` | `{"total_copay_spend": 1480}` |
| `/spend-by-brand` | `{"results": [{"BRAND": "BRAND_A", "TOTAL_COPAY_SPEND": 660}, ...]}` |
| `/patient-count` | `{"unique_patients": 8}` |

## 7. Troubleshooting

### Docker Build Fails

| Error | Fix |
|-------|-----|
| `cannot find Dockerfile` | `mv Dockerfile.py Dockerfile` (Databricks adds .py extension) |
| `pip install failed` | Check `requirements.txt` syntax; ensure no extra blank lines |
| `ModuleNotFoundError: No module named 'fastapi'` | Ensure `requirements.txt` is copied before `app.py` in Dockerfile (it is) |
| `permission denied` on scripts | `chmod +x *.sh` to make all scripts executable |

### AWS CLI / Credentials

| Error | Fix |
|-------|-----|
| `Unable to locate credentials` | Run `aws configure` — enter Access Key, Secret Key, Region: `ap-southeast-7` |
| `ExpiredToken` | Refresh temporary credentials or use long-term keys |
| `AccessDenied` on ECR/IAM/AppRunner | Ensure your IAM user has these policies: `AmazonEC2ContainerRegistryFullAccess`, `AWSAppRunnerFullAccess`, `IAMFullAccess` (or specific permissions) |
| `Region not found` | Verify region: `aws configure get region` — should be `ap-southeast-7` |

### ECR Push Issues

| Error | Fix |
|-------|-----|
| `RepositoryNotFoundException` | Script auto-creates the repo; if manual, run `aws ecr create-repository --repository-name cortex-analyst-copay --region ap-southeast-7` |
| `denied: Your authorization token has expired` | Re-login: `aws ecr get-login-password --region ap-southeast-7 \| docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.ap-southeast-7.amazonaws.com` |
| `no basic auth credentials` | Same as above — ECR login expired, re-run the login command |
| `PushFailed` / large image | Ensure Dockerfile uses `python:3.12-slim` (not full image); check `.dockerignore` |

### App Runner Deployment Issues

| Error | Fix |
|-------|-----|
| `ResourceNotFoundException: ECR access role` | Run `deploy-app-runner.sh` which auto-creates `AppRunnerECRAccessRole`; or create manually (see section 4a) |
| `Service creation failed` | Check CloudWatch logs: `aws logs describe-log-groups --log-group-name-prefix /aws/apprunner/cortex-analyst-copay` |
| `CREATE_FAILED` status | Most common cause: App Runner can't pull ECR image. Verify: (1) IAM role has `AWSAppRunnerServicePolicyForECRAccess`, (2) ECR image exists: `aws ecr describe-images --repository-name cortex-analyst-copay` |
| `Health check failed` | App expects `/health` to return 200. Test locally first: `docker run -p 8080:8080 --env-file .env cortex-analyst-copay` then `curl http://localhost:8080/health` |
| `Service stuck in CREATE_FAILED` | Delete and retry: `aws apprunner delete-service --service-arn <arn> --region ap-southeast-7` then re-run `deploy-app-runner.sh` |
| `SSM parameter not found` | Create it: `aws ssm put-parameter --name "/cortex-analyst/sf-password" --value "your_password" --type SecureString --region ap-southeast-7` |

### API Returns Errors After Deployment

| Endpoint | Error | Fix |
|----------|-------|-----|
| `/health` → 503 | Snowflake connection failed. Check: (1) `.env` has correct `SF_PASSWORD`, (2) Snowflake account `EMXEKCM-PC15902` is accessible, (3) warehouse `COMPUTE_WH` is running |
| `/total-spend` → 503 | Same as /health — Snowflake connectivity issue |
| `/query` → 200 with `error` field | Expected on trial accounts — Cortex Analyst REST API returns 404. Use `/sql`, `/total-spend`, `/spend-by-brand`, `/patient-count` instead |
| `/sql` → 400 `SQL execution failed` | Check the SQL syntax; ensure table `PRAK_DB.PRAK_SCH.COPAY_CLAIMS` exists and is accessible |
| All endpoints → connection refused | App Runner service not ready yet. Check status: `aws apprunner describe-service --service-arn <arn> --region ap-southeast-7 --query 'Service.Status'` |
| All endpoints → 502 bad gateway | Container crashed. Check logs: `aws apprunner list-observability --service-arn <arn> --region ap-southeast-7` |

### Snowflake Trial Account Limitations

This Snowflake account (`ZS62970`) is a **trial account** with these limitations:

| Feature | Status | Workaround |
|---------|--------|------------|
| Cortex Analyst REST API (`/query`) | Returns 404 | Use direct SQL endpoints (`/sql`, `/total-spend`, `/spend-by-brand`, `/patient-count`) |
| `SYSTEM$BEARER_TOKEN` | Unavailable | App uses `conn.rest.request()` with session token instead |
| Cortex LLM functions (COMPLETE, SUMMARIZE, etc.) | Not available | Upgrade to full Snowflake account |

### Debug Commands

```bash
# Check App Runner service status
aws apprunner list-services --region ap-southeast-7
aws apprunner describe-service --service-arn <SERVICE_ARN> --region ap-southeast-7

# Check ECR images
aws ecr describe-images --repository-name cortex-analyst-copay --region ap-southeast-7

# Check SSM parameter
aws ssm get-parameter --name "/cortex-analyst/sf-password" --region ap-southeast-7

# Check IAM role
aws iam get-role --role-name AppRunnerECRAccessRole

# View App Runner logs (if observability enabled)
aws logs get-log-events --log-group-name /aws/apprunner/cortex-analyst-copay --log-stream-name <stream>

# Test locally before deploying
docker run -p 8080:8080 --env-file .env cortex-analyst-copay
curl http://localhost:8080/health
curl http://localhost:8080/total-spend
```

## 8. Production Optimization (Latency & Performance)

### Latency Breakdown

| Source | Cold Start | Warm Request | Mitigation |
|--------|-----------|-------------|------------|
| App Runner instance startup | 10-30s | 0ms | Use min-1 instance (auto-scaling config) |
| Snowflake warehouse auto-resume | 5-15s | 0ms | Keep warehouse warm or set AUTO_SUSPEND=60 |
| Snowflake connection establishment | 1-3s | 0ms (reused) | client_session_keep_alive=True (already set) |
| Cortex Analyst LLM call (/query) | 5-30s | 5-30s | Cache results; use direct SQL endpoints |
| Aggregate SQL query | 0.5-2s | 0.5-2s | TTL cache (60s default, configurable) |

### Optimization Features (v1.1.0)

All optimizations are already implemented in app.py v1.1.0:

| Feature | Config | Effect |
|--------|--------|-------|
| TTL caching | CACHE_TTL_SECONDS=60 | Pre-built queries cached 60s - eliminates redundant Snowflake calls |
| Thread-safe connection | _conn_lock | Concurrent requests don't corrupt shared connection |
| Session keep-alive | client_session_keep_alive=True | Prevents idle connection close - no reconnect latency |
| Connect timeout | connect_timeout=10 | Fails fast if Snowflake unreachable |
| Network timeout | network_timeout=30 | Detects stalled connections |
| Query timeout | SF_QUERY_TIMEOUT=30 | Kills long queries - prevents blocking |
| Structured logging | LOG_LEVEL=INFO | All requests, cache hits/misses, connections logged |
| Graceful reconnection | Auto-reconnect | Connection drops trigger reconnect on next request |

### Environment Variables for Tuning

```env
CACHE_TTL_SECONDS=60       # Cache TTL for pre-built queries
SF_QUERY_TIMEOUT=30        # Max query execution time
LOG_LEVEL=INFO             # DEBUG for verbose, WARNING for production
```

### App Runner Auto-Scaling (reduce cold starts)

```bash
aws apprunner create-auto-scaling-configuration \
  --auto-scaling-configuration-name "cortex-min1" \
  --max-concurrency 10 \
  --min-size 1 \
  --max-size 3 \
  --region ap-southeast-7

aws apprunner update-service \
  --service-arn <SERVICE_ARN> \
  --auto-scaling-configuration-arn <AUTO_SCALING_ARN> \
  --region ap-southeast-7
```

### Snowflake Warehouse Optimization

```sql
ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;
-- Or keep warm (no auto-suspend):
ALTER WAREHOUSE COMPUTE_WH SET AUTO_SUSPEND = NULL;
```

### Production Checklist

- [ ] Set CACHE_TTL_SECONDS based on data freshness (60s default, 300s for slow-changing)
- [ ] Configure App Runner auto-scaling with min-1 to eliminate cold starts
- [ ] Set AUTO_SUSPEND=60 on Snowflake warehouse (balance cost vs latency)
- [ ] Set LOG_LEVEL=WARNING in production to reduce log noise
- [ ] Monitor /health with App Runner health checks (already configured)
- [ ] Run test-api.sh after each deployment
- [ ] Set up CloudWatch alarms for 5xx errors and high latency
- [ ] Add API key authentication before public exposure

## 9. Cost Saving Tips

### AWS App Runner Costs

| Setting | Default | Cost-Saving Alternative | Savings |
|---------|---------|----------------------|---------|
| Instance size | 0.5 vCPU / 1 GB | 0.25 vCPU / 0.5 GB | ~50% (if app fits in memory) |
| Auto-scaling min-size | 1 (always-on) | 0 (scale-to-zero) | ~70% (adds cold start latency) |
| Auto-scaling max-size | 3 | 2 | Cap spend under load spikes |
| Data transfer | NAT gateway | VPC endpoints for ECR/SSM | ~$0.045/GB saved |

```bash
# Cheapest viable config: scale-to-zero, smallest instance
aws apprunner create-auto-scaling-configuration \
  --auto-scaling-configuration-name "cortex-budget" \
  --max-concurrency 5 \
  --min-size 0 \
  --max-size 2 \
  --region ap-southeast-7

# Update service to use cheaper instance
aws apprunner update-service \
  --service-arn <SERVICE_ARN> \
  --instance-configuration '{"CPU": "0.25 vCPU", "Memory": "0.5 GB"}' \
  --auto-scaling-configuration-arn <AUTO_SCALING_ARN> \
  --region ap-southeast-7
```

**Estimated monthly cost (ap-southeast-7):**

| Scenario | Instance | Min Size | Est. Monthly Cost |
|----------|----------|----------|-------------------|
| Always-on (min-1) | 0.5 vCPU / 1 GB | 1 | ~$25-30 |
| Always-on (min-1) | 0.25 vCPU / 0.5 GB | 1 | ~$15-18 |
| Scale-to-zero | 0.25 vCPU / 0.5 GB | 0 | ~$5-8 (usage-based) |

### Snowflake Warehouse Costs

The warehouse is the most expensive component. Optimize aggressively:

```sql
-- CHEAPEST: auto-suspend after 60s of idle, auto-resume on next query
ALTER WAREHOUSE COMPUTE_WH SET
  WAREHOUSE_SIZE = XSMALL          -- smallest size (1 credit/hr)
  AUTO_SUSPEND = 60                -- suspend after 1 min idle
  AUTO_RESUME = TRUE              -- resume on next query
  INITIALLY_SUSPENDED = TRUE;     -- start suspended

-- If XSMALL is too slow, try SMALL but keep AUTO_SUSPEND=60
ALTER WAREHOUSE COMPUTE_WH SET WAREHOUSE_SIZE = SMALL AUTO_SUSPEND = 60;
```

| Warehouse Size | Credits/hr | Cost/hr (AWS ap-southeast-7) | Best For |
|---------------|-----------|----------------------------|----------|
| XSMALL | 1 | ~$2.00 | Dev/testing, small tables |
| SMALL | 2 | ~$4.00 | Production (recommended) |
| MEDIUM | 4 | ~$8.00 | Concurrent users |
| LARGE | 8 | ~$16.00 | Heavy analytics |

**Snowflake cost estimate for this app:**
- COPAY_CLAIMS table is small (~20 rows) — queries complete in <1 second
- XSMALL warehouse with AUTO_SUSPEND=60: ~$2-5/month (only bills for active query time)
- If warehouse runs 8h/day: ~$16/day — avoid this by keeping AUTO_SUSPEND low

### ECR Storage Costs

```bash
# Delete old images (keep only latest)
aws ecr batch-delete-image \
  --repository-name cortex-analyst-copay \
  --image-ids $(aws ecr list-images --repository-name cortex-analyst-copay --region ap-southeast-7 --filter tagStatus=UNTAGGED --query 'imageIds[*]' --output text | sed 's/\t/ /g') \
  --region ap-southeast-7

# Set lifecycle policy: keep only 3 most recent images
aws ecr put-lifecycle-policy \
  --repository-name cortex-analyst-copay \
  --lifecycle-policy-text '{"rules":[{"rulePriority":1,"description":"Keep 3 images","selection":{"tagStatus":"Any","countType":"ImageCountMoreThan","countNumber":3},"action":{"type":"expire"}}]}' \
  --region ap-southeast-7
```

ECR storage: ~$0.10/GB/month. A slim Python image is ~150MB, so even 10 images costs <$0.20/month.

### SSM Parameter Store

- Standard parameters: **FREE** (no cost for up to 10,000 parameters)
- Advanced parameters: $0.05/parameter/month (not needed for this app)
- The app uses 1 SecureString parameter — **$0/month**

### Data Transfer Costs

| Transfer | Cost | Optimization |
|----------|------|-------------|
| App Runner → Internet (Snowflake) | $0.09/GB | Minimal — API responses are small JSON |
| ECR → App Runner (image pull) | $0.09/GB | One-time per deployment (~150MB) |
| Snowflake → App Runner (query results) | $0.09/GB | Negligible — small result sets |

Total data transfer for this app: **<$1/month** (small JSON responses).

### Total Monthly Cost Estimate

| Component | Budget Mode | Production Mode |
|-----------|------------|----------------|
| App Runner (scale-to-zero, 0.25 vCPU) | $5-8 | $15-30 (min-1, 0.5 vCPU) |
| Snowflake (XSMALL, AUTO_SUSPEND=60) | $2-5 | $10-20 (SMALL) |
| ECR storage | $0.10 | $0.10 |
| SSM Parameter Store | $0 | $0 |
| Data transfer | $0.50 | $1 |
| **Total** | **~$8-14/month** | **~$26-51/month** |

### Cost vs Latency Tradeoff

| Setting | Cost | Cold Start Latency | Recommended For |
|--------|------|-------------------|----------------|
| Scale-to-zero + XSMALL + AUTO_SUSPEND=60 | Lowest | 15-45s | Dev/testing, low-traffic |
| Min-1 + XSMALL + AUTO_SUSPEND=60 | Low | 0ms (warm) | Production (small dataset) |
| Min-1 + SMALL + AUTO_SUSPEND=NULL | High | 0ms (warm) | Production (concurrent users) |