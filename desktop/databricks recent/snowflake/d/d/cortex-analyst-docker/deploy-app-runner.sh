#!/bin/bash
set -euo pipefail

# =============================================================================
# Deploy Cortex Analyst API to AWS App Runner
# =============================================================================
# Prerequisites: AWS CLI configured, Docker image already pushed to ECR.
# Run push-to-ecr.sh first, then run this script.
# =============================================================================

REGION="ap-southeast-7"
SERVICE_NAME="cortex-analyst-copay"
REPO_NAME="cortex-analyst-copay"
IMAGE_TAG="latest"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
echo_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }
echo_err()  { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Check prerequisites ---
if ! command -v aws &> /dev/null; then
    echo_err "AWS CLI not installed. Run: pip install awscli && aws configure"
    exit 1
fi

# --- Get AWS account ID ---
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region $REGION 2>/dev/null)
if [ -z "$ACCOUNT_ID" ]; then
    echo_err "Could not get AWS account ID. Run 'aws configure' first."
    exit 1
fi
ECR_IMAGE="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"

echo_info "AWS Account:  $ACCOUNT_ID"
echo_info "ECR Image:    $ECR_IMAGE"
echo_info "Service:      $SERVICE_NAME"
echo_info "Region:       $REGION"
echo ""

# --- Step 0: Create App Runner ECR access role (one-time setup) ---
IAM_ROLE="AppRunnerECRAccessRole"
if ! aws iam get-role --role-name "$IAM_ROLE" &> /dev/null; then
    echo_info "Creating IAM role $IAM_ROLE..."
    aws iam create-role \
        --role-name "$IAM_ROLE" \
        --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"build.apprunner.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    aws iam attach-role-policy \
        --role-name "$IAM_ROLE" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
    echo_ok "IAM role created: $IAM_ROLE"
else
    echo_ok "IAM role exists: $IAM_ROLE"
fi

# --- Load .env file for Snowflake credentials ---
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo_err ".env file not found at $ENV_FILE"
    echo_info "Create it with SF_USER, SF_PASSWORD, SF_ACCOUNT, SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA"
    exit 1
fi

source "$ENV_FILE"

# --- Step 1: Store SF_PASSWORD in SSM Parameter Store ---
echo_info "Step 1/3: Storing Snowflake password in SSM Parameter Store..."
aws ssm put-parameter \
    --name "/cortex-analyst/sf-password" \
    --value "$SF_PASSWORD" \
    --type SecureString \
    --overwrite \
    --region $REGION
SSM_ARN="arn:aws:ssm:$REGION:$ACCOUNT_ID:parameter/cortex-analyst/sf-password"
echo_ok "Password stored in SSM: /cortex-analyst/sf-password"

# --- Step 2: Check if App Runner service already exists ---
echo_info "Step 2/3: Creating App Runner service..."
EXISTING=$(aws apprunner list-services --region $REGION --query 'ServiceSummaryList[?ServiceName==`'$SERVICE_NAME''].ServiceArn' --output text 2>/dev/null || echo "")

if [ -n "$EXISTING" ]; then
    echo_info "Service already exists. Deleting old service..."
    aws apprunner delete-service --service-arn "$EXISTING" --region $REGION
    echo_info "Waiting for old service to delete..."
    aws apprunner wait service-deleted --service-arn "$EXISTING" --region $REGION 2>/dev/null || sleep 30
    echo_ok "Old service deleted"
fi

# --- Step 3: Create App Runner service ---
aws apprunner create-service \
    --service-name "$SERVICE_NAME" \
    --region $REGION \
    --source-configuration "{
        \"ImageRepository\": {
            \"ImageIdentifier\": \"$ECR_IMAGE\",
            \"ImageRepositoryType\": \"ECR\",
            \"ImageConfiguration\": {
                \"Port\": \"8080\",
                \"RuntimeEnvironmentVariables\": [
                    {\"Name\": \"SF_USER\", \"Value\": \"$SF_USER\"},
                    {\"Name\": \"SF_ACCOUNT\", \"Value\": \"$SF_ACCOUNT\"},
                    {\"Name\": \"SF_WAREHOUSE\", \"Value\": \"$SF_WAREHOUSE\"},
                    {\"Name\": \"SF_DATABASE\", \"Value\": \"$SF_DATABASE\"},
                    {\"Name\": \"SF_SCHEMA\", \"Value\": \"$SF_SCHEMA\"}
                ],
                \"RuntimeEnvironmentSecrets\": [
                    {\"Name\": \"SF_PASSWORD\", \"SecretArn\": \"$SSM_ARN\"}
                ]
            },
            \"ECRConfiguration\": {
                \"AccessRoleArn\": \"arn:aws:iam::$ACCOUNT_ID:role/AppRunnerECRAccessRole\"
            }
        },
        \"AutoDeploymentsEnabled\": false
    }" \
    --instance-configuration "{\"CPU\": \"0.5 vCPU\", \"Memory\": \"1 GB\"}" \
    --health-check-configuration "{\"Protocol\": \"HTTP\", \"Path\": \"/health\", \"Interval\": 10, \"Timeout\": 5, \"HealthyThreshold\": 2, \"UnhealthyThreshold\": 3}"

SERVICE_ARN=$(aws apprunner list-services --region $REGION --query 'ServiceSummaryList[?ServiceName==`'$SERVICE_NAME''].ServiceArn' --output text)
echo_ok "App Runner service created: $SERVICE_ARN"

# --- Wait for service to be ready ---
echo_info "Step 3/3: Waiting for service to start (this may take 2-5 minutes)..."
echo_info "  Status: $(aws apprunner describe-service --service-arn "$SERVICE_ARN" --region $REGION --query 'Service.Status' --output text)"

for i in $(seq 1 30); do
    STATUS=$(aws apprunner describe-service --service-arn "$SERVICE_ARN" --region $REGION --query 'Service.Status' --output text 2>/dev/null || echo "UNKNOWN")
    if [ "$STATUS" = "RUNNING" ]; then
        SERVICE_URL=$(aws apprunner describe-service --service-arn "$SERVICE_ARN" --region $REGION --query 'Service.ServiceUrl' --output text)
        echo ""
        echo "================================================"
        echo "  SERVICE IS RUNNING!"
        echo "================================================"
        echo "  URL:   https://$SERVICE_URL"
        echo "  Docs:  https://$SERVICE_URL/docs"
        echo "================================================"
        echo ""
        echo_info "Test endpoints:"
        echo "  curl https://$SERVICE_URL/health"
        echo "  curl https://$SERVICE_URL/total-spend"
        echo "  curl https://$SERVICE_URL/spend-by-brand"
        echo "  curl https://$SERVICE_URL/patient-count"
        echo "  curl -X POST https://$SERVICE_URL/query -H 'Content-Type: application/json' -d '{\"question\": \"What is the total Copay spend?\"}'"
        exit 0
    elif [ "$STATUS" = "CREATE_FAILED" ]; then
        echo_err "Service creation failed!"
        aws apprunner describe-service --service-arn "$SERVICE_ARN" --region $REGION --query 'Service' --output json
        exit 1
    fi
    echo -ne "\r  [$i/30] Status: $STATUS..."
    sleep 10
done

echo_err "Service did not become ready within 5 minutes."
echo_info "Check status: aws apprunner describe-service --service-arn $SERVICE_ARN --region $REGION"
exit 1