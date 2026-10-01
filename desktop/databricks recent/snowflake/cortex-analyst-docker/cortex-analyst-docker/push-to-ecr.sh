#!/bin/bash
set -euo pipefail

# =============================================================================
# Push Cortex Analyst API Docker image to AWS ECR
# =============================================================================
# Prerequisites: Docker, AWS CLI (aws configure), and a .env file
# Run this script on your local machine or CI/CD runner.
# =============================================================================

# --- Configuration ---
REGION="ap-southeast-7"
REPO_NAME="cortex-analyst-copay"
IMAGE_TAG="latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Colors ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
echo_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }
echo_err()  { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Check prerequisites ---
if ! command -v docker &> /dev/null; then
    echo_err "Docker is not installed. Install it first: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v aws &> /dev/null; then
    echo_err "AWS CLI is not installed. Install it first: https://aws.amazon.com/cli/"
    exit 1
fi

# --- Check .env file ---
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo_err ".env file not found at $SCRIPT_DIR/.env"
    echo_info "Create it with:"
    echo '  SF_USER=PRAKHAR1207SRIVASTAVA'
    echo '  SF_PASSWORD=your_password'
    echo '  SF_ACCOUNT=EMXEKCM-PC15902'
    echo '  SF_WAREHOUSE=COMPUTE_WH'
    echo '  SF_DATABASE=PRAK_DB'
    echo '  SF_SCHEMA=PRAK_SCH'
    exit 1
fi

# --- Check Dockerfile ---
DOCKERFILE="$SCRIPT_DIR/Dockerfile"
if [ ! -f "$DOCKERFILE" ]; then
    # Try Dockerfile.py (Databricks adds .py extension)
    if [ -f "$SCRIPT_DIR/Dockerfile.py" ]; then
        echo_info "Renaming Dockerfile.py -> Dockerfile"
        cp "$SCRIPT_DIR/Dockerfile.py" "$DOCKERFILE"
    else
        echo_err "Dockerfile not found in $SCRIPT_DIR"
        exit 1
    fi
fi

# --- Get AWS account ID ---
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region $REGION 2>/dev/null)
if [ -z "$ACCOUNT_ID" ]; then
    echo_err "Could not get AWS account ID. Run 'aws configure' first."
    exit 1
fi
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
FULL_IMAGE="$ECR_URI/$REPO_NAME:$IMAGE_TAG"

echo_info "AWS Account:  $ACCOUNT_ID"
echo_info "ECR URI:       $ECR_URI"
echo_info "Image:         $FULL_IMAGE"
echo_info "Region:        $REGION"
echo ""

# --- Step 1: Build Docker image ---
echo_info "Step 1/4: Building Docker image..."
docker build -t $REPO_NAME "$SCRIPT_DIR"
echo_ok "Image built: $REPO_NAME:$IMAGE_TAG"

# --- Step 2: Create ECR repository (if not exists) ---
echo_info "Step 2/4: Creating ECR repository..."
if ! aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION &> /dev/null; then
    aws ecr create-repository --repository-name $REPO_NAME --region $REGION
    echo_ok "Created ECR repository: $REPO_NAME"
else
    echo_ok "ECR repository already exists: $REPO_NAME"
fi

# --- Step 3: Login to ECR ---
echo_info "Step 3/4: Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI
echo_ok "Logged into ECR"

# --- Step 4: Tag and push ---
echo_info "Step 4/4: Tagging and pushing image..."
docker tag $REPO_NAME:$IMAGE_TAG $FULL_IMAGE
docker push $FULL_IMAGE
echo_ok "Image pushed to ECR!"

echo ""
echo "================================================"
echo "  ECR Image URI:"
echo "  $FULL_IMAGE"
echo "================================================"
echo ""
echo_info "Next steps:"
echo "  Deploy to App Runner:  See deploy-aws.md section 4a"
echo "  Deploy to ECS Fargate: See deploy-aws.md section 4b"
echo ""
echo_info "Quick App Runner deploy:"
echo "  aws apprunner create-service \\
    --service-name $REPO_NAME \\
    --source-configuration '{\"ImageRepository\":{\"ImageIdentifier\":\"$FULL_IMAGE\",\"ImageRepositoryType\":\"ECR\",\"ImageConfiguration\":{\"Port\":\"8080\"}}}' \\
    --region $REGION"