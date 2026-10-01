#!/bin/bash
set -euo pipefail

# Full Deploy: Push to ECR, Deploy to App Runner, Test API
# Run on your local machine (requires Docker + AWS CLI)
# Usage: chmod +x full-deploy.sh && ./full-deploy.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================"
echo "  Cortex Analyst API - Full AWS Deployment"
echo "  Step 1: Build and Push to ECR"
echo "  Step 2: Deploy to App Runner"
echo "  Step 3: Test All Endpoints"
echo "======================================================"
echo ""

# Step 1: Push to ECR
echo "--- STEP 1/3: Build and Push Docker Image to ECR ---"
echo ""
cd "$SCRIPT_DIR"
chmod +x push-to-ecr.sh
./push-to-ecr.sh
echo ""
echo "Step 1 complete - image pushed to ECR"
echo ""

# Step 2: Deploy to App Runner
echo "--- STEP 2/3: Deploy to AWS App Runner ---"
echo ""
cd "$SCRIPT_DIR"
chmod +x deploy-app-runner.sh
./deploy-app-runner.sh 2>&1 | tee /tmp/apprunner-output.txt
echo ""
echo "Step 2 complete - App Runner service deployed"
echo ""

# Extract App Runner URL
APP_URL=$(grep -oE 'https://[a-z0-9]+\.ap-southeast-7\.awsapprunner\.com' /tmp/apprunner-output.txt | head -1)
if [ -z "$APP_URL" ]; then
    echo "Could not auto-detect App Runner URL."
    echo "  Find it at: https://console.aws.amazon.com/apprunner/"
    echo "  Then run: ./test-api.sh <your-url>"
    exit 0
fi

# Step 3: Test
echo "--- STEP 3/3: Test All API Endpoints ---"
echo ""
cd "$SCRIPT_DIR"
chmod +x test-api.sh
./test-api.sh "$APP_URL"

echo ""
echo "======================================================"
echo "  Deployment Complete!"
echo "  API URL: $APP_URL"
echo "  Docs:   $APP_URL/docs"
echo "======================================================"