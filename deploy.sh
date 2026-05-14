#!/bin/bash

# Cloud Run Deployment Script for Retail Insights Assistant
# Usage: ./deploy.sh YOUR_PROJECT_ID [region] [service-account-email]

set -e

PROJECT_ID="${1:-}"
REGION="${2:-us-central1}"
SERVICE_ACCOUNT_EMAIL="${3:-}"
IMAGE_NAME="retail-insights-assistant"
SERVICE_NAME="retail-insights-assistant"

# Validate inputs
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: PROJECT_ID is required"
    echo "Usage: $0 YOUR_PROJECT_ID [region] [service-account-email]"
    exit 1
fi

echo "🚀 Retail Insights Assistant - Cloud Run Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Project ID:      $PROJECT_ID"
echo "Region:          $REGION"
echo "Image:           gcr.io/$PROJECT_ID/$IMAGE_NAME"
echo "Service:         $SERVICE_NAME"
echo ""

# Step 1: Enable required APIs
echo "📡 Enabling required Google Cloud APIs..."
gcloud services enable run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com \
    --project="$PROJECT_ID" || true

# Step 2: Create service account if needed
if [ -z "$SERVICE_ACCOUNT_EMAIL" ]; then
    SA_NAME="retail-insights-runner"
    SERVICE_ACCOUNT_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    
    # Check if service account exists
    if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
        echo "👤 Creating service account: $SA_NAME"
        gcloud iam service-accounts create "$SA_NAME" \
            --display-name="Retail Insights Cloud Run Service Account" \
            --project="$PROJECT_ID"
        
        # Grant necessary roles
        echo "🔐 Granting IAM roles..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
            --role="roles/aiplatform.user" \
            --quiet || true
        
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
            --role="roles/logging.logWriter" \
            --quiet || true
    else
        echo "✓ Service account already exists: $SERVICE_ACCOUNT_EMAIL"
    fi
fi

# Step 3: Build and push container image
echo ""
echo "🐳 Building and pushing container image..."
gcloud builds submit \
    --tag "gcr.io/$PROJECT_ID/$IMAGE_NAME" \
    --project="$PROJECT_ID"

# Step 4: Deploy to Cloud Run
echo ""
echo "☁️  Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image "gcr.io/$PROJECT_ID/$IMAGE_NAME" \
    --platform managed \
    --region "$REGION" \
    --service-account "$SERVICE_ACCOUNT_EMAIL" \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION,VERTEX_MODEL=gemini-2.5-flash" \
    --memory 2Gi \
    --cpu 1 \
    --timeout 900 \
    --allow-unauthenticated \
    --project="$PROJECT_ID"

echo ""
echo "✅ Deployment completed!"
echo ""
echo "📋 Next steps:"
echo "1. Open your Cloud Run service URL in the browser"
echo "2. Upload a test CSV file"
echo "3. Verify schema detection and KPI cards"
echo "4. Test summarization and Q&A"
echo ""
gcloud run services describe "$SERVICE_NAME" \
    --platform managed \
    --region "$REGION" \
    --project="$PROJECT_ID" \
    --format='value(status.url)'
