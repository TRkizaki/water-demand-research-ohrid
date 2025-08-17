#!/bin/bash

# Quick GCP Setup Script for Water Demand Research
# Run this after creating your GCP project

echo "🚀 Setting up GCP for Water Demand Research..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud CLI not found. Install it first:"
    echo "curl https://sdk.cloud.google.com | bash"
    exit 1
fi

# Get project ID
read -p "Enter your GCP Project ID: " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Project ID is required"
    exit 1
fi

echo "📋 Setting up project: $PROJECT_ID"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudscheduler.googleapis.com

# Create service account
echo "👤 Creating service account..."
gcloud iam service-accounts create water-demand-research \
    --display-name="Water Demand Research Service Account" \
    --description="Service account for water demand prediction research"

# Grant necessary permissions
echo "🔐 Setting up permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:water-demand-research@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:water-demand-research@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:water-demand-research@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download service account key
echo "🔑 Creating service account key..."
gcloud iam service-accounts keys create ./gcp-credentials.json \
    --iam-account=water-demand-research@$PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-credentials.json"

# Run our Python setup script
echo "🏗️ Setting up infrastructure..."
python infrastructure/gcp/setup_gcp.py --project-id $PROJECT_ID

echo "✅ GCP setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Add to your .env file:"
echo "   GOOGLE_CLOUD_PROJECT=$PROJECT_ID"
echo "   GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/gcp-credentials.json"
echo ""
echo "2. Test the setup:"
echo "   python -c \"from google.cloud import storage; print('GCP connection successful!')\""
echo ""
echo "3. Your GCP resources:"
echo "   - Bucket: water-demand-ohrid-$PROJECT_ID"
echo "   - BigQuery Dataset: water_demand_ohrid"
echo "   - Vertex AI: Initialized in europe-west3"