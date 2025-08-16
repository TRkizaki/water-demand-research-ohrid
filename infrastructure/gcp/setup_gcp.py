"""
GCP Infrastructure Setup for Ohrid Water Demand Research

This script automates the setup of Google Cloud Platform resources
for the water demand prediction research project.
"""

import os
import json
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import aiplatform
from google.auth import default
import yaml


class GCPInfrastructureSetup:
    """Set up GCP infrastructure for water demand research."""
    
    def __init__(self, project_id: str, region: str = "europe-west3"):
        self.project_id = project_id
        self.region = region
        self.bucket_name = f"water-demand-ohrid-{project_id}"
        
        # Initialize clients
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)
        
    def create_storage_bucket(self) -> None:
        """Create Cloud Storage bucket for data lake."""
        try:
            # Check if bucket already exists
            bucket = self.storage_client.bucket(self.bucket_name)
            if bucket.exists():
                print(f"Bucket {self.bucket_name} already exists")
                return
            
            # Create bucket
            bucket = self.storage_client.create_bucket(
                self.bucket_name, 
                location=self.region
            )
            
            # Set up folder structure
            folders = [
                "data/raw/",
                "data/processed/", 
                "data/external/",
                "data/features/",
                "models/",
                "experiments/",
                "logs/"
            ]
            
            for folder in folders:
                blob = bucket.blob(folder + ".gitkeep")
                blob.upload_from_string("")
            
            print(f"Created bucket: {self.bucket_name}")
            
        except Exception as e:
            print(f"Error creating bucket: {e}")
    
    def create_bigquery_dataset(self) -> None:
        """Create BigQuery dataset for structured data."""
        dataset_id = "water_demand_ohrid"
        
        try:
            dataset = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
            dataset.location = self.region
            dataset.description = "Water demand prediction data for Ohrid, North Macedonia"
            
            # Check if dataset exists
            try:
                self.bq_client.get_dataset(dataset)
                print(f"Dataset {dataset_id} already exists")
                return
            except:
                pass
            
            # Create dataset
            dataset = self.bq_client.create_dataset(dataset, timeout=30)
            print(f"Created dataset: {dataset_id}")
            
            # Create tables
            self._create_bq_tables(dataset_id)
            
        except Exception as e:
            print(f"Error creating BigQuery dataset: {e}")
    
    def _create_bq_tables(self, dataset_id: str) -> None:
        """Create BigQuery tables for different data types."""
        
        # Water demand data table
        demand_schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("water_demand_m3_per_hour", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("water_production_m3_per_hour", "FLOAT"),
            bigquery.SchemaField("population", "INTEGER"),
            bigquery.SchemaField("tourists_estimated", "INTEGER"),
            bigquery.SchemaField("temperature", "FLOAT"),
            bigquery.SchemaField("humidity", "FLOAT"),
            bigquery.SchemaField("precipitation", "FLOAT"),
            bigquery.SchemaField("wind_speed", "FLOAT"),
            bigquery.SchemaField("pressure", "FLOAT"),
            bigquery.SchemaField("cloud_cover", "INTEGER"),
            bigquery.SchemaField("hour", "INTEGER"),
            bigquery.SchemaField("day_of_week", "INTEGER"),
            bigquery.SchemaField("month", "INTEGER"),
            bigquery.SchemaField("is_weekend", "BOOLEAN"),
            bigquery.SchemaField("is_holiday", "BOOLEAN"),
            bigquery.SchemaField("is_tourist_season", "BOOLEAN"),
            bigquery.SchemaField("is_festival_period", "BOOLEAN")
        ]
        
        # Model predictions table
        predictions_schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("prediction", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("actual", "FLOAT"),
            bigquery.SchemaField("error", "FLOAT"),
            bigquery.SchemaField("model_version", "STRING"),
            bigquery.SchemaField("prediction_horizon_hours", "INTEGER")
        ]
        
        # Model performance table
        performance_schema = [
            bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("evaluation_date", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("mae", "FLOAT"),
            bigquery.SchemaField("rmse", "FLOAT"),
            bigquery.SchemaField("mape", "FLOAT"),
            bigquery.SchemaField("r2", "FLOAT"),
            bigquery.SchemaField("test_period_start", "TIMESTAMP"),
            bigquery.SchemaField("test_period_end", "TIMESTAMP")
        ]
        
        tables = [
            ("water_demand_data", demand_schema),
            ("model_predictions", predictions_schema),
            ("model_performance", performance_schema)
        ]
        
        for table_name, schema in tables:
            table_id = f"{self.project_id}.{dataset_id}.{table_name}"
            
            try:
                self.bq_client.get_table(table_id)
                print(f"Table {table_name} already exists")
                continue
            except:
                pass
            
            table = bigquery.Table(table_id, schema=schema)
            table = self.bq_client.create_table(table)
            print(f"Created table: {table_name}")
    
    def setup_vertex_ai(self) -> None:
        """Initialize Vertex AI for model training and deployment."""
        try:
            aiplatform.init(
                project=self.project_id,
                location=self.region
            )
            print(f"Vertex AI initialized for project {self.project_id}")
            
        except Exception as e:
            print(f"Error setting up Vertex AI: {e}")
    
    def create_cloud_function_config(self) -> None:
        """Generate Cloud Function configuration for automated data collection."""
        
        config = {
            "runtime": "python39",
            "entry_point": "collect_data",
            "environment_variables": {
                "PROJECT_ID": self.project_id,
                "BUCKET_NAME": self.bucket_name,
                "DATASET_ID": "water_demand_ohrid"
            },
            "trigger": {
                "schedule": "0 * * * *",  # Every hour
                "timezone": "Europe/Skopje"
            }
        }
        
        os.makedirs("infrastructure/gcp/cloud_functions", exist_ok=True)
        
        with open("infrastructure/gcp/cloud_functions/config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("Cloud Function configuration created")
    
    def generate_terraform_config(self) -> None:
        """Generate Terraform configuration for infrastructure as code."""
        
        terraform_config = f"""
# Terraform configuration for Ohrid Water Demand Research

terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 4.0"
    }}
  }}
}}

provider "google" {{
  project = "{self.project_id}"
  region  = "{self.region}"
}}

# Cloud Storage bucket
resource "google_storage_bucket" "data_lake" {{
  name          = "{self.bucket_name}"
  location      = "{self.region}"
  force_destroy = true

  versioning {{
    enabled = true
  }}

  lifecycle_rule {{
    condition {{
      age = 90
    }}
    action {{
      type = "Delete"
    }}
  }}
}}

# BigQuery dataset
resource "google_bigquery_dataset" "water_demand" {{
  dataset_id    = "water_demand_ohrid"
  friendly_name = "Water Demand Ohrid"
  description   = "Water demand prediction data for Ohrid, North Macedonia"
  location      = "{self.region}"

  default_table_expiration_ms = 7776000000  # 90 days
}}

# Vertex AI Workbench instance for ML development
resource "google_notebooks_instance" "ml_workbench" {{
  name         = "ohrid-water-demand-workbench"
  location     = "{self.region}-a"
  machine_type = "n1-standard-4"

  vm_image {{
    project      = "deeplearning-platform-release"
    image_family = "tf-2-11-cu113-notebooks"
  }}

  install_gpu_driver = false
  boot_disk_type     = "PD_SSD"
  boot_disk_size_gb  = 100

  no_public_ip    = false
  no_proxy_access = false

  network = "default"
  subnet  = "default"

  labels = {{
    environment = "research"
    project     = "water-demand-ohrid"
  }}
}}

# Cloud Scheduler for automated data collection
resource "google_cloud_scheduler_job" "data_collection" {{
  name             = "collect-water-demand-data"
  description      = "Hourly water demand data collection"
  schedule         = "0 * * * *"
  time_zone        = "Europe/Skopje"
  attempt_deadline = "320s"

  http_target {{
    http_method = "POST"
    uri         = "https://{self.region}-{self.project_id}.cloudfunctions.net/collect-water-data"
  }}
}}

# Outputs
output "bucket_name" {{
  value = google_storage_bucket.data_lake.name
}}

output "bigquery_dataset" {{
  value = google_bigquery_dataset.water_demand.dataset_id
}}

output "workbench_instance" {{
  value = google_notebooks_instance.ml_workbench.name
}}
"""
        
        os.makedirs("infrastructure/gcp/terraform", exist_ok=True)
        
        with open("infrastructure/gcp/terraform/main.tf", "w") as f:
            f.write(terraform_config)
        
        print("Terraform configuration generated")
    
    def setup_all(self) -> None:
        """Set up all GCP infrastructure components."""
        print(f"Setting up GCP infrastructure for project: {self.project_id}")
        print(f"Region: {self.region}")
        print("=" * 50)
        
        self.create_storage_bucket()
        self.create_bigquery_dataset()
        self.setup_vertex_ai()
        self.create_cloud_function_config()
        self.generate_terraform_config()
        
        print("\n" + "=" * 50)
        print("GCP Infrastructure setup completed!")
        print(f"Bucket: {self.bucket_name}")
        print(f"BigQuery dataset: water_demand_ohrid")
        print("Next steps:")
        print("1. Upload synthetic data to Cloud Storage")
        print("2. Load data into BigQuery")
        print("3. Set up Cloud Functions for data collection")
        print("4. Start ML experiments in Vertex AI Workbench")


def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up GCP infrastructure for water demand research")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--region", default="europe-west3", help="GCP Region")
    
    args = parser.parse_args()
    
    setup = GCPInfrastructureSetup(args.project_id, args.region)
    setup.setup_all()


if __name__ == "__main__":
    main()