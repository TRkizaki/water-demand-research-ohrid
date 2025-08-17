
# Terraform configuration for Ohrid Water Demand Research

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = "expanded-flame-469305-k1"
  region  = "europe-west3"
}

# Cloud Storage bucket
resource "google_storage_bucket" "data_lake" {
  name          = "water-demand-ohrid-expanded-flame-469305-k1"
  location      = "europe-west3"
  force_destroy = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

# BigQuery dataset
resource "google_bigquery_dataset" "water_demand" {
  dataset_id    = "water_demand_ohrid"
  friendly_name = "Water Demand Ohrid"
  description   = "Water demand prediction data for Ohrid, North Macedonia"
  location      = "europe-west3"

  default_table_expiration_ms = 7776000000  # 90 days
}

# Vertex AI Workbench instance for ML development
resource "google_notebooks_instance" "ml_workbench" {
  name         = "ohrid-water-demand-workbench"
  location     = "europe-west3-a"
  machine_type = "n1-standard-4"

  vm_image {
    project      = "deeplearning-platform-release"
    image_family = "tf-2-11-cu113-notebooks"
  }

  install_gpu_driver = false
  boot_disk_type     = "PD_SSD"
  boot_disk_size_gb  = 100

  no_public_ip    = false
  no_proxy_access = false

  network = "default"
  subnet  = "default"

  labels = {
    environment = "research"
    project     = "water-demand-ohrid"
  }
}

# Cloud Scheduler for automated data collection
resource "google_cloud_scheduler_job" "data_collection" {
  name             = "collect-water-demand-data"
  description      = "Hourly water demand data collection"
  schedule         = "0 * * * *"
  time_zone        = "Europe/Skopje"
  attempt_deadline = "320s"

  http_target {
    http_method = "POST"
    uri         = "https://europe-west3-expanded-flame-469305-k1.cloudfunctions.net/collect-water-data"
  }
}

# Outputs
output "bucket_name" {
  value = google_storage_bucket.data_lake.name
}

output "bigquery_dataset" {
  value = google_bigquery_dataset.water_demand.dataset_id
}

output "workbench_instance" {
  value = google_notebooks_instance.ml_workbench.name
}
