# Processed Data Directory

Contains cleaned and preprocessed datasets ready for feature engineering.

## Processing Pipeline

### 1. Data Cleaning
- Remove outliers and anomalies
- Handle missing values
- Standardize timestamps
- Validate data quality

### 2. Data Integration
- Merge weather, tourism, and water consumption data
- Align timestamps to hourly frequency
- Handle different data source formats

### 3. Quality Control
- Data validation rules
- Completeness checks
- Range validation
- Consistency verification

## Processed Datasets

### ohrid_water_demand_cleaned.csv
- Cleaned synthetic water demand data
- Quality score: 100/100
- Records: 26,257 hours (2021-2023)

### ohrid_integrated_dataset.csv (future)
- Combined real data from all sources
- Weather + Tourism + Water consumption
- Quality score varies by API availability

## Processing Scripts

Run data processing:
```bash
python src/data_processing/clean_raw_data.py
python src/data_processing/integrate_sources.py
```