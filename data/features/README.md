# Features Directory

Contains engineered features ready for machine learning model training.

## Feature Categories

### Temporal Features
- Hour of day (0-23)
- Day of week (0-6)
- Month (1-12)
- Season indicators
- Holiday flags
- Weekend indicators

### Weather Features
- Temperature (°C)
- Humidity (%)
- Precipitation (mm/h)
- Wind speed (m/s)
- Atmospheric pressure (hPa)
- Cloud cover (%)

### Tourism Features
- Estimated tourist count
- Hotel occupancy rate
- Festival period indicators
- UNESCO site visitor impact
- Seasonal tourism multipliers

### Lag Features
- Previous 1, 2, 7, 14, 30 hour values
- Rolling statistics (mean, std, min, max)
- Seasonal decomposition components

### Interaction Features
- Weather × Tourism interactions
- Time × Season interactions
- Temperature × Tourist season

## Feature Files

### ohrid_features_complete.csv
- All engineered features for model training
- Target variable: water_demand_m3_per_hour
- Feature count: 50+ variables
- Ready for ML pipeline

## Feature Engineering Scripts

Generate features:
```bash
python src/feature_engineering/temporal_features.py
python src/feature_engineering/weather_features.py
python src/feature_engineering/tourism_features.py
python src/feature_engineering/lag_features.py
```