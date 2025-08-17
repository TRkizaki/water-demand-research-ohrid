# External Data Directory

This directory contains data from external sources for Ohrid water demand research.

## Data Sources

### Weather Data
- **Source**: OpenWeatherMap API
- **Format**: JSON/CSV hourly records
- **Variables**: temperature, humidity, precipitation, wind_speed, pressure
- **Location**: Ohrid coordinates (41.1175, 20.8016)

### Tourism Data
- **Source**: Municipal tourism office, booking APIs
- **Format**: CSV daily/monthly aggregates
- **Variables**: visitor_arrivals, hotel_occupancy, festival_events
- **Seasonality**: Peak (June-August), Shoulder (May, September)

### Municipal Water Data
- **Source**: JP Vodovod Ohrid (when available)
- **Format**: CSV/Database exports
- **Variables**: flow_rates, pressure_levels, treatment_plant_status
- **Frequency**: Real-time sensors, hourly aggregates

## File Naming Convention

```
weather_ohrid_YYYY-MM-DD.csv
tourism_ohrid_YYYY-MM.csv
municipal_water_YYYY-MM-DD.csv
```

## Data Integration

External data gets processed and combined in the feature engineering pipeline:
- Raw external → data/processed → data/features → ML models