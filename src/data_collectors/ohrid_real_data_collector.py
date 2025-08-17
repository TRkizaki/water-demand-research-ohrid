"""
Ohrid Water Demand Real Data Collector

This module collects real-world data for Ohrid water demand prediction:
- Weather data from meteorological APIs
- Tourism estimates from various sources
- Municipal water system data (when available)
"""

import pandas as pd
import numpy as np
import requests
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WeatherData:
    """Weather data structure"""
    timestamp: datetime
    temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    pressure: float
    cloud_cover: int
    description: str


@dataclass
class TourismData:
    """Tourism data structure"""
    timestamp: datetime
    estimated_tourists: int
    hotel_occupancy_rate: float
    is_festival_period: bool
    booking_index: float


class OhridRealDataCollector:
    """
    Collect real-world data for Ohrid water demand prediction.
    Integrates multiple data sources with fallback mechanisms.
    """
    
    def __init__(self, config_path: str = "config/ohrid_config.yaml"):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.location = self.config['location']
        
        # API keys from environment variables
        self.weather_api_key = os.getenv('OPENWEATHER_API_KEY')
        self.weather_backup_key = os.getenv('WORLDWEATHER_API_KEY')
        
        # API endpoints
        self.weather_endpoints = {
            'openweather': 'https://api.openweathermap.org/data/2.5',
            'weatherapi': 'http://api.weatherapi.com/v1'
        }
        
        # Tourism estimation parameters
        self.tourism_baseline = 1000  # Daily baseline tourists
        
    def fetch_current_weather(self) -> Optional[WeatherData]:
        """Fetch current weather data from OpenWeatherMap API."""
        if not self.weather_api_key:
            logger.warning("OpenWeatherMap API key not found. Please set OPENWEATHER_API_KEY environment variable.")
            return None
            
        try:
            # Current weather endpoint
            url = f"{self.weather_endpoints['openweather']}/weather"
            params = {
                'lat': self.location['coordinates']['latitude'],
                'lon': self.location['coordinates']['longitude'],
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return WeatherData(
                timestamp=datetime.now(),
                temperature=data['main']['temp'],
                humidity=data['main']['humidity'],
                precipitation=data.get('rain', {}).get('1h', 0.0),
                wind_speed=data['wind']['speed'],
                pressure=data['main']['pressure'],
                cloud_cover=data['clouds']['all'],
                description=data['weather'][0]['description']
            )
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch weather data: {e}")
            return None
        except KeyError as e:
            logger.error(f"Unexpected weather API response format: {e}")
            return None
    
    def fetch_historical_weather(self, start_date: datetime, end_date: datetime) -> List[WeatherData]:
        """
        Fetch historical weather data.
        Note: OpenWeatherMap historical data requires paid subscription.
        This is a placeholder for the implementation.
        """
        logger.warning("Historical weather data requires paid API subscription.")
        logger.info("Consider using alternative sources or synthetic data for historical analysis.")
        
        # For demonstration, return empty list
        # In production, implement paid historical weather API
        return []
    
    def estimate_current_tourism(self) -> TourismData:
        """
        Estimate current tourism levels using multiple indicators.
        This uses heuristic methods since direct tourism APIs are limited.
        """
        now = datetime.now()
        
        # Seasonal tourism estimation
        month = now.month
        base_tourists = self.tourism_baseline
        
        # Seasonal adjustments (based on historical patterns)
        seasonal_multipliers = {
            1: 0.4, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.5,
            6: 2.5, 7: 3.0, 8: 3.0, 9: 1.8, 10: 1.0,
            11: 0.6, 12: 0.5
        }
        
        estimated_tourists = int(base_tourists * seasonal_multipliers.get(month, 1.0))
        
        # Weekend adjustment
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            estimated_tourists = int(estimated_tourists * 1.2)
        
        # Festival period check (Ohrid Summer Festival: July-August)
        is_festival = month in [7, 8]
        if is_festival:
            estimated_tourists = int(estimated_tourists * 1.4)
        
        # Estimate hotel occupancy (simplified model)
        max_capacity = 5000  # Estimated hotel capacity in Ohrid
        occupancy_rate = min(0.95, estimated_tourists / max_capacity)
        
        # Booking index (placeholder - could integrate with booking APIs)
        booking_index = occupancy_rate * 100
        
        return TourismData(
            timestamp=now,
            estimated_tourists=estimated_tourists,
            hotel_occupancy_rate=occupancy_rate,
            is_festival_period=is_festival,
            booking_index=booking_index
        )
    
    def fetch_water_system_data(self) -> Optional[Dict]:
        """
        Fetch real water system data from municipal SCADA/IoT systems.
        This requires integration with local water utility systems.
        """
        logger.info("Water system integration not implemented.")
        logger.info("This requires partnership with local water utility (JP Vodovod Ohrid).")
        
        # Placeholder for municipal water system integration
        # In production, this would connect to:
        # - SCADA systems
        # - Smart water meters
        # - Pressure/flow sensors
        # - Treatment plant data
        
        return None
    
    def collect_real_time_data(self) -> Dict:
        """Collect all available real-time data sources."""
        logger.info("Collecting real-time data for Ohrid...")
        
        # Collect weather data
        weather = self.fetch_current_weather()
        
        # Estimate tourism
        tourism = self.estimate_current_tourism()
        
        # Try to fetch water system data
        water_system = self.fetch_water_system_data()
        
        current_data = {
            'timestamp': datetime.now(),
            'weather': weather,
            'tourism': tourism,
            'water_system': water_system,
            'data_sources': {
                'weather_available': weather is not None,
                'tourism_estimated': True,
                'water_system_available': water_system is not None
            }
        }
        
        return current_data
    
    def create_prediction_features(self, real_data: Dict) -> Dict:
        """Convert real data into features for ML prediction."""
        now = real_data['timestamp']
        weather = real_data['weather']
        tourism = real_data['tourism']
        
        features = {
            # Temporal features
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'month': now.month,
            'is_weekend': now.weekday() >= 5,
            
            # Weather features (if available)
            'temperature': weather.temperature if weather else 20.0,
            'humidity': weather.humidity if weather else 65.0,
            'precipitation': weather.precipitation if weather else 0.0,
            'wind_speed': weather.wind_speed if weather else 2.0,
            'pressure': weather.pressure if weather else 1013.0,
            'cloud_cover': weather.cloud_cover if weather else 50,
            
            # Tourism features
            'tourists_estimated': tourism.estimated_tourists,
            'hotel_occupancy_rate': tourism.hotel_occupancy_rate,
            'is_festival_period': tourism.is_festival_period,
            'booking_index': tourism.booking_index,
            
            # Population (static)
            'population': self.location['population']
        }
        
        return features
    
    def validate_data_quality(self, data: Dict) -> Dict:
        """Validate and score data quality."""
        quality_score = 0
        quality_report = {}
        
        # Weather data quality
        if data['weather']:
            quality_score += 40
            quality_report['weather'] = 'real_api_data'
        else:
            quality_report['weather'] = 'missing_fallback_needed'
        
        # Tourism data quality
        quality_score += 30  # Always available (estimated)
        quality_report['tourism'] = 'estimated_from_patterns'
        
        # Water system data quality
        if data['water_system']:
            quality_score += 30
            quality_report['water_system'] = 'real_sensor_data'
        else:
            quality_report['water_system'] = 'not_available'
        
        quality_report['overall_score'] = quality_score
        quality_report['recommendation'] = self._get_quality_recommendation(quality_score)
        
        return quality_report
    
    def _get_quality_recommendation(self, score: int) -> str:
        """Get recommendation based on data quality score."""
        if score >= 80:
            return "Excellent data quality. Proceed with real-time prediction."
        elif score >= 60:
            return "Good data quality. Minor fallbacks may be needed."
        elif score >= 40:
            return "Moderate data quality. Consider hybrid approach with synthetic data."
        else:
            return "Poor data quality. Use synthetic data or wait for better conditions."


def main():
    """Demonstrate real data collection."""
    collector = OhridRealDataCollector()
    
    print("Ohrid Real Data Collection Demo")
    print("=" * 50)
    
    # Collect real-time data
    real_data = collector.collect_real_time_data()
    
    # Validate data quality
    quality = collector.validate_data_quality(real_data)
    
    # Create prediction features
    features = collector.create_prediction_features(real_data)
    
    # Display results
    print(f"\nData Collection Results:")
    print(f"Timestamp: {real_data['timestamp']}")
    print(f"Quality Score: {quality['overall_score']}/100")
    print(f"Recommendation: {quality['recommendation']}")
    
    if real_data['weather']:
        weather = real_data['weather']
        print(f"\nWeather Data:")
        print(f"  Temperature: {weather.temperature}°C")
        print(f"  Humidity: {weather.humidity}%")
        print(f"  Precipitation: {weather.precipitation} mm/h")
    
    tourism = real_data['tourism']
    print(f"\nTourism Estimates:")
    print(f"  Estimated Tourists: {tourism.estimated_tourists:,}")
    print(f"  Hotel Occupancy: {tourism.hotel_occupancy_rate:.1%}")
    print(f"  Festival Period: {tourism.is_festival_period}")
    
    print(f"\nML Features Ready: {len(features)} features available")


if __name__ == "__main__":
    main()