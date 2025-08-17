"""
Ohrid Water Demand Data Manager

Hybrid system that manages both synthetic and real data collection
for the Ohrid water demand research project.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
import os

from .ohrid_synthetic_generator import OhridWaterDemandGenerator
from .ohrid_real_data_collector import OhridRealDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OhridDataManager:
    """
    Unified data management system for Ohrid water demand research.
    Handles both synthetic and real data collection with intelligent fallbacks.
    """
    
    def __init__(self, 
                 config_path: str = "config/ohrid_config.yaml",
                 prefer_real_data: bool = True,
                 quality_threshold: int = 50):
        """
        Initialize the data manager.
        
        Args:
            config_path: Path to configuration file
            prefer_real_data: Whether to prefer real data over synthetic
            quality_threshold: Minimum quality score to use real data
        """
        self.config_path = config_path
        self.prefer_real_data = prefer_real_data
        self.quality_threshold = quality_threshold
        
        # Initialize data collectors
        self.synthetic_generator = OhridWaterDemandGenerator(config_path)
        self.real_collector = OhridRealDataCollector(config_path)
        
        logger.info(f"Data Manager initialized (prefer_real: {prefer_real_data})")
    
    def get_current_data(self) -> Dict:
        """
        Get current water demand prediction data.
        Uses real data if available and quality is sufficient, otherwise synthetic.
        """
        if self.prefer_real_data:
            # Try real data first
            try:
                real_data = self.real_collector.collect_real_time_data()
                quality = self.real_collector.validate_data_quality(real_data)
                
                if quality['overall_score'] >= self.quality_threshold:
                    logger.info(f"Using real data (quality: {quality['overall_score']}/100)")
                    features = self.real_collector.create_prediction_features(real_data)
                    
                    return {
                        'features': features,
                        'data_source': 'real',
                        'quality_score': quality['overall_score'],
                        'recommendation': quality['recommendation'],
                        'timestamp': real_data['timestamp']
                    }
                else:
                    logger.warning(f"Real data quality insufficient ({quality['overall_score']}/100). Falling back to synthetic.")
            
            except Exception as e:
                logger.error(f"Real data collection failed: {e}. Falling back to synthetic.")
        
        # Fall back to synthetic data
        return self._get_synthetic_current_data()
    
    def _get_synthetic_current_data(self) -> Dict:
        """Generate synthetic data for current timestamp."""
        now = datetime.now()
        
        # Generate one hour of synthetic data for current time
        synthetic_df = self.synthetic_generator.generate_synthetic_data(
            start_date=now.strftime("%Y-%m-%d"),
            end_date=now.strftime("%Y-%m-%d"),
            frequency="1h"
        )
        
        if len(synthetic_df) > 0:
            row = synthetic_df.iloc[0]
            features = {
                'hour': row['hour'],
                'day_of_week': row['day_of_week'],
                'month': row['month'],
                'is_weekend': row['is_weekend'],
                'temperature': row['temperature'],
                'humidity': row['humidity'],
                'precipitation': row['precipitation'],
                'wind_speed': row['wind_speed'],
                'pressure': row['pressure'],
                'cloud_cover': row['cloud_cover'],
                'tourists_estimated': row['tourists_estimated'],
                'hotel_occupancy_rate': row.get('hotel_occupancy_rate', 0.6),
                'is_festival_period': row['is_festival_period'],
                'booking_index': row['tourists_estimated'] / 50,  # Derived feature
                'population': row['population']
            }
            
            logger.info("Using synthetic data for current prediction")
            return {
                'features': features,
                'data_source': 'synthetic',
                'quality_score': 100,  # Synthetic data is always "perfect"
                'recommendation': 'Synthetic data - ideal for research and testing',
                'timestamp': now
            }
        
        else:
            raise ValueError("Failed to generate synthetic data")
    
    def get_historical_data(self, 
                          start_date: str, 
                          end_date: str,
                          data_source: str = "auto") -> pd.DataFrame:
        """
        Get historical data for model training.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            data_source: "real", "synthetic", or "auto"
        """
        if data_source == "real":
            logger.warning("Historical real data collection not fully implemented.")
            logger.info("Using synthetic data for historical analysis.")
            data_source = "synthetic"
        
        if data_source == "synthetic" or data_source == "auto":
            logger.info(f"Generating synthetic historical data: {start_date} to {end_date}")
            return self.synthetic_generator.generate_synthetic_data(
                start_date=start_date,
                end_date=end_date,
                frequency="1h"
            )
        
        else:
            raise ValueError(f"Unknown data source: {data_source}")
    
    def get_prediction_dataset(self, 
                             hours_back: int = 168,  # 1 week
                             include_current: bool = True) -> pd.DataFrame:
        """
        Get dataset suitable for immediate prediction.
        Combines recent historical data with current conditions.
        
        Args:
            hours_back: How many hours of historical data to include
            include_current: Whether to include current hour data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Get historical data (synthetic for now)
        historical_df = self.get_historical_data(
            start_date=start_time.strftime("%Y-%m-%d"),
            end_date=end_time.strftime("%Y-%m-%d"),
            data_source="synthetic"
        )
        
        if include_current:
            # Add current real-time data point
            current_data = self.get_current_data()
            current_features = current_data['features']
            
            # Convert to DataFrame row
            current_row = pd.DataFrame([{
                'timestamp': current_data['timestamp'],
                'water_demand_m3_per_hour': np.nan,  # To be predicted
                **current_features
            }])
            
            # Combine historical and current
            full_dataset = pd.concat([historical_df, current_row], ignore_index=True)
            
            # Add metadata
            full_dataset.attrs['data_sources'] = {
                'historical': 'synthetic',
                'current': current_data['data_source'],
                'current_quality': current_data['quality_score']
            }
            
            return full_dataset
        
        return historical_df
    
    def validate_api_connections(self) -> Dict:
        """Test all external API connections and return status."""
        status = {
            'timestamp': datetime.now(),
            'apis': {}
        }
        
        # Test weather API
        try:
            weather_data = self.real_collector.fetch_current_weather()
            status['apis']['weather'] = {
                'status': 'connected' if weather_data else 'failed',
                'service': 'OpenWeatherMap',
                'data_available': weather_data is not None
            }
        except Exception as e:
            status['apis']['weather'] = {
                'status': 'error',
                'service': 'OpenWeatherMap',
                'error': str(e)
            }
        
        # Tourism estimation (always available)
        status['apis']['tourism'] = {
            'status': 'connected',
            'service': 'Internal Estimation',
            'data_available': True
        }
        
        # Water system integration
        status['apis']['water_system'] = {
            'status': 'not_implemented',
            'service': 'Municipal SCADA',
            'data_available': False
        }
        
        # Overall status
        connected_apis = sum(1 for api in status['apis'].values() 
                           if api['status'] == 'connected')
        status['overall'] = {
            'connected_apis': connected_apis,
            'total_apis': len(status['apis']),
            'ready_for_real_data': connected_apis >= 2
        }
        
        return status
    
    def get_data_quality_report(self) -> Dict:
        """Generate comprehensive data quality report."""
        report = {
            'timestamp': datetime.now(),
            'api_status': self.validate_api_connections(),
            'recommendations': []
        }
        
        # Check real data availability
        try:
            real_data = self.real_collector.collect_real_time_data()
            real_quality = self.real_collector.validate_data_quality(real_data)
            
            report['real_data'] = {
                'available': True,
                'quality_score': real_quality['overall_score'],
                'recommendation': real_quality['recommendation']
            }
            
            if real_quality['overall_score'] >= self.quality_threshold:
                report['recommendations'].append("Real data quality is sufficient for production use")
            else:
                report['recommendations'].append("Consider hybrid approach or improve data sources")
                
        except Exception as e:
            report['real_data'] = {
                'available': False,
                'error': str(e)
            }
            report['recommendations'].append("Real data collection failed - use synthetic data")
        
        # Synthetic data (always available)
        report['synthetic_data'] = {
            'available': True,
            'quality_score': 100,
            'recommendation': "Excellent for research and development"
        }
        
        # Overall recommendation
        if report.get('real_data', {}).get('available'):
            if report['real_data']['quality_score'] >= 80:
                report['overall_recommendation'] = "Ready for production deployment with real data"
            elif report['real_data']['quality_score'] >= 50:
                report['overall_recommendation'] = "Suitable for testing with real data validation"
            else:
                report['overall_recommendation'] = "Continue development with synthetic data"
        else:
            report['overall_recommendation'] = "Use synthetic data for development and research"
        
        return report


def main():
    """Demonstrate hybrid data management."""
    print("Ohrid Water Demand Data Manager Demo")
    print("=" * 50)
    
    # Initialize data manager
    manager = OhridDataManager(prefer_real_data=True, quality_threshold=40)
    
    # Test API connections
    print("\n1. API Connection Status:")
    api_status = manager.validate_api_connections()
    for api_name, status in api_status['apis'].items():
        print(f"   {api_name}: {status['status']} ({status['service']})")
    
    # Get current data
    print("\n2. Current Data Collection:")
    current = manager.get_current_data()
    print(f"   Data Source: {current['data_source']}")
    print(f"   Quality Score: {current['quality_score']}/100")
    print(f"   Features Available: {len(current['features'])}")
    
    # Get prediction dataset
    print("\n3. Prediction Dataset:")
    dataset = manager.get_prediction_dataset(hours_back=24)
    print(f"   Records: {len(dataset)}")
    print(f"   Date Range: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}")
    
    # Data quality report
    print("\n4. Data Quality Report:")
    quality_report = manager.get_data_quality_report()
    print(f"   Overall Recommendation: {quality_report['overall_recommendation']}")
    
    print(f"\n✅ Data Manager successfully demonstrated!")


if __name__ == "__main__":
    main()