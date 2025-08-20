"""
Test Suite for Data Collection Components

Tests synthetic data generation, real data collection,
and hybrid data management functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collectors.ohrid_synthetic_generator import OhridWaterDemandGenerator
from data_collectors.ohrid_real_data_collector import OhridRealDataCollector
from data_collectors.ohrid_data_manager import OhridDataManager


class TestSyntheticDataGeneration:
    """Test synthetic data generation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = OhridWaterDemandGenerator()
        
    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        assert self.generator is not None
        assert self.generator.location['city'] == 'Ohrid'
        assert self.generator.location['population'] == 42033
        
    def test_synthetic_data_generation(self):
        """Test synthetic data generation produces valid data."""
        # Generate small dataset for testing
        data = self.generator.generate_synthetic_data(
            start_date="2023-01-01",
            end_date="2023-01-02",
            frequency="1h"
        )
        
        # Basic structure tests
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 25  # 25 hours (including start hour)
        assert 'water_demand_m3_per_hour' in data.columns
        assert data.index.name == 'timestamp'
        
        # Data quality tests
        assert data['water_demand_m3_per_hour'].min() > 0
        assert data['water_demand_m3_per_hour'].max() < 1000  # Reasonable upper bound
        assert not data['water_demand_m3_per_hour'].isna().any()
        
        # Temporal features tests
        assert 'hour' in data.columns
        assert 'day_of_week' in data.columns
        assert 'month' in data.columns
        assert data['hour'].min() >= 0
        assert data['hour'].max() <= 23
        
    def test_tourism_estimation(self):
        """Test tourism estimation logic."""
        # Test summer peak
        summer_date = datetime(2023, 7, 15)  # Mid-July
        tourists_summer = self.generator._estimate_tourist_numbers(summer_date)
        
        # Test winter low season
        winter_date = datetime(2023, 1, 15)  # Mid-January
        tourists_winter = self.generator._estimate_tourist_numbers(winter_date)
        
        # Summer should have more tourists
        assert tourists_summer > tourists_winter
        assert tourists_summer > 2000  # Peak season minimum
        assert tourists_winter < 1500   # Off-season maximum
        
    def test_weather_data_generation(self):
        """Test weather data generation."""
        start_date = datetime(2023, 6, 1)
        end_date = datetime(2023, 6, 2)
        
        weather_df = self.generator._generate_weather_data(start_date, end_date)
        
        # Structure tests
        assert isinstance(weather_df, pd.DataFrame)
        assert len(weather_df) == 25  # 25 hours
        
        # Weather variables tests
        required_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
        for col in required_cols:
            assert col in weather_df.columns
            assert not weather_df[col].isna().any()
        
        # Reasonable ranges
        assert weather_df['temperature'].min() >= -10
        assert weather_df['temperature'].max() <= 45
        assert weather_df['humidity'].min() >= 0
        assert weather_df['humidity'].max() <= 100


class TestRealDataCollection:
    """Test real data collection functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.collector = OhridRealDataCollector()
        
    def test_collector_initialization(self):
        """Test collector initializes correctly."""
        assert self.collector is not None
        assert self.collector.location['coordinates']['latitude'] == 41.1175
        assert self.collector.location['coordinates']['longitude'] == 20.8016
        
    def test_tourism_estimation(self):
        """Test tourism estimation without API calls."""
        tourism_data = self.collector.estimate_current_tourism()
        
        # Structure tests
        assert tourism_data.estimated_tourists > 0
        assert 0 <= tourism_data.hotel_occupancy_rate <= 1
        assert isinstance(tourism_data.is_festival_period, bool)
        assert tourism_data.booking_index >= 0
        
        # Seasonal logic test
        now = datetime.now()
        if now.month in [7, 8]:  # Summer
            assert tourism_data.estimated_tourists > 2000
        else:
            assert tourism_data.estimated_tourists <= 3000
            
    def test_data_quality_validation(self):
        """Test data quality scoring system."""
        # Collect sample data
        real_data = self.collector.collect_real_time_data()
        quality = self.collector.validate_data_quality(real_data)
        
        # Quality report structure
        assert 'overall_score' in quality
        assert 'recommendation' in quality
        assert isinstance(quality['overall_score'], (int, float))
        assert 0 <= quality['overall_score'] <= 100
        
        # Quality components
        assert 'weather' in quality
        assert 'tourism' in quality
        assert 'water_system' in quality
        
    def test_feature_creation(self):
        """Test ML feature creation from real data."""
        real_data = self.collector.collect_real_time_data()
        features = self.collector.create_prediction_features(real_data)
        
        # Required features
        required_features = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'temperature', 'humidity', 'precipitation',
            'tourists_estimated', 'population'
        ]
        
        for feature in required_features:
            assert feature in features
            assert features[feature] is not None


class TestHybridDataManager:
    """Test hybrid data management system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.manager = OhridDataManager(
            prefer_real_data=True, 
            quality_threshold=50
        )
        
    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        assert self.manager is not None
        assert self.manager.prefer_real_data is True
        assert self.manager.quality_threshold == 50
        
    def test_current_data_retrieval(self):
        """Test current data retrieval with fallback."""
        current_data = self.manager.get_current_data()
        
        # Structure tests
        assert 'features' in current_data
        assert 'data_source' in current_data
        assert 'quality_score' in current_data
        assert 'timestamp' in current_data
        
        # Data source validation
        assert current_data['data_source'] in ['real', 'synthetic']
        assert isinstance(current_data['quality_score'], (int, float))
        assert 0 <= current_data['quality_score'] <= 100
        
        # Features validation
        features = current_data['features']
        assert len(features) > 10  # Should have multiple features
        assert 'hour' in features
        assert 'temperature' in features
        assert 'tourists_estimated' in features
        
    def test_historical_data_generation(self):
        """Test historical data generation."""
        # Test synthetic historical data
        historical_data = self.manager.get_historical_data(
            start_date="2023-01-01",
            end_date="2023-01-02",
            data_source="synthetic"
        )
        
        assert isinstance(historical_data, pd.DataFrame)
        assert len(historical_data) > 0
        assert 'water_demand_m3_per_hour' in historical_data.columns
        
    def test_prediction_dataset_creation(self):
        """Test prediction dataset creation."""
        dataset = self.manager.get_prediction_dataset(
            hours_back=24,
            include_current=True
        )
        
        assert isinstance(dataset, pd.DataFrame)
        assert len(dataset) >= 24  # At least 24 hours + current
        assert 'timestamp' in dataset.columns or dataset.index.name == 'timestamp'
        
    def test_api_validation(self):
        """Test API connection validation."""
        status = self.manager.validate_api_connections()
        
        # Structure tests
        assert 'timestamp' in status
        assert 'apis' in status
        assert 'overall' in status
        
        # API status tests
        apis = status['apis']
        assert 'weather' in apis
        assert 'tourism' in apis
        assert 'water_system' in apis
        
        for api_name, api_status in apis.items():
            assert 'status' in api_status
            assert 'service' in api_status
            assert api_status['status'] in ['connected', 'failed', 'error', 'not_implemented']
            
    def test_quality_reporting(self):
        """Test data quality reporting."""
        quality_report = self.manager.get_data_quality_report()
        
        # Structure tests
        assert 'timestamp' in quality_report
        assert 'api_status' in quality_report
        assert 'recommendations' in quality_report
        assert 'overall_recommendation' in quality_report
        
        # Content validation
        assert isinstance(quality_report['recommendations'], list)
        assert isinstance(quality_report['overall_recommendation'], str)


class TestDataIntegration:
    """Test data integration and consistency."""
    
    def test_synthetic_vs_real_feature_compatibility(self):
        """Test that synthetic and real data have compatible features."""
        # Generate synthetic data
        generator = OhridWaterDemandGenerator()
        synthetic_data = generator.generate_synthetic_data(
            start_date="2023-01-01",
            end_date="2023-01-01",
            frequency="1h"
        )
        
        # Get real data features
        collector = OhridRealDataCollector()
        real_data = collector.collect_real_time_data()
        real_features = collector.create_prediction_features(real_data)
        
        # Check feature compatibility
        synthetic_features = set(synthetic_data.columns)
        real_features_set = set(real_features.keys())
        
        # Common core features should exist in both
        core_features = {
            'hour', 'day_of_week', 'month', 'is_weekend',
            'temperature', 'humidity', 'precipitation',
            'tourists_estimated', 'population'
        }
        
        assert core_features.issubset(synthetic_features)
        assert core_features.issubset(real_features_set)
        
    def test_data_manager_fallback_consistency(self):
        """Test that data manager provides consistent output regardless of source."""
        manager = OhridDataManager(prefer_real_data=True, quality_threshold=50)
        
        # Get data multiple times
        data1 = manager.get_current_data()
        data2 = manager.get_current_data()
        
        # Structure should be consistent
        assert set(data1.keys()) == set(data2.keys())
        assert set(data1['features'].keys()) == set(data2['features'].keys())
        
        # Both should be valid
        for data in [data1, data2]:
            assert data['data_source'] in ['real', 'synthetic']
            assert 0 <= data['quality_score'] <= 100
            assert len(data['features']) > 5


def run_all_tests():
    """Run all tests with pytest."""
    test_classes = [
        TestSyntheticDataGeneration,
        TestRealDataCollection, 
        TestHybridDataManager,
        TestDataIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        
        # Get test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                # Create instance and run test
                instance = test_class()
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                
                getattr(instance, test_method)()
                print(f"  ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method}: {e}")
    
    print(f"\nTest Results: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)