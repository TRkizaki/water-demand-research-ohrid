#!/usr/bin/env python3
"""
Test Real Data Collection for Ohrid Water Demand Research

This script demonstrates the real data collection capabilities
and compares them with synthetic data generation.
"""

import os
import sys
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append('src')

from data_collectors.ohrid_data_manager import OhridDataManager
from data_collectors.ohrid_real_data_collector import OhridRealDataCollector
from data_collectors.ohrid_synthetic_generator import OhridWaterDemandGenerator


def test_real_data_collection():
    """Test real data collection functionality."""
    print("=" * 60)
    print("OHRID WATER DEMAND - REAL DATA COLLECTION TEST")
    print("=" * 60)
    
    # Check environment variables
    print("\n1. ENVIRONMENT SETUP")
    print("-" * 30)
    
    weather_api_key = os.getenv('OPENWEATHER_API_KEY')
    if weather_api_key:
        print(f"✅ OpenWeatherMap API Key: {'*' * 8}{weather_api_key[-4:]}")
    else:
        print("⚠️  OpenWeatherMap API Key: Not set")
        print("   Set with: export OPENWEATHER_API_KEY='your_api_key'")
        print("   Get free key at: https://openweathermap.org/api")
    
    # Test real data collector
    print("\n2. REAL DATA COLLECTOR TEST")
    print("-" * 30)
    
    collector = OhridRealDataCollector()
    
    # Test weather data
    print("Weather Data:")
    weather = collector.fetch_current_weather()
    if weather:
        print(f"✅ Temperature: {weather.temperature}°C")
        print(f"✅ Humidity: {weather.humidity}%")
        print(f"✅ Precipitation: {weather.precipitation} mm/h")
        print(f"✅ Description: {weather.description}")
    else:
        print("❌ Weather data collection failed")
    
    # Test tourism estimation
    print("\nTourism Estimation:")
    tourism = collector.estimate_current_tourism()
    print(f"✅ Estimated Tourists: {tourism.estimated_tourists:,}")
    print(f"✅ Hotel Occupancy: {tourism.hotel_occupancy_rate:.1%}")
    print(f"✅ Festival Period: {tourism.is_festival_period}")
    
    # Test comprehensive data collection
    print("\n3. COMPREHENSIVE DATA COLLECTION")
    print("-" * 30)
    
    real_data = collector.collect_real_time_data()
    quality = collector.validate_data_quality(real_data)
    features = collector.create_prediction_features(real_data)
    
    print(f"Data Quality Score: {quality['overall_score']}/100")
    print(f"Recommendation: {quality['recommendation']}")
    print(f"Available Features: {len(features)}")
    
    # Test data manager (hybrid system)
    print("\n4. HYBRID DATA MANAGER TEST")
    print("-" * 30)
    
    manager = OhridDataManager(prefer_real_data=True, quality_threshold=40)
    
    # API status
    api_status = manager.validate_api_connections()
    ready_for_real = api_status['overall']['ready_for_real_data']
    connected_apis = api_status['overall']['connected_apis']
    
    print(f"Connected APIs: {connected_apis}/3")
    print(f"Ready for Real Data: {'✅' if ready_for_real else '❌'}")
    
    # Get current data (will choose best source)
    current_data = manager.get_current_data()
    print(f"Selected Data Source: {current_data['data_source']}")
    print(f"Data Quality: {current_data['quality_score']}/100")
    
    # Test prediction dataset
    print("\n5. PREDICTION DATASET GENERATION")
    print("-" * 30)
    
    dataset = manager.get_prediction_dataset(hours_back=24, include_current=True)
    print(f"Dataset Size: {len(dataset)} records")
    print(f"Date Range: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}")
    
    # Show feature comparison
    print("\n6. FEATURE COMPARISON")
    print("-" * 30)
    
    current_features = current_data['features']
    print(f"Available features for prediction:")
    for feature, value in list(current_features.items())[:10]:  # Show first 10
        if isinstance(value, float):
            print(f"  {feature}: {value:.2f}")
        else:
            print(f"  {feature}: {value}")
    
    if len(current_features) > 10:
        print(f"  ... and {len(current_features) - 10} more features")
    
    # Generate quality report
    print("\n7. DATA QUALITY REPORT")
    print("-" * 30)
    
    quality_report = manager.get_data_quality_report()
    print(f"Overall Assessment: {quality_report['overall_recommendation']}")
    
    for recommendation in quality_report['recommendations']:
        print(f"• {recommendation}")
    
    # Compare with synthetic data
    print("\n8. SYNTHETIC vs REAL DATA COMPARISON")
    print("-" * 30)
    
    # Generate one hour of synthetic data for comparison
    synthetic_gen = OhridWaterDemandGenerator()
    now = datetime.now()
    synthetic_df = synthetic_gen.generate_synthetic_data(
        start_date=now.strftime("%Y-%m-%d"),
        end_date=now.strftime("%Y-%m-%d"),
        frequency="1h"
    )
    
    if len(synthetic_df) > 0:
        synthetic_row = synthetic_df.iloc[0]
        
        print("Temperature Comparison:")
        if weather:
            print(f"  Real: {weather.temperature}°C")
            print(f"  Synthetic: {synthetic_row['temperature']}°C")
            diff = abs(weather.temperature - synthetic_row['temperature'])
            print(f"  Difference: {diff:.1f}°C")
        
        print("Tourism Comparison:")
        print(f"  Real (estimated): {tourism.estimated_tourists:,}")
        print(f"  Synthetic: {synthetic_row['tourists_estimated']:,}")
    
    print("\n" + "=" * 60)
    print("REAL DATA COLLECTION TEST COMPLETED")
    print("=" * 60)
    
    # Return status for programmatic use
    return {
        'weather_available': weather is not None,
        'tourism_available': True,
        'overall_quality': quality['overall_score'],
        'ready_for_production': ready_for_real,
        'recommended_source': current_data['data_source']
    }


def get_setup_instructions():
    """Provide setup instructions for real data collection."""
    print("\n" + "=" * 60)
    print("SETUP INSTRUCTIONS FOR REAL DATA COLLECTION")
    print("=" * 60)
    
    print("\n1. Weather Data (OpenWeatherMap):")
    print("   • Sign up: https://openweathermap.org/api")
    print("   • Get free API key (60 calls/minute)")
    print("   • Set environment variable:")
    print("     export OPENWEATHER_API_KEY='your_api_key_here'")
    
    print("\n2. Tourism Data:")
    print("   • Currently using estimation algorithms")
    print("   • Future: Integrate with booking APIs")
    print("   • Consider: Municipal tourism office data")
    
    print("\n3. Water System Data:")
    print("   • Requires partnership with JP Vodovod Ohrid")
    print("   • SCADA system integration needed")
    print("   • IoT sensor deployment possible")
    
    print("\n4. Usage in Your Code:")
    print("   from src.data_collectors.ohrid_data_manager import OhridDataManager")
    print("   manager = OhridDataManager(prefer_real_data=True)")
    print("   current_data = manager.get_current_data()")
    
    print("\n5. Fallback Strategy:")
    print("   • System automatically falls back to synthetic data")
    print("   • Quality threshold configurable (default: 50/100)")
    print("   • Hybrid approach ensures continuous operation")


if __name__ == "__main__":
    # Run the test
    test_results = test_real_data_collection()
    
    # Show setup instructions if needed
    if not test_results['weather_available']:
        get_setup_instructions()
    
    # Exit with appropriate code
    if test_results['ready_for_production']:
        print("\n🎉 System ready for real data collection!")
        sys.exit(0)
    else:
        print("\n⚠️  System not ready for production. Continue with synthetic data.")
        sys.exit(1)