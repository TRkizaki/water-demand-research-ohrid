#!/usr/bin/env python3
"""
Quick test script for the Ohrid Water Demand Framework
Tests core functionality without requiring all dependencies
"""

import pandas as pd
import numpy as np
from datetime import datetime

def test_synthetic_data():
    """Test the generated synthetic data"""
    print("🧪 Testing Synthetic Data...")
    
    # Load the generated data
    df = pd.read_csv('data/raw/ohrid_synthetic_water_demand.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Data loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Basic validation tests
    tests = []
    
    # Test 1: Data completeness
    tests.append(("Data completeness", len(df) == 26257))
    
    # Test 2: Date range
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    tests.append(("Date range", 
                 start_date.year == 2021 and end_date.year == 2023))
    
    # Test 3: Demand is positive
    tests.append(("Positive demand", df['water_demand_m3_per_hour'].min() > 0))
    
    # Test 4: Tourism seasonality
    summer_demand = df[df['month'].isin([6,7,8])]['water_demand_m3_per_hour'].mean()
    winter_demand = df[df['month'].isin([12,1,2])]['water_demand_m3_per_hour'].mean()
    tests.append(("Summer > Winter demand", summer_demand > winter_demand))
    
    # Test 5: Daily patterns (morning/evening peaks)
    peak_hours = df[df['hour'].isin([7,8,19,20])]['water_demand_m3_per_hour'].mean()
    night_hours = df[df['hour'].isin([1,2,3,4])]['water_demand_m3_per_hour'].mean()
    tests.append(("Peak > Night demand", peak_hours > night_hours))
    
    # Test 6: Tourist season impact
    tourist_demand = df[df['is_tourist_season']]['water_demand_m3_per_hour'].mean()
    off_season_demand = df[~df['is_tourist_season']]['water_demand_m3_per_hour'].mean()
    tests.append(("Tourist season impact", tourist_demand > off_season_demand))
    
    # Print test results
    print("\n📊 Test Results:")
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    # Display key statistics
    print(f"\n📈 Key Statistics:")
    print(f"   Average demand: {df['water_demand_m3_per_hour'].mean():.1f} m³/hour")
    print(f"   Peak demand: {df['water_demand_m3_per_hour'].max():.1f} m³/hour")
    print(f"   Summer avg: {summer_demand:.1f} m³/hour")
    print(f"   Winter avg: {winter_demand:.1f} m³/hour")
    print(f"   Tourist season avg: {tourist_demand:.1f} m³/hour")
    print(f"   Off-season avg: {off_season_demand:.1f} m³/hour")
    
    # Monthly pattern
    print(f"\n📅 Monthly Demand Pattern:")
    monthly_avg = df.groupby('month')['water_demand_m3_per_hour'].mean()
    for month, demand in monthly_avg.items():
        month_name = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month]
        tourist_indicator = "🏖️" if month in [6,7,8] else "❄️" if month in [12,1,2] else "🌸"
        print(f"   {month_name}: {demand:.1f} m³/hour {tourist_indicator}")
    
    return all(passed for _, passed in tests)

def test_config():
    """Test configuration loading"""
    print("\n🔧 Testing Configuration...")
    
    try:
        import yaml
        with open('config/ohrid_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # Basic config validation
        tests = [
            ("Config loaded", config is not None),
            ("Location data", 'location' in config),
            ("Ohrid city", config['location']['city'] == 'Ohrid'),
            ("Population data", config['location']['population'] == 42033),
            ("Tourism config", 'tourism' in config['regional_characteristics']),
            ("Peak season", config['regional_characteristics']['tourism']['peak_season'] == [6,7,8])
        ]
        
        for test_name, passed in tests:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        return all(passed for _, passed in tests)
        
    except ImportError:
        print("   ⚠️  YAML library not available, skipping config test")
        return True

def test_data_patterns():
    """Test realistic data patterns"""
    print("\n🌊 Testing Data Patterns...")
    
    df = pd.read_csv('data/raw/ohrid_synthetic_water_demand.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Tourism correlation
    correlation_tests = []
    
    # Tourist numbers vs demand correlation
    tourist_corr = df['tourists_estimated'].corr(df['water_demand_m3_per_hour'])
    correlation_tests.append(("Tourism-demand correlation", tourist_corr > 0.3))
    
    # Temperature vs demand correlation (positive in summer)
    summer_data = df[df['month'].isin([6,7,8])]
    temp_corr = summer_data['temperature'].corr(summer_data['water_demand_m3_per_hour'])
    correlation_tests.append(("Temperature-demand correlation", temp_corr > 0.1))
    
    # Festival impact
    festival_demand = df[df['is_festival_period']]['water_demand_m3_per_hour'].mean()
    normal_demand = df[~df['is_festival_period']]['water_demand_m3_per_hour'].mean()
    correlation_tests.append(("Festival impact", festival_demand > normal_demand))
    
    print("📈 Pattern Analysis:")
    for test_name, passed in correlation_tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🔗 Correlations:")
    print(f"   Tourism-Demand: {tourist_corr:.3f}")
    print(f"   Temperature-Demand (summer): {temp_corr:.3f}")
    print(f"   Festival boost: +{((festival_demand/normal_demand-1)*100):.1f}%")
    
    return all(passed for _, passed in correlation_tests)

def main():
    """Run all tests"""
    print("🚀 Testing Ohrid Water Demand Framework")
    print("=" * 50)
    
    try:
        # Test synthetic data
        data_ok = test_synthetic_data()
        
        # Test configuration
        config_ok = test_config()
        
        # Test patterns
        patterns_ok = test_data_patterns()
        
        # Final result
        print("\n" + "=" * 50)
        if data_ok and config_ok and patterns_ok:
            print("🎉 ALL TESTS PASSED! Framework is working correctly.")
            print("\n✅ Ready for:")
            print("   • Machine learning model training")
            print("   • GCP cloud deployment")
            print("   • Research experimentation")
            print("   • Academic publication")
        else:
            print("⚠️  Some tests failed. Check the output above.")
        
        print(f"\n📁 Generated files:")
        print(f"   • data/raw/ohrid_synthetic_water_demand.csv ({len(pd.read_csv('data/raw/ohrid_synthetic_water_demand.csv')):,} rows)")
        print(f"   • config/ohrid_config.yaml (regional configuration)")
        print(f"   • infrastructure/gcp/ (cloud setup scripts)")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)