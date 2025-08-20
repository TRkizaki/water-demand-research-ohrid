"""
Test Runner for Ohrid Water Demand Research Framework

Comprehensive test suite runner that executes all tests
and provides detailed reporting.
"""

import sys
import os
import importlib
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import test modules
from test_data_collection import run_all_tests as run_data_tests
from test_models import run_model_tests


def run_comprehensive_test_suite():
    """Run all tests in the framework."""
    print("=" * 60)
    print("OHRID WATER DEMAND RESEARCH - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test results tracking
    test_results = {}
    
    # 1. Data Collection Tests
    print("1. DATA COLLECTION TESTS")
    print("-" * 30)
    try:
        data_success = run_data_tests()
        test_results['data_collection'] = data_success
        print(f"Data Collection Tests: {'PASSED' if data_success else 'FAILED'}")
    except Exception as e:
        print(f"Data Collection Tests: FAILED - {e}")
        test_results['data_collection'] = False
    
    print()
    
    # 2. Model Tests
    print("2. MODEL TESTS")
    print("-" * 30)
    try:
        model_success = run_model_tests()
        test_results['models'] = model_success
        print(f"Model Tests: {'PASSED' if model_success else 'FAILED'}")
    except Exception as e:
        print(f"Model Tests: FAILED - {e}")
        test_results['models'] = False
    
    print()
    
    # 3. Integration Tests
    print("3. INTEGRATION TESTS")
    print("-" * 30)
    try:
        integration_success = run_integration_tests()
        test_results['integration'] = integration_success
        print(f"Integration Tests: {'PASSED' if integration_success else 'FAILED'}")
    except Exception as e:
        print(f"Integration Tests: FAILED - {e}")
        test_results['integration'] = False
    
    print()
    
    # 4. GCP Tests (if credentials available)
    print("4. GCP INTEGRATION TESTS")
    print("-" * 30)
    try:
        gcp_success = run_gcp_tests()
        test_results['gcp'] = gcp_success
        print(f"GCP Tests: {'PASSED' if gcp_success else 'SKIPPED'}")
    except Exception as e:
        print(f"GCP Tests: SKIPPED - {e}")
        test_results['gcp'] = None
    
    print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_suites = len([k for k, v in test_results.items() if v is not None])
    passed_suites = len([k for k, v in test_results.items() if v is True])
    
    for suite_name, result in test_results.items():
        if result is True:
            status = "PASSED"
        elif result is False:
            status = "FAILED"
        else:
            status = "SKIPPED"
        print(f"{suite_name.replace('_', ' ').title()}: {status}")
    
    print()
    print(f"Overall Result: {passed_suites}/{total_suites} test suites passed")
    
    if passed_suites == total_suites:
        print("🎉 ALL TESTS PASSED - Framework ready for deployment!")
        return True
    else:
        print("⚠️  Some tests failed - Review failures before deployment")
        return False


def run_integration_tests():
    """Run integration tests between components."""
    print("Running Integration Tests...")
    
    try:
        # Test data flow: generation -> features -> models
        from data_collectors.ohrid_synthetic_generator import OhridWaterDemandGenerator
        from feature_engineering.temporal_features import TemporalFeatureEngineer
        from models.ohrid_predictor import OhridWaterDemandPredictor
        
        # Generate small dataset
        generator = OhridWaterDemandGenerator()
        data = generator.generate_synthetic_data(
            start_date="2023-01-01",
            end_date="2023-01-02",
            frequency="1h"
        )
        
        # Add temporal features
        feature_engineer = TemporalFeatureEngineer()
        featured_data = feature_engineer.create_all_temporal_features(data)
        
        # Test model pipeline
        predictor = OhridWaterDemandPredictor()
        X_train, X_val, X_test, y_train, y_val, y_test, features = \
            predictor.prepare_data_for_modeling(featured_data)
        
        print("  ✓ Data generation -> Feature engineering -> Model preparation")
        
        # Test hybrid data manager
        from data_collectors.ohrid_data_manager import OhridDataManager
        manager = OhridDataManager()
        current_data = manager.get_current_data()
        
        print("  ✓ Hybrid data manager integration")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        return False


def run_gcp_tests():
    """Run GCP integration tests if credentials available."""
    print("Running GCP Tests...")
    
    try:
        # Check for GCP credentials
        import os
        from google.auth import default
        
        # Try to get default credentials
        credentials, project = default()
        
        if not project:
            print("  - No GCP project configured, skipping")
            return None
        
        print(f"  ✓ GCP credentials found for project: {project}")
        
        # Test basic GCP connectivity
        from google.cloud import storage
        client = storage.Client(project=project)
        
        # List buckets (this will fail if no access)
        buckets = list(client.list_buckets())
        print(f"  ✓ GCP Storage access verified ({len(buckets)} buckets)")
        
        return True
        
    except Exception as e:
        print(f"  - GCP tests skipped: {e}")
        return None


def check_environment():
    """Check if environment is properly set up for tests."""
    print("Checking Environment...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm',
        'tensorflow', 'statsmodels', 'holidays'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✓ All required packages available")
    return True


if __name__ == "__main__":
    # Check environment first
    if not check_environment():
        print("Environment check failed. Please install requirements.")
        sys.exit(1)
    
    # Run comprehensive test suite
    success = run_comprehensive_test_suite()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)