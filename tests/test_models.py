"""
Test Suite for Model Components

Tests model training, evaluation, and prediction functionality.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.ohrid_predictor import OhridWaterDemandPredictor


class TestOhridPredictor:
    """Test the main predictor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.predictor = OhridWaterDemandPredictor()
        
        # Create sample data for testing
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'water_demand_m3_per_hour': np.random.uniform(100, 400, 100),
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'is_weekend': dates.dayofweek.isin([5, 6]),
            'temperature': np.random.uniform(10, 30, 100),
            'humidity': np.random.uniform(40, 80, 100),
            'precipitation': np.random.uniform(0, 5, 100),
            'tourists_estimated': np.random.uniform(500, 3000, 100),
            'population': [42033] * 100,
            'is_tourist_season': dates.month.isin([6, 7, 8]),
            'is_festival_period': [False] * 100
        })
        self.sample_data.set_index('timestamp', inplace=True)
        
    def test_data_preparation(self):
        """Test data preparation for modeling."""
        X_train, X_val, X_test, y_train, y_val, y_test, features = \
            self.predictor.prepare_data_for_modeling(self.sample_data)
        
        # Check splits
        total_samples = len(self.sample_data)
        assert len(X_train) + len(X_val) + len(X_test) <= total_samples
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)
        
        # Check features
        assert len(features) > 0
        assert 'hour' in features
        assert 'temperature' in features
        
    def test_model_training(self):
        """Test basic model training functionality."""
        X_train, X_val, X_test, y_train, y_val, y_test, features = \
            self.predictor.prepare_data_for_modeling(self.sample_data)
        
        # Test machine learning models
        ml_models = self.predictor.fit_machine_learning_models(X_train, y_train, X_val, y_val)
        
        assert len(ml_models) > 0
        assert 'RandomForest' in ml_models
        
    def test_model_evaluation(self):
        """Test model evaluation functionality."""
        X_train, X_val, X_test, y_train, y_val, y_test, features = \
            self.predictor.prepare_data_for_modeling(self.sample_data)
        
        # Train a simple model
        ml_models = self.predictor.fit_machine_learning_models(X_train, y_train, X_val, y_val)
        
        # Evaluate models
        results = self.predictor.evaluate_models(X_test, y_test)
        
        assert len(results) > 0
        for model_name, metrics in results.items():
            assert 'MAE' in metrics
            assert 'RMSE' in metrics
            assert 'R2' in metrics
            assert metrics['MAE'] >= 0
            assert metrics['RMSE'] >= 0


def run_model_tests():
    """Run all model tests."""
    test_class = TestOhridPredictor
    
    print("Running Model Tests...")
    
    total_tests = 0
    passed_tests = 0
    
    # Get test methods
    test_methods = [method for method in dir(test_class) 
                   if method.startswith('test_')]
    
    for test_method in test_methods:
        total_tests += 1
        try:
            # Create instance and run test
            instance = test_class()
            instance.setup_method()
            
            getattr(instance, test_method)()
            print(f"  ✓ {test_method}")
            passed_tests += 1
            
        except Exception as e:
            print(f"  ✗ {test_method}: {e}")
    
    print(f"Model Tests: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_model_tests()
    exit(0 if success else 1)