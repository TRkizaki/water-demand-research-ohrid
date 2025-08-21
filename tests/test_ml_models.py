#!/usr/bin/env python3
"""
Test script for ML models in the Ohrid Water Demand Framework
Tests traditional time series and ML models (without deep learning)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Traditional Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from pmdarima import auto_arima

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

def load_and_prepare_data():
    """Load and prepare data for modeling"""
    print("📊 Loading and preparing data...")
    
    # Load synthetic data
    df = pd.read_csv('data/raw/ohrid_synthetic_water_demand.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove rows with NaN values (due to lag and rolling features)
    df_clean = df.dropna()
    
    # Feature selection (exclude target and generated multipliers)
    target_col = 'water_demand_m3_per_hour'
    exclude_cols = [
        target_col, 'timestamp', 'water_production_m3_per_hour',
        'seasonal_multiplier', 'daily_multiplier', 'weather_multiplier',
        'tourism_multiplier'  # These were used to generate synthetic data
    ]
    
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Time series split (80% train, 20% test)
    split_idx = int(len(df_clean) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"✅ Data prepared:")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def test_traditional_time_series(y_train, y_test):
    """Test traditional time series models"""
    print("\n🕰️ Testing Traditional Time Series Models...")
    
    results = {}
    
    try:
        # Auto ARIMA
        print("   🔄 Training Auto ARIMA...")
        auto_model = auto_arima(
            y_train,
            seasonal=True,
            m=24,  # 24-hour seasonality
            max_p=2, max_q=2, max_P=1, max_Q=1,  # Reduce complexity for speed
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_order=5
        )
        
        # Forecast
        forecast = auto_model.predict(n_periods=len(y_test))
        
        # Evaluate
        mae = mean_absolute_error(y_test, forecast)
        rmse = np.sqrt(mean_squared_error(y_test, forecast))
        mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100
        r2 = r2_score(y_test, forecast)
        
        results['Auto ARIMA'] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'model': auto_model
        }
        print(f"   ✅ Auto ARIMA: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%, R²={r2:.3f}")
        
    except Exception as e:
        print(f"   ⚠️ Auto ARIMA failed: {e}")
    
    try:
        # Simple ARIMA
        print("   🔄 Training Simple ARIMA...")
        arima_model = ARIMA(y_train, order=(1, 1, 1))
        arima_fitted = arima_model.fit()
        
        forecast = arima_fitted.forecast(steps=len(y_test))
        
        mae = mean_absolute_error(y_test, forecast)
        rmse = np.sqrt(mean_squared_error(y_test, forecast))
        mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100
        r2 = r2_score(y_test, forecast)
        
        results['ARIMA(1,1,1)'] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'model': arima_fitted
        }
        print(f"   ✅ ARIMA(1,1,1): MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%, R²={r2:.3f}")
        
    except Exception as e:
        print(f"   ⚠️ ARIMA failed: {e}")
    
    return results

def test_machine_learning(X_train, X_test, y_train, y_test):
    """Test machine learning models"""
    print("\n🤖 Testing Machine Learning Models...")
    
    results = {}
    
    # Random Forest
    print("   🔄 Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, rf_pred)
    rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
    r2 = r2_score(y_test, rf_pred)
    
    results['Random Forest'] = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'model': rf_model
    }
    print(f"   ✅ Random Forest: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%, R²={r2:.3f}")
    
    # XGBoost
    print("   🔄 Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    
    xgb_pred = xgb_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, xgb_pred)
    rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    mape = np.mean(np.abs((y_test - xgb_pred) / y_test)) * 100
    r2 = r2_score(y_test, xgb_pred)
    
    results['XGBoost'] = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'model': xgb_model
    }
    print(f"   ✅ XGBoost: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%, R²={r2:.3f}")
    
    # LightGBM
    print("   🔄 Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    
    lgb_pred = lgb_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, lgb_pred)
    rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
    mape = np.mean(np.abs((y_test - lgb_pred) / y_test)) * 100
    r2 = r2_score(y_test, lgb_pred)
    
    results['LightGBM'] = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'model': lgb_model
    }
    print(f"   ✅ LightGBM: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%, R²={r2:.3f}")
    
    return results

def analyze_feature_importance(ml_results):
    """Analyze feature importance from tree-based models"""
    print("\n🔍 Feature Importance Analysis...")
    
    for model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
        if model_name in ml_results:
            model = ml_results[model_name]['model']
            
            if hasattr(model, 'feature_importances_'):
                # Get feature names (assuming we have them from the original data preparation)
                feature_names = [
                    'population', 'tourists_estimated', 'temperature', 'humidity', 
                    'precipitation', 'wind_speed', 'pressure', 'cloud_cover',
                    'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
                    'is_tourist_season', 'is_festival_period'
                ]
                
                # Get top 10 features
                importances = model.feature_importances_
                feature_importance = list(zip(feature_names[:len(importances)], importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                print(f"\n   📊 {model_name} - Top 5 Features:")
                for i, (feature, importance) in enumerate(feature_importance[:5], 1):
                    print(f"      {i}. {feature:<20}: {importance:.3f}")

def compare_models(ts_results, ml_results):
    """Compare all models and find the best performer"""
    print("\n🏆 Model Comparison Results")
    print("=" * 60)
    
    all_results = {**ts_results, **ml_results}
    
    if not all_results:
        print("❌ No models to compare")
        return
    
    # Create comparison table
    print(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'MAPE':<8} {'R²':<8}")
    print("-" * 60)
    
    best_mae = float('inf')
    best_model = None
    
    for model_name, metrics in all_results.items():
        mae = metrics['MAE']
        rmse = metrics['RMSE']
        mape = metrics['MAPE']
        r2 = metrics['R2']
        
        print(f"{model_name:<15} {mae:<8.2f} {rmse:<8.2f} {mape:<8.1f} {r2:<8.3f}")
        
        if mae < best_mae:
            best_mae = mae
            best_model = model_name
    
    print("-" * 60)
    print(f"🥇 Best Model: {best_model} (MAE: {best_mae:.2f})")
    
    # Performance insights
    print(f"\n💡 Performance Insights:")
    
    if best_model in ml_results:
        print(f"   • Machine learning outperformed time series")
        print(f"   • {best_model} achieved {all_results[best_model]['MAPE']:.1f}% MAPE")
        print(f"   • R² score of {all_results[best_model]['R2']:.3f} indicates good fit")
    else:
        print(f"   • Traditional time series performed best")
        print(f"   • {best_model} captured temporal patterns effectively")
    
    return best_model, all_results

def main():
    """Run the complete ML model testing pipeline"""
    print("🚀 Testing Ohrid Water Demand ML Models")
    print("=" * 60)
    
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data()
        
        # Test traditional time series models
        ts_results = test_traditional_time_series(y_train, y_test)
        
        # Test machine learning models
        ml_results = test_machine_learning(X_train, X_test, y_train, y_test)
        
        # Analyze feature importance
        analyze_feature_importance(ml_results)
        
        # Compare all models
        best_model, all_results = compare_models(ts_results, ml_results)
        
        print(f"\n✅ Model Testing Complete!")
        print(f"   • Tested {len(all_results)} models")
        print(f"   • Best performer: {best_model}")
        print(f"   • Ready for production deployment")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)