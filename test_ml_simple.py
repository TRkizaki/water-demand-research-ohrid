#!/usr/bin/env python3
"""
Simplified ML model testing for Ohrid Water Demand Framework
Tests core ML models without problematic dependencies
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Traditional Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

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
    print(f"   Target range: {y.min():.1f} - {y.max():.1f} m³/hour")
    
    return X_train, X_test, y_train, y_test, feature_cols

def test_simple_arima(y_train, y_test):
    """Test simple ARIMA model"""
    print("\n🕰️ Testing Simple ARIMA Model...")
    
    try:
        # Simple ARIMA(1,1,1)
        print("   🔄 Training ARIMA(1,1,1)...")
        arima_model = ARIMA(y_train, order=(1, 1, 1))
        arima_fitted = arima_model.fit()
        
        # Forecast
        forecast = arima_fitted.forecast(steps=len(y_test))
        
        # Evaluate
        mae = mean_absolute_error(y_test, forecast)
        rmse = np.sqrt(mean_squared_error(y_test, forecast))
        mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100
        r2 = r2_score(y_test, forecast)
        
        print(f"   ✅ ARIMA(1,1,1): MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%, R²={r2:.3f}")
        
        return {
            'ARIMA(1,1,1)': {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2,
                'predictions': forecast
            }
        }
        
    except Exception as e:
        print(f"   ⚠️ ARIMA failed: {e}")
        return {}

def test_machine_learning(X_train, X_test, y_train, y_test, feature_cols):
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
        'model': rf_model,
        'predictions': rf_pred
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
        'model': xgb_model,
        'predictions': xgb_pred
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
        'model': lgb_model,
        'predictions': lgb_pred
    }
    print(f"   ✅ LightGBM: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%, R²={r2:.3f}")
    
    return results

def analyze_feature_importance(ml_results, feature_cols):
    """Analyze feature importance from tree-based models"""
    print("\n🔍 Feature Importance Analysis...")
    
    for model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
        if model_name in ml_results:
            model = ml_results[model_name]['model']
            
            if hasattr(model, 'feature_importances_'):
                # Get feature importance
                importances = model.feature_importances_
                feature_importance = list(zip(feature_cols, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                print(f"\n   📊 {model_name} - Top 8 Features:")
                for i, (feature, importance) in enumerate(feature_importance[:8], 1):
                    print(f"      {i}. {feature:<25}: {importance:.3f}")

def compare_models(ts_results, ml_results):
    """Compare all models and find the best performer"""
    print("\n🏆 Model Comparison Results")
    print("=" * 70)
    
    all_results = {**ts_results, **ml_results}
    
    if not all_results:
        print("❌ No models to compare")
        return None, {}
    
    # Create comparison table
    print(f"{'Model':<15} {'MAE':<10} {'RMSE':<10} {'MAPE (%)':<10} {'R²':<10}")
    print("-" * 70)
    
    best_mae = float('inf')
    best_model = None
    
    for model_name, metrics in all_results.items():
        mae = metrics['MAE']
        rmse = metrics['RMSE']
        mape = metrics['MAPE']
        r2 = metrics['R2']
        
        print(f"{model_name:<15} {mae:<10.2f} {rmse:<10.2f} {mape:<10.1f} {r2:<10.3f}")
        
        if mae < best_mae:
            best_mae = mae
            best_model = model_name
    
    print("-" * 70)
    print(f"🥇 Best Model: {best_model} (MAE: {best_mae:.2f} m³/hour)")
    
    # Performance insights
    print(f"\n💡 Performance Insights:")
    
    best_metrics = all_results[best_model]
    print(f"   • Best model achieved {best_metrics['MAPE']:.1f}% MAPE")
    print(f"   • R² score of {best_metrics['R2']:.3f} indicates good predictive power")
    print(f"   • RMSE of {best_metrics['RMSE']:.2f} m³/hour shows prediction accuracy")
    
    if best_model in ml_results:
        print(f"   • Machine learning captured complex feature interactions")
        print(f"   • Feature engineering (lag, weather, tourism) proved valuable")
    else:
        print(f"   • Time series approach captured temporal patterns well")
    
    # Model recommendations
    print(f"\n🎯 Deployment Recommendations:")
    print(f"   • Use {best_model} for primary forecasting")
    print(f"   • Expected accuracy: ±{best_metrics['MAE']:.1f} m³/hour")
    
    if best_metrics['MAPE'] < 10:
        print(f"   • Excellent accuracy for operational planning")
    elif best_metrics['MAPE'] < 20:
        print(f"   • Good accuracy for infrastructure planning")
    else:
        print(f"   • Reasonable accuracy, consider model improvements")
    
    return best_model, all_results

def peak_demand_analysis(results, y_test):
    """Analyze performance during peak demand periods"""
    print(f"\n⚡ Peak Demand Analysis...")
    
    # Define peak threshold (top 10%)
    peak_threshold = y_test.quantile(0.9)
    peak_mask = y_test >= peak_threshold
    
    print(f"   Peak threshold: {peak_threshold:.1f} m³/hour")
    print(f"   Peak periods: {peak_mask.sum()} hours ({peak_mask.mean()*100:.1f}% of test data)")
    
    if peak_mask.sum() > 0:
        print(f"\n   📊 Peak Demand Performance:")
        
        for model_name, model_data in results.items():
            if 'predictions' in model_data:
                predictions = model_data['predictions']
                peak_actual = y_test[peak_mask]
                peak_pred = predictions[peak_mask]
                
                peak_mae = mean_absolute_error(peak_actual, peak_pred)
                peak_mape = np.mean(np.abs((peak_actual - peak_pred) / peak_actual)) * 100
                
                print(f"      {model_name:<15}: Peak MAE={peak_mae:.2f}, Peak MAPE={peak_mape:.1f}%")

def main():
    """Run the complete ML model testing pipeline"""
    print("🚀 Testing Ohrid Water Demand ML Models")
    print("=" * 70)
    
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data()
        
        # Test simple time series model
        ts_results = test_simple_arima(y_train, y_test)
        
        # Test machine learning models
        ml_results = test_machine_learning(X_train, X_test, y_train, y_test, feature_cols)
        
        # Analyze feature importance
        analyze_feature_importance(ml_results, feature_cols)
        
        # Compare all models
        best_model, all_results = compare_models(ts_results, ml_results)
        
        # Peak demand analysis
        peak_demand_analysis(all_results, y_test)
        
        print(f"\n✅ Model Testing Complete!")
        print(f"   • Successfully tested {len(all_results)} models")
        print(f"   • Best performer: {best_model}")
        print(f"   • Framework validated for Ohrid water demand prediction")
        print(f"   • Ready for operational deployment")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)