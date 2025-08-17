"""
Comprehensive Water Demand Prediction Framework for Ohrid

Implements traditional time series, machine learning, and hybrid approaches
with evaluation framework tailored for water demand forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import yaml
import joblib
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Traditional Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Handle pmdarima import gracefully 
PMDARIMA_AVAILABLE = False
try:
    import pmdarima
    PMDARIMA_AVAILABLE = True
except ImportError:
    pass
except Exception:
    pass

# Import comprehensive time series analyzer
from models.time_series_analyzer import TimeSeriesAnalyzer

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')


class OhridWaterDemandPredictor:
    """
    Comprehensive water demand prediction framework for Ohrid, North Macedonia.
    
    Features:
    - Traditional time series methods (ARIMA, ETS)
    - Machine learning models (RF, XGBoost, LightGBM)
    - Deep learning models (LSTM, GRU)
    - Hybrid ensemble approaches
    - Comprehensive evaluation framework
    - Peak demand analysis
    - Tourism impact assessment
    """
    
    def __init__(self, config_path: str = "config/ohrid_config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.scalers = {}
        self.evaluation_results = {}
        self.feature_importance = {}
        
        # Initialize comprehensive time series analyzer
        self.ts_analyzer = TimeSeriesAnalyzer()
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and prepare data for modeling."""
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add additional temporal features
        df = self._engineer_temporal_features(df)
        
        return df
    
    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional temporal features."""
        df = df.copy()
        
        # Cyclical encoding for better periodicity capture
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Interaction features
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
        df['tourist_temp_interaction'] = df['tourists_estimated'] * df['temperature']
        
        # Peak demand indicators
        df['is_morning_peak'] = ((df['hour'] >= 6) & (df['hour'] <= 8)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
        
        return df
    
    def prepare_data_for_modeling(self, 
                                 df: pd.DataFrame, 
                                 target_col: str = 'water_demand_m3_per_hour',
                                 test_size: float = 0.2,
                                 val_size: float = 0.2) -> Tuple:
        """
        Prepare data for modeling with proper time series splits.
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
        """
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        # Feature selection
        exclude_cols = [
            target_col, 'timestamp', 'water_production_m3_per_hour',
            'seasonal_multiplier', 'daily_multiplier', 'weather_multiplier',
            'tourism_multiplier'  # These are used to generate synthetic data
        ]
        
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Time series split
        n_samples = len(df_clean)
        train_end = int(n_samples * (1 - test_size - val_size))
        val_end = int(n_samples * (1 - test_size))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    
    def fit_comprehensive_time_series_models(self, y_train: pd.Series, seasonal_period: int = 24) -> Dict:
        """Fit comprehensive academically rigorous time series models."""
        print("Fitting Comprehensive Time Series Models...")
        print("=" * 60)
        
        # Run comprehensive time series analysis
        ts_results = self.ts_analyzer.comprehensive_analysis(y_train, seasonal_period)
        
        # Extract fitted models from analyzer results
        ts_models = {}
        
        # ARIMA models
        if 'arima' in ts_results:
            for model_name, model_result in ts_results['arima'].items():
                ts_models[f"TS_{model_name}"] = model_result.fitted_model
                
        # SARIMA models
        if 'sarima' in ts_results:
            for model_name, model_result in ts_results['sarima'].items():
                ts_models[f"TS_{model_name}"] = model_result.fitted_model
                
        # Exponential Smoothing models
        if 'exponential_smoothing' in ts_results:
            for model_name, model_result in ts_results['exponential_smoothing'].items():
                ts_models[f"TS_{model_name}"] = model_result.fitted_model
        
        # Store comprehensive analysis results
        self.ts_analysis_results = ts_results
        
        # Update main models dictionary
        self.models.update(ts_models)
        
        print(f"\nTime Series Models Fitted: {len(ts_models)}")
        print("Comprehensive analysis results available in self.ts_analysis_results")
        
        return ts_models
    
    def fit_arima_models(self, y_train: pd.Series, seasonal: bool = True) -> Dict:
        """Legacy method - now uses comprehensive time series analyzer."""
        print("Using comprehensive time series analysis framework...")
        return self.fit_comprehensive_time_series_models(y_train)
    
    def fit_exponential_smoothing(self, y_train: pd.Series) -> Dict:
        """Legacy method - now uses comprehensive time series analyzer."""
        print("Exponential smoothing included in comprehensive time series analysis...")
        # Return empty dict as models are handled by comprehensive analyzer
        return {}
    
    def fit_machine_learning_models(self, 
                                   X_train: pd.DataFrame, 
                                   y_train: pd.Series,
                                   X_val: pd.DataFrame = None,
                                   y_val: pd.Series = None) -> Dict:
        """Fit machine learning models."""
        models = {}
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['RandomForest'] = rf_model
        
        # Store feature importance
        self.feature_importance['RandomForest'] = dict(
            zip(X_train.columns, rf_model.feature_importances_)
        )
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Use validation set if provided
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=20 if eval_set else None,
            verbose=False
        )
        models['XGBoost'] = xgb_model
        
        # Store feature importance
        self.feature_importance['XGBoost'] = dict(
            zip(X_train.columns, xgb_model.feature_importances_)
        )
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(20)] if eval_set else None
        )
        models['LightGBM'] = lgb_model
        
        # Store feature importance
        self.feature_importance['LightGBM'] = dict(
            zip(X_train.columns, lgb_model.feature_importances_)
        )
        
        self.models.update(models)
        return models
    
    def fit_deep_learning_models(self, 
                                X_train: pd.DataFrame, 
                                y_train: pd.Series,
                                X_val: pd.DataFrame = None,
                                y_val: pd.Series = None,
                                sequence_length: int = 24) -> Dict:
        """Fit deep learning models."""
        models = {}
        
        # Scale features for neural networks
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['features'] = scaler
        
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
        
        # Target scaler
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        self.scalers['target'] = target_scaler
        
        if y_val is not None:
            y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
        
        # Dense Neural Network
        nn_model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        nn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10)
        ]
        
        # Validation data for training
        validation_data = (X_val_scaled, y_val_scaled) if X_val is not None else None
        
        nn_model.fit(
            X_train_scaled, y_train_scaled,
            epochs=100,
            batch_size=64,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        models['NeuralNetwork'] = nn_model
        
        # LSTM Model (requires sequence preparation)
        if len(X_train) > sequence_length:
            X_lstm, y_lstm = self._prepare_sequences(X_train_scaled, y_train_scaled, sequence_length)
            
            if X_val is not None:
                X_val_lstm, y_val_lstm = self._prepare_sequences(X_val_scaled, y_val_scaled, sequence_length)
                val_data_lstm = (X_val_lstm, y_val_lstm)
            else:
                val_data_lstm = None
            
            lstm_model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(sequence_length, X_train_scaled.shape[1])),
                Dropout(0.3),
                LSTM(32, return_sequences=False),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            lstm_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            lstm_model.fit(
                X_lstm, y_lstm,
                epochs=50,
                batch_size=32,
                validation_data=val_data_lstm,
                callbacks=callbacks,
                verbose=0
            )
            
            models['LSTM'] = lstm_model
        
        self.models.update(models)
        return models
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple:
        """Prepare sequences for LSTM/GRU models."""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def create_hybrid_ensemble(self, 
                             X_train: pd.DataFrame, 
                             y_train: pd.Series,
                             base_models: List[str] = None) -> Dict:
        """Create hybrid ensemble model."""
        if base_models is None:
            base_models = ['RandomForest', 'XGBoost', 'LightGBM']
        
        # Get base model predictions
        base_predictions = {}
        
        for model_name in base_models:
            if model_name in self.models:
                if model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                    pred = self.models[model_name].predict(X_train)
                elif model_name == 'NeuralNetwork':
                    scaler = self.scalers['features']
                    X_scaled = scaler.transform(X_train)
                    pred_scaled = self.models[model_name].predict(X_scaled)
                    pred = self.scalers['target'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                else:
                    continue
                
                base_predictions[model_name] = pred
        
        if not base_predictions:
            print("No base models available for ensemble")
            return {}
        
        # Create ensemble features
        ensemble_df = pd.DataFrame(base_predictions)
        
        # Train meta-learner (simple weighted average for now)
        weights = np.ones(len(ensemble_df.columns)) / len(ensemble_df.columns)
        
        ensemble_model = {
            'base_models': list(base_predictions.keys()),
            'weights': weights,
            'type': 'weighted_average'
        }
        
        self.models['Ensemble'] = ensemble_model
        return {'Ensemble': ensemble_model}
    
    def predict(self, model_name: str, X_test: pd.DataFrame, horizon_hours: int = 24) -> np.ndarray:
        """Generate predictions using specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name.startswith('TS_') or model_name in ['AutoARIMA', 'SARIMA', 'ETS']:
            # Time series models from comprehensive analyzer
            try:
                predictions = model.forecast(steps=min(len(X_test), horizon_hours))
                if len(predictions) < len(X_test):
                    # Extend predictions if needed (simplified approach)
                    predictions = np.tile(predictions, len(X_test) // len(predictions) + 1)[:len(X_test)]
            except Exception as e:
                print(f"Forecast error for {model_name}: {e}")
                # Fallback to mean prediction
                predictions = np.full(len(X_test), X_test.mean().mean())
            
        elif model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
            predictions = model.predict(X_test)
            
        elif model_name == 'NeuralNetwork':
            scaler = self.scalers['features']
            X_scaled = scaler.transform(X_test)
            pred_scaled = model.predict(X_scaled)
            predictions = self.scalers['target'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            
        elif model_name == 'LSTM':
            # Simplified LSTM prediction (would need proper sequence handling in production)
            scaler = self.scalers['features']
            X_scaled = scaler.transform(X_test)
            # For demonstration, use last sequence_length samples
            sequence_length = 24
            if len(X_scaled) >= sequence_length:
                X_seq = X_scaled[-sequence_length:].reshape(1, sequence_length, -1)
                pred_scaled = model.predict(X_seq)
                single_pred = self.scalers['target'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
                predictions = np.full(len(X_test), single_pred)  # Simplified
            else:
                predictions = np.full(len(X_test), X_test.mean())
                
        elif model_name == 'Ensemble':
            # Ensemble prediction
            base_preds = {}
            for base_model in model['base_models']:
                base_preds[base_model] = self.predict(base_model, X_test, horizon_hours)
            
            # Weighted average
            predictions = np.zeros(len(X_test))
            for i, (model_name, weight) in enumerate(zip(model['base_models'], model['weights'])):
                predictions += weight * base_preds[model_name]
        
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        return predictions
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Comprehensive model evaluation."""
        results = {}
        
        for model_name in self.models.keys():
            try:
                predictions = self.predict(model_name, X_test)
                
                # Basic metrics
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
                r2 = r2_score(y_test, predictions)
                
                # Peak demand accuracy (top 10% of demands)
                peak_threshold = y_test.quantile(0.9)
                peak_mask = y_test >= peak_threshold
                
                if peak_mask.sum() > 0:
                    peak_mae = mean_absolute_error(y_test[peak_mask], predictions[peak_mask])
                    peak_mape = np.mean(np.abs((y_test[peak_mask] - predictions[peak_mask]) / y_test[peak_mask])) * 100
                else:
                    peak_mae = mae
                    peak_mape = mape
                
                # Directional accuracy
                actual_direction = np.diff(y_test) > 0
                pred_direction = np.diff(predictions) > 0
                directional_accuracy = np.mean(actual_direction == pred_direction) * 100
                
                results[model_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'R2': r2,
                    'Peak_MAE': peak_mae,
                    'Peak_MAPE': peak_mape,
                    'Directional_Accuracy': directional_accuracy,
                    'predictions': predictions
                }
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        self.evaluation_results = results
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """Create comprehensive model comparison."""
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_models first.")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        metrics = ['MAE', 'RMSE', 'MAPE', 'R2', 'Peak_MAE', 'Peak_MAPE', 'Directional_Accuracy']
        
        comparison_data = {}
        for model_name, results in self.evaluation_results.items():
            comparison_data[model_name] = {metric: results[metric] for metric in metrics}
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        print("="*80)
        print("MODEL PERFORMANCE COMPARISON - OHRID WATER DEMAND PREDICTION")
        print("="*80)
        print(comparison_df.round(4))
        
        # Rank models
        print("\n" + "="*80)
        print("MODEL RANKINGS BY METRIC")
        print("="*80)
        
        for metric in ['MAE', 'RMSE', 'MAPE']:
            best_model = comparison_df[metric].idxmin()
            best_value = comparison_df.loc[best_model, metric]
            print(f"{metric:20s}: {best_model:15s} ({best_value:.4f})")
        
        best_r2_model = comparison_df['R2'].idxmax()
        best_r2_value = comparison_df.loc[best_r2_model, 'R2']
        print(f"{'R2':20s}: {best_r2_model:15s} ({best_r2_value:.4f})")
        
        best_dir_model = comparison_df['Directional_Accuracy'].idxmax()
        best_dir_value = comparison_df.loc[best_dir_model, 'Directional_Accuracy']
        print(f"{'Directional_Accuracy':20s}: {best_dir_model:15s} ({best_dir_value:.2f}%)")
        
        return comparison_df
    
    def get_time_series_analysis_summary(self) -> pd.DataFrame:
        """Get comprehensive time series analysis summary for academic reporting."""
        if not hasattr(self, 'ts_analysis_results') or not self.ts_analysis_results:
            print("No time series analysis results available. Run fit_comprehensive_time_series_models first.")
            return pd.DataFrame()
        
        print("=" * 80)
        print("COMPREHENSIVE TIME SERIES ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Display stationarity results
        if 'stationarity' in self.ts_analysis_results:
            stationarity = self.ts_analysis_results['stationarity']
            print("\nSTATIONARITY ANALYSIS:")
            print(f"ADF Test - Statistic: {stationarity['adf']['statistic']:.4f}, p-value: {stationarity['adf']['pvalue']:.4f}")
            print(f"KPSS Test - Statistic: {stationarity['kpss']['statistic']:.4f}, p-value: {stationarity['kpss']['pvalue']:.4f}")
            print(f"Series Assessment: {'Stationary' if stationarity['adf']['is_stationary'] and stationarity['kpss']['is_stationary'] else 'Non-stationary'}")
        
        # Get model comparison from comprehensive analysis
        if 'comparison' in self.ts_analysis_results:
            comparison_df = self.ts_analysis_results['comparison']
            print("\nTIME SERIES MODEL COMPARISON:")
            print(comparison_df.head(10))
            
            # Academic insights
            print("\nACADEMIC INSIGHTS:")
            if not comparison_df.empty:
                best_model = comparison_df.iloc[0]['Model']
                best_mae = comparison_df.iloc[0]['Forecast MAE']
                print(f"• Best performing model: {best_model} (MAE: {best_mae:.4f})")
                
                # Model category performance
                category_performance = comparison_df.groupby('Category')['Forecast MAE'].mean()
                print("• Average performance by model category:")
                for category, mae in category_performance.items():
                    print(f"  - {category}: {mae:.4f} MAE")
            
            return comparison_df
        
        return pd.DataFrame()
    
    def plot_feature_importance(self, top_n: int = 15) -> None:
        """Plot feature importance for tree-based models."""
        if not self.feature_importance:
            print("No feature importance data available")
            return
        
        fig, axes = plt.subplots(1, len(self.feature_importance), figsize=(5*len(self.feature_importance), 8))
        if len(self.feature_importance) == 1:
            axes = [axes]
        
        for i, (model_name, importance) in enumerate(self.feature_importance.items()):
            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, importances = zip(*sorted_features)
            
            axes[i].barh(range(len(features)), importances)
            axes[i].set_yticks(range(len(features)))
            axes[i].set_yticklabels(features)
            axes[i].set_xlabel('Importance')
            axes[i].set_title(f'{model_name} - Feature Importance')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_test: pd.Series, model_names: List[str] = None, days_to_show: int = 7) -> None:
        """Plot predictions vs actual for visual comparison."""
        if not self.evaluation_results:
            print("No evaluation results available")
            return
        
        if model_names is None:
            model_names = list(self.evaluation_results.keys())[:4]  # Show top 4 models
        
        # Limit to specified number of days
        samples_to_show = min(len(y_test), days_to_show * 24)  # Assuming hourly data
        
        fig = make_subplots(
            rows=len(model_names), cols=1,
            subplot_titles=[f'{name} Predictions' for name in model_names],
            shared_xaxes=True
        )
        
        for i, model_name in enumerate(model_names, 1):
            if model_name in self.evaluation_results:
                predictions = self.evaluation_results[model_name]['predictions'][:samples_to_show]
                actual = y_test.iloc[:samples_to_show]
                
                fig.add_trace(
                    go.Scatter(
                        y=actual.values,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue', width=1),
                        showlegend=(i == 1)
                    ),
                    row=i, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        y=predictions,
                        mode='lines',
                        name='Predicted',
                        line=dict(color='red', width=1),
                        showlegend=(i == 1)
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            height=300*len(model_names),
            title_text="Water Demand Predictions vs Actual",
            showlegend=True
        )
        
        fig.show()
    
    def save_models(self, model_dir: str = "models/") -> None:
        """Save trained models."""
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                joblib.dump(model, f"{model_dir}/{model_name}_model.pkl")
            elif model_name in ['NeuralNetwork', 'LSTM']:
                model.save(f"{model_dir}/{model_name}_model.h5")
            elif model_name in ['AutoARIMA', 'SARIMA', 'ETS']:
                joblib.dump(model, f"{model_dir}/{model_name}_model.pkl")
        
        # Save scalers
        if self.scalers:
            joblib.dump(self.scalers, f"{model_dir}/scalers.pkl")
        
        print(f"Models saved to {model_dir}")


def main():
    """Example usage of the Ohrid Water Demand Predictor."""
    # Initialize predictor
    predictor = OhridWaterDemandPredictor()
    
    # Load synthetic data (assuming it was generated)
    print("Loading synthetic data...")
    df = predictor.load_data("data/raw/ohrid_synthetic_water_demand.csv")
    
    # Prepare data
    print("Preparing data for modeling...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = predictor.prepare_data_for_modeling(df)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    
    # Fit models
    print("\nFitting comprehensive time series models...")
    predictor.fit_comprehensive_time_series_models(y_train)
    
    print("Fitting machine learning models...")
    predictor.fit_machine_learning_models(X_train, y_train, X_val, y_val)
    
    print("Fitting deep learning models...")
    predictor.fit_deep_learning_models(X_train, y_train, X_val, y_val)
    
    print("Creating ensemble model...")
    predictor.create_hybrid_ensemble(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = predictor.evaluate_models(X_test, y_test)
    
    # Compare models
    comparison_df = predictor.compare_models()
    
    # Time series analysis summary
    print("\nGenerating time series analysis summary...")
    ts_summary = predictor.get_time_series_analysis_summary()
    
    # Feature importance
    print("\nFeature importance analysis...")
    predictor.plot_feature_importance()
    
    # Visualization
    print("Generating prediction plots...")
    predictor.plot_predictions(y_test, days_to_show=14)
    
    # Save models
    print("Saving models...")
    predictor.save_models()
    
    print("\nOhrid Water Demand Prediction Framework - Complete!")


if __name__ == "__main__":
    main()