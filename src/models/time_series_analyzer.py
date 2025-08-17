"""
Comprehensive Time Series Analysis for Water Demand Prediction

Academic-grade implementation with rigorous statistical testing,
model diagnostics, and comparative evaluation framework.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import itertools

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Handle pmdarima import gracefully
PMDARIMA_AVAILABLE = False
try:
    import pmdarima
    PMDARIMA_AVAILABLE = True
except ImportError:
    pass
except Exception as e:
    print(f"Warning: pmdarima import failed: {e}. Auto-ARIMA will be skipped.")

warnings.filterwarnings('ignore')


@dataclass
class ModelResults:
    """Container for model evaluation results."""
    model_name: str
    fitted_model: Any
    aic: float
    bic: float
    mse: float
    mae: float
    mape: float
    ljung_box_pvalue: float
    residual_normality_pvalue: float
    forecast_accuracy: Dict[str, float]
    parameters: Dict[str, Any]


class TimeSeriesAnalyzer:
    """
    Comprehensive time series analysis framework for academic research.
    
    Implements rigorous statistical testing, model diagnostics,
    and comparative evaluation following academic standards.
    """
    
    def __init__(self):
        self.results = {}
        self.diagnostics = {}
        self.data = None
        self.train_data = None
        self.test_data = None
        
    def prepare_data(self, data: pd.Series, train_ratio: float = 0.8) -> Tuple[pd.Series, pd.Series]:
        """Prepare time series data with proper train/test split."""
        self.data = data.copy()
        
        # Time series split (no shuffling)
        split_point = int(len(data) * train_ratio)
        self.train_data = data.iloc[:split_point]
        self.test_data = data.iloc[split_point:]
        
        print(f"Data prepared: {len(self.train_data)} train, {len(self.test_data)} test observations")
        return self.train_data, self.test_data
    
    def analyze_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Comprehensive stationarity analysis."""
        print("Analyzing Stationarity...")
        print("-" * 40)
        
        results = {}
        
        # Augmented Dickey-Fuller Test
        adf_result = adfuller(series.dropna())
        results['adf'] = {
            'statistic': adf_result[0],
            'pvalue': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        
        print(f"ADF Test:")
        print(f"  Statistic: {adf_result[0]:.4f}")
        print(f"  p-value: {adf_result[1]:.4f}")
        print(f"  Critical Values: {adf_result[4]}")
        print(f"  Result: {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'}")
        
        # KPSS Test
        kpss_result = kpss(series.dropna())
        results['kpss'] = {
            'statistic': kpss_result[0],
            'pvalue': kpss_result[1],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > 0.05
        }
        
        print(f"\nKPSS Test:")
        print(f"  Statistic: {kpss_result[0]:.4f}")
        print(f"  p-value: {kpss_result[1]:.4f}")
        print(f"  Result: {'Stationary' if kpss_result[1] > 0.05 else 'Non-stationary'}")
        
        return results
    
    def seasonal_decomposition(self, series: pd.Series, period: int = 24) -> Dict[str, pd.Series]:
        """Perform seasonal decomposition analysis."""
        print(f"\nPerforming Seasonal Decomposition (period={period})...")
        
        # Handle missing values
        series_clean = series.dropna()
        
        # Additive decomposition
        decomp_add = seasonal_decompose(series_clean, model='additive', period=period)
        
        # Multiplicative decomposition (if no zeros/negatives)
        if (series_clean > 0).all():
            decomp_mult = seasonal_decompose(series_clean, model='multiplicative', period=period)
        else:
            decomp_mult = None
        
        results = {
            'additive': {
                'trend': decomp_add.trend,
                'seasonal': decomp_add.seasonal,
                'residual': decomp_add.resid
            }
        }
        
        if decomp_mult:
            results['multiplicative'] = {
                'trend': decomp_mult.trend,
                'seasonal': decomp_mult.seasonal,
                'residual': decomp_mult.resid
            }
        
        return results
    
    def determine_arima_orders(self, series: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Dict:
        """Determine optimal ARIMA orders using multiple methods."""
        print("\nDetermining ARIMA Orders...")
        print("-" * 40)
        
        results = {}
        
        # Method 1: Auto ARIMA (if available)
        if PMDARIMA_AVAILABLE:
            try:
                auto_model = pmdarima.auto_arima(
                    series,
                    start_p=0, start_q=0,
                    max_p=max_p, max_q=max_q, max_d=max_d,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    information_criterion='aic'
                )
                results['auto_arima'] = {
                    'order': auto_model.order,
                    'aic': auto_model.aic(),
                    'bic': auto_model.bic()
                }
                print(f"Auto-ARIMA: {auto_model.order}, AIC: {auto_model.aic():.2f}")
            except Exception as e:
                print(f"Auto-ARIMA failed: {e}")
        else:
            print("Auto-ARIMA skipped (pmdarima not available)")
        
        # Method 2: Grid Search
        print("\nGrid Search for optimal parameters...")
        best_aic = np.inf
        best_order = None
        aic_results = []
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        aic = fitted.aic
                        aic_results.append(((p, d, q), aic))
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            
                    except:
                        continue
        
        results['grid_search'] = {
            'best_order': best_order,
            'best_aic': best_aic,
            'all_results': sorted(aic_results, key=lambda x: x[1])[:10]  # Top 10
        }
        
        print(f"Grid Search Best: {best_order}, AIC: {best_aic:.2f}")
        
        # Method 3: ACF/PACF Analysis
        acf_values = acf(series.dropna(), nlags=40, fft=False)
        pacf_values = pacf(series.dropna(), nlags=40)
        
        results['acf_pacf'] = {
            'acf_values': acf_values,
            'pacf_values': pacf_values
        }
        
        return results
    
    def fit_arima_models(self, series: pd.Series) -> Dict[str, ModelResults]:
        """Fit comprehensive ARIMA model suite."""
        print("\nFitting ARIMA Models...")
        print("-" * 40)
        
        models = {}
        order_analysis = self.determine_arima_orders(series)
        
        # Model configurations to test
        model_configs = [
            ('Auto-ARIMA', order_analysis.get('auto_arima', {}).get('order', (1, 1, 1))),
            ('Grid-Best', order_analysis.get('grid_search', {}).get('best_order', (1, 1, 1))),
            ('ARIMA(1,1,1)', (1, 1, 1)),
            ('ARIMA(2,1,2)', (2, 1, 2)),
            ('ARIMA(3,1,3)', (3, 1, 3)),
        ]
        
        for model_name, order in model_configs:
            if order is None:
                continue
                
            try:
                print(f"Fitting {model_name} {order}...")
                model = ARIMA(series, order=order)
                fitted = model.fit()
                
                # Model diagnostics
                residuals = fitted.resid
                ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
                ljung_box_pvalue = ljung_box['lb_pvalue'].iloc[-1]
                
                # Residual normality test
                _, normality_pvalue = stats.jarque_bera(residuals.dropna())
                
                # Forecast accuracy on test data
                forecast_result = self._evaluate_forecast_accuracy(fitted, len(self.test_data))
                
                result = ModelResults(
                    model_name=model_name,
                    fitted_model=fitted,
                    aic=fitted.aic,
                    bic=fitted.bic,
                    mse=np.mean(residuals**2),
                    mae=np.mean(np.abs(residuals)),
                    mape=np.mean(np.abs(residuals / series.iloc[len(residuals):]) * 100),
                    ljung_box_pvalue=ljung_box_pvalue,
                    residual_normality_pvalue=normality_pvalue,
                    forecast_accuracy=forecast_result,
                    parameters={'order': order}
                )
                
                models[model_name] = result
                print(f"  AIC: {fitted.aic:.2f}, BIC: {fitted.bic:.2f}")
                
            except Exception as e:
                print(f"  Failed: {e}")
        
        return models
    
    def determine_sarima_orders(self, series: pd.Series, m: int = 24) -> Dict:
        """Determine optimal SARIMA orders."""
        print(f"\nDetermining SARIMA Orders (seasonal period = {m})...")
        print("-" * 50)
        
        results = {}
        
        # Auto SARIMA (if available)
        if PMDARIMA_AVAILABLE:
            try:
                auto_model = pmdarima.auto_arima(
                    series,
                    seasonal=True, m=m,
                    max_p=3, max_q=3, max_P=2, max_Q=2, max_d=2, max_D=1,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore'
                )
                results['auto_sarima'] = {
                    'order': auto_model.order,
                    'seasonal_order': auto_model.seasonal_order,
                    'aic': auto_model.aic(),
                    'bic': auto_model.bic()
                }
                print(f"Auto-SARIMA: {auto_model.order}x{auto_model.seasonal_order}, AIC: {auto_model.aic():.2f}")
            except Exception as e:
                print(f"Auto-SARIMA failed: {e}")
        else:
            print("Auto-SARIMA skipped (pmdarima not available)")
        
        # Manual SARIMA grid search (limited for computational efficiency)
        print("Limited SARIMA grid search...")
        best_aic = np.inf
        best_config = None
        
        p_range = range(0, 3)
        d_range = range(0, 2)
        q_range = range(0, 3)
        P_range = range(0, 2)
        D_range = range(0, 2)
        Q_range = range(0, 2)
        
        configs_tested = 0
        max_configs = 50  # Limit for computational efficiency
        
        for p, d, q, P, D, Q in itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range):
            if configs_tested >= max_configs:
                break
                
            try:
                model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, m))
                fitted = model.fit(disp=False)
                aic = fitted.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_config = ((p, d, q), (P, D, Q, m))
                
                configs_tested += 1
                
            except:
                continue
        
        if best_config:
            results['manual_best'] = {
                'order': best_config[0],
                'seasonal_order': best_config[1],
                'aic': best_aic
            }
            print(f"Manual Best: {best_config[0]}x{best_config[1]}, AIC: {best_aic:.2f}")
        
        return results
    
    def fit_sarima_models(self, series: pd.Series, m: int = 24) -> Dict[str, ModelResults]:
        """Fit comprehensive SARIMA model suite."""
        print("\nFitting SARIMA Models...")
        print("-" * 40)
        
        models = {}
        order_analysis = self.determine_sarima_orders(series, m)
        
        # Model configurations
        model_configs = []
        
        if 'auto_sarima' in order_analysis:
            auto_config = order_analysis['auto_sarima']
            model_configs.append(('Auto-SARIMA', auto_config['order'], auto_config['seasonal_order']))
        
        if 'manual_best' in order_analysis:
            manual_config = order_analysis['manual_best']
            model_configs.append(('Manual-Best-SARIMA', manual_config['order'], manual_config['seasonal_order']))
        
        # Standard configurations
        model_configs.extend([
            ('SARIMA(1,1,1)(1,1,1,24)', (1, 1, 1), (1, 1, 1, m)),
            ('SARIMA(2,1,2)(1,1,1,24)', (2, 1, 2), (1, 1, 1, m)),
            ('SARIMA(1,1,1)(2,1,2,24)', (1, 1, 1), (2, 1, 2, m)),
        ])
        
        for model_name, order, seasonal_order in model_configs:
            try:
                print(f"Fitting {model_name}...")
                model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
                fitted = model.fit(disp=False)
                
                # Diagnostics
                residuals = fitted.resid
                ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
                ljung_box_pvalue = ljung_box['lb_pvalue'].iloc[-1]
                
                _, normality_pvalue = stats.jarque_bera(residuals.dropna())
                
                forecast_result = self._evaluate_forecast_accuracy(fitted, len(self.test_data))
                
                result = ModelResults(
                    model_name=model_name,
                    fitted_model=fitted,
                    aic=fitted.aic,
                    bic=fitted.bic,
                    mse=np.mean(residuals**2),
                    mae=np.mean(np.abs(residuals)),
                    mape=np.mean(np.abs(residuals / series.iloc[len(residuals):]) * 100),
                    ljung_box_pvalue=ljung_box_pvalue,
                    residual_normality_pvalue=normality_pvalue,
                    forecast_accuracy=forecast_result,
                    parameters={'order': order, 'seasonal_order': seasonal_order}
                )
                
                models[model_name] = result
                print(f"  AIC: {fitted.aic:.2f}, BIC: {fitted.bic:.2f}")
                
            except Exception as e:
                print(f"  Failed: {e}")
        
        return models
    
    def fit_exponential_smoothing_models(self, series: pd.Series) -> Dict[str, ModelResults]:
        """Fit comprehensive exponential smoothing model suite."""
        print("\nFitting Exponential Smoothing Models...")
        print("-" * 45)
        
        models = {}
        
        # Model configurations: (trend, seasonal, seasonal_periods)
        es_configs = [
            ('Simple ES', None, None, None),
            ('Double ES (Holt)', 'add', None, None),
            ('Triple ES Add', 'add', 'add', 24),
            ('Triple ES Mult', 'add', 'mul', 24),
            ('Holt-Winters Add', 'add', 'add', 24),
            ('Holt-Winters Mult', 'mul', 'mul', 24),
            ('ETS(A,A,A)', 'add', 'add', 24),
            ('ETS(A,M,A)', 'mul', 'add', 24),
            ('ETS(M,A,M)', 'add', 'mul', 24),
        ]
        
        for model_name, trend, seasonal, seasonal_periods in es_configs:
            try:
                print(f"Fitting {model_name}...")
                
                if 'ETS' in model_name:
                    # Use ETSModel for ETS variants
                    if trend == 'add':
                        trend_param = 'add'
                    elif trend == 'mul':
                        trend_param = 'mul'
                    else:
                        trend_param = None
                    
                    model = ETSModel(
                        series,
                        error='add',
                        trend=trend_param,
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods
                    )
                else:
                    # Use ExponentialSmoothing for traditional methods
                    model = ExponentialSmoothing(
                        series,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods
                    )
                
                fitted = model.fit()
                
                # Calculate residuals
                fitted_values = fitted.fittedvalues
                residuals = series - fitted_values
                residuals = residuals.dropna()
                
                # Diagnostics
                if len(residuals) > 10:
                    ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//2), return_df=True)
                    ljung_box_pvalue = ljung_box['lb_pvalue'].iloc[-1]
                    _, normality_pvalue = stats.jarque_bera(residuals)
                else:
                    ljung_box_pvalue = np.nan
                    normality_pvalue = np.nan
                
                forecast_result = self._evaluate_forecast_accuracy(fitted, len(self.test_data))
                
                result = ModelResults(
                    model_name=model_name,
                    fitted_model=fitted,
                    aic=fitted.aic,
                    bic=fitted.bic,
                    mse=np.mean(residuals**2),
                    mae=np.mean(np.abs(residuals)),
                    mape=np.mean(np.abs(residuals / series.iloc[-len(residuals):]) * 100) if len(residuals) > 0 else np.nan,
                    ljung_box_pvalue=ljung_box_pvalue,
                    residual_normality_pvalue=normality_pvalue,
                    forecast_accuracy=forecast_result,
                    parameters={'trend': trend, 'seasonal': seasonal, 'seasonal_periods': seasonal_periods}
                )
                
                models[model_name] = result
                print(f"  AIC: {fitted.aic:.2f}, BIC: {fitted.bic:.2f}")
                
            except Exception as e:
                print(f"  Failed: {e}")
        
        return models
    
    def _evaluate_forecast_accuracy(self, model, horizon: int) -> Dict[str, float]:
        """Evaluate forecast accuracy on test data."""
        try:
            forecast = model.forecast(steps=horizon)
            
            if len(forecast) != len(self.test_data):
                # Adjust if lengths don't match
                min_len = min(len(forecast), len(self.test_data))
                forecast = forecast[:min_len]
                test_data = self.test_data.iloc[:min_len]
            else:
                test_data = self.test_data
            
            mae = mean_absolute_error(test_data, forecast)
            mse = mean_squared_error(test_data, forecast)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
            
            return {
                'forecast_mae': mae,
                'forecast_mse': mse,
                'forecast_rmse': rmse,
                'forecast_mape': mape
            }
        except Exception as e:
            return {
                'forecast_mae': np.nan,
                'forecast_mse': np.nan,
                'forecast_rmse': np.nan,
                'forecast_mape': np.nan
            }
    
    def comprehensive_analysis(self, series: pd.Series, seasonal_period: int = 24) -> Dict:
        """Run comprehensive time series analysis."""
        print("=" * 60)
        print("COMPREHENSIVE TIME SERIES ANALYSIS")
        print("=" * 60)
        
        # Prepare data
        train_data, test_data = self.prepare_data(series)
        
        # Store all results
        all_results = {}
        
        # 1. Stationarity Analysis
        stationarity = self.analyze_stationarity(train_data)
        all_results['stationarity'] = stationarity
        
        # 2. Seasonal Decomposition
        decomposition = self.seasonal_decomposition(train_data, seasonal_period)
        all_results['decomposition'] = decomposition
        
        # 3. ARIMA Models
        arima_results = self.fit_arima_models(train_data)
        all_results['arima'] = arima_results
        
        # 4. SARIMA Models
        sarima_results = self.fit_sarima_models(train_data, seasonal_period)
        all_results['sarima'] = sarima_results
        
        # 5. Exponential Smoothing Models
        es_results = self.fit_exponential_smoothing_models(train_data)
        all_results['exponential_smoothing'] = es_results
        
        # 6. Model Comparison
        comparison = self.compare_all_models(all_results)
        all_results['comparison'] = comparison
        
        self.results = all_results
        return all_results
    
    def compare_all_models(self, results: Dict) -> pd.DataFrame:
        """Create comprehensive model comparison."""
        print("\nModel Comparison Summary...")
        print("-" * 40)
        
        comparison_data = []
        
        # Collect all model results
        for category in ['arima', 'sarima', 'exponential_smoothing']:
            if category in results:
                for model_name, model_result in results[category].items():
                    comparison_data.append({
                        'Model': model_name,
                        'Category': category.replace('_', ' ').title(),
                        'AIC': model_result.aic,
                        'BIC': model_result.bic,
                        'In-Sample MAE': model_result.mae,
                        'In-Sample MSE': model_result.mse,
                        'In-Sample MAPE': model_result.mape,
                        'Forecast MAE': model_result.forecast_accuracy.get('forecast_mae', np.nan),
                        'Forecast RMSE': model_result.forecast_accuracy.get('forecast_rmse', np.nan),
                        'Forecast MAPE': model_result.forecast_accuracy.get('forecast_mape', np.nan),
                        'Ljung-Box p-value': model_result.ljung_box_pvalue,
                        'Residual Normality p-value': model_result.residual_normality_pvalue
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            # Sort by forecast accuracy (handle NaN values)
            comparison_df = comparison_df.sort_values('Forecast MAE')
            
            print("\nTop 5 Models by Forecast Accuracy:")
            print(comparison_df[['Model', 'Category', 'AIC', 'BIC', 'Forecast MAE', 'Forecast MAPE']].head())
        
        return comparison_df


def main():
    """Demonstrate comprehensive time series analysis."""
    # Load sample data
    try:
        df = pd.read_csv('../../data/raw/ohrid_synthetic_water_demand.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Extract water demand series
        series = df['water_demand_m3_per_hour'].dropna()
        
        print(f"Loaded data: {len(series)} observations")
        print(f"Date range: {series.index[0]} to {series.index[-1]}")
        
        # Initialize analyzer
        analyzer = TimeSeriesAnalyzer()
        
        # Run comprehensive analysis
        results = analyzer.comprehensive_analysis(series, seasonal_period=24)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Results available in analyzer.results")
        print("Access via: analyzer.results['comparison'] for model comparison")
        
    except FileNotFoundError:
        print("Sample data not found. Please ensure the data file exists.")
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()