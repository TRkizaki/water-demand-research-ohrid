"""
Temporal Feature Engineering for Ohrid Water Demand Research

Creates time-based features for water demand prediction including:
- Cyclical encoding of time components
- Holiday and special event detection
- Peak period identification
- Seasonal decomposition
"""

import pandas as pd
import numpy as np
from datetime import datetime
import holidays
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class TemporalFeatureEngineer:
    """Create temporal features for water demand prediction."""
    
    def __init__(self, location: str = "North Macedonia"):
        self.location = location
        self.holidays = holidays.NorthMacedonia()
        
        # Define peak hours for Ohrid
        self.morning_peak = (6, 8)   # 6-8 AM
        self.evening_peak = (18, 21) # 6-9 PM
        self.night_minimum = (23, 5) # 11 PM - 5 AM
        
        # Festival dates (Ohrid Summer Festival)
        self.festival_months = [7, 8]  # July-August
        
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical encoding for temporal features."""
        df_copy = df.copy()
        
        # Hour cyclical encoding (24-hour cycle)
        df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
        df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
        
        # Day of week cyclical encoding (7-day cycle)
        df_copy['day_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['day_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        
        # Month cyclical encoding (12-month cycle)
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        
        # Day of year cyclical encoding (365-day cycle)
        day_of_year = df_copy.index.dayofyear
        df_copy['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        df_copy['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        
        return df_copy
    
    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar-based features."""
        df_copy = df.copy()
        
        # Basic calendar features
        df_copy['year'] = df_copy.index.year
        df_copy['quarter'] = df_copy.index.quarter
        df_copy['week_of_year'] = df_copy.index.isocalendar().week
        df_copy['day_of_month'] = df_copy.index.day
        
        # Season encoding
        df_copy['season'] = df_copy['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring', 
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        # One-hot encode seasons
        season_dummies = pd.get_dummies(df_copy['season'], prefix='season')
        df_copy = pd.concat([df_copy, season_dummies], axis=1)
        
        return df_copy
    
    def create_peak_hour_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify peak consumption periods."""
        df_copy = df.copy()
        
        # Morning peak (6-8 AM)
        df_copy['is_morning_peak'] = (
            (df_copy['hour'] >= self.morning_peak[0]) & 
            (df_copy['hour'] <= self.morning_peak[1])
        ).astype(int)
        
        # Evening peak (6-9 PM)
        df_copy['is_evening_peak'] = (
            (df_copy['hour'] >= self.evening_peak[0]) & 
            (df_copy['hour'] <= self.evening_peak[1])
        ).astype(int)
        
        # Night minimum (11 PM - 5 AM)
        df_copy['is_night_minimum'] = (
            (df_copy['hour'] >= self.night_minimum[0]) | 
            (df_copy['hour'] <= self.night_minimum[1])
        ).astype(int)
        
        # General peak indicator (morning OR evening)
        df_copy['is_peak_hour'] = (
            df_copy['is_morning_peak'] | df_copy['is_evening_peak']
        ).astype(int)
        
        # Work hours (8 AM - 6 PM)
        df_copy['is_work_hour'] = (
            (df_copy['hour'] >= 8) & (df_copy['hour'] <= 18)
        ).astype(int)
        
        return df_copy
    
    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create holiday and special event features."""
        df_copy = df.copy()
        
        # Standard holidays
        df_copy['is_holiday'] = df_copy.index.date.isin(self.holidays)
        
        # Holiday proximity features
        df_copy['days_since_holiday'] = self._days_since_holiday(df_copy.index)
        df_copy['days_until_holiday'] = self._days_until_holiday(df_copy.index)
        
        # Weekend features
        df_copy['is_friday'] = (df_copy['day_of_week'] == 4).astype(int)
        df_copy['is_saturday'] = (df_copy['day_of_week'] == 5).astype(int) 
        df_copy['is_sunday'] = (df_copy['day_of_week'] == 6).astype(int)
        
        # Extended weekend (Friday-Sunday)
        df_copy['is_extended_weekend'] = (
            df_copy['day_of_week'].isin([4, 5, 6])
        ).astype(int)
        
        # Holiday weekend combination
        df_copy['is_holiday_weekend'] = (
            df_copy['is_holiday'] & df_copy['is_weekend']
        ).astype(int)
        
        return df_copy
    
    def create_festival_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Ohrid-specific festival features."""
        df_copy = df.copy()
        
        # Ohrid Summer Festival (July-August)
        df_copy['is_summer_festival'] = (
            df_copy['month'].isin(self.festival_months)
        ).astype(int)
        
        # Festival intensity (peak in mid-July to mid-August)
        df_copy['festival_intensity'] = 0.0
        
        # Calculate festival intensity
        for idx, row in df_copy.iterrows():
            if row['month'] == 7:  # July
                if row['day_of_month'] >= 15:  # Mid-July onwards
                    df_copy.loc[idx, 'festival_intensity'] = 0.8
                else:
                    df_copy.loc[idx, 'festival_intensity'] = 0.3
            elif row['month'] == 8:  # August
                if row['day_of_month'] <= 15:  # Mid-August
                    df_copy.loc[idx, 'festival_intensity'] = 1.0
                else:
                    df_copy.loc[idx, 'festival_intensity'] = 0.6
        
        # Festival weekend boost
        df_copy['festival_weekend_boost'] = (
            df_copy['is_summer_festival'] & df_copy['is_weekend']
        ).astype(int)
        
        return df_copy
    
    def create_lag_features(self, df: pd.DataFrame, 
                          target_col: str = 'water_demand_m3_per_hour',
                          lag_periods: List[int] = [1, 2, 7, 14, 30]) -> pd.DataFrame:
        """Create lag features for temporal dependencies."""
        df_copy = df.copy()
        
        if target_col not in df_copy.columns:
            print(f"Warning: Target column '{target_col}' not found. Skipping lag features.")
            return df_copy
        
        # Create lag features
        for lag in lag_periods:
            df_copy[f'{target_col}_lag_{lag}h'] = df_copy[target_col].shift(lag)
        
        # Create difference features (change from previous periods)
        for lag in [1, 7, 24]:  # 1 hour, 1 day, 1 week
            if lag <= max(lag_periods):
                df_copy[f'{target_col}_diff_{lag}h'] = (
                    df_copy[target_col] - df_copy[target_col].shift(lag)
                )
        
        return df_copy
    
    def create_rolling_features(self, df: pd.DataFrame,
                              target_col: str = 'water_demand_m3_per_hour',
                              windows: List[int] = [24, 168, 720]) -> pd.DataFrame:
        """Create rolling window statistical features."""
        df_copy = df.copy()
        
        if target_col not in df_copy.columns:
            print(f"Warning: Target column '{target_col}' not found. Skipping rolling features.")
            return df_copy
        
        for window in windows:
            # Rolling statistics
            df_copy[f'{target_col}_rolling_mean_{window}h'] = (
                df_copy[target_col].rolling(window=window, min_periods=1).mean()
            )
            df_copy[f'{target_col}_rolling_std_{window}h'] = (
                df_copy[target_col].rolling(window=window, min_periods=1).std()
            )
            df_copy[f'{target_col}_rolling_min_{window}h'] = (
                df_copy[target_col].rolling(window=window, min_periods=1).min()
            )
            df_copy[f'{target_col}_rolling_max_{window}h'] = (
                df_copy[target_col].rolling(window=window, min_periods=1).max()
            )
            
            # Rolling percentiles
            df_copy[f'{target_col}_rolling_q25_{window}h'] = (
                df_copy[target_col].rolling(window=window, min_periods=1).quantile(0.25)
            )
            df_copy[f'{target_col}_rolling_q75_{window}h'] = (
                df_copy[target_col].rolling(window=window, min_periods=1).quantile(0.75)
            )
        
        return df_copy
    
    def create_all_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all temporal features in one function."""
        print("Creating temporal features...")
        
        # Apply all feature engineering functions
        df_features = self.create_cyclical_features(df)
        df_features = self.create_calendar_features(df_features)
        df_features = self.create_peak_hour_features(df_features)
        df_features = self.create_holiday_features(df_features)
        df_features = self.create_festival_features(df_features)
        df_features = self.create_lag_features(df_features)
        df_features = self.create_rolling_features(df_features)
        
        # Count new features
        original_cols = len(df.columns)
        new_cols = len(df_features.columns)
        print(f"Created {new_cols - original_cols} temporal features")
        
        return df_features
    
    def _days_since_holiday(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate days since last holiday."""
        days_since = pd.Series(index=dates, dtype='float64')
        
        for date in dates:
            days_back = 0
            check_date = date.date()
            
            while days_back <= 30:  # Look back up to 30 days
                if check_date in self.holidays:
                    days_since[date] = days_back
                    break
                check_date = check_date - pd.Timedelta(days=1)
                days_back += 1
            else:
                days_since[date] = 30  # Cap at 30 days
        
        return days_since
    
    def _days_until_holiday(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate days until next holiday."""
        days_until = pd.Series(index=dates, dtype='float64')
        
        for date in dates:
            days_forward = 0
            check_date = date.date()
            
            while days_forward <= 30:  # Look forward up to 30 days
                if check_date in self.holidays:
                    days_until[date] = days_forward
                    break
                check_date = check_date + pd.Timedelta(days=1)
                days_forward += 1
            else:
                days_until[date] = 30  # Cap at 30 days
        
        return days_until


def main():
    """Demonstrate temporal feature engineering."""
    # Load sample data
    try:
        df = pd.read_csv('../../data/raw/ohrid_synthetic_water_demand.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        print(f"Loaded data: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Initialize feature engineer
        feature_engineer = TemporalFeatureEngineer()
        
        # Create all temporal features
        df_with_features = feature_engineer.create_all_temporal_features(df)
        
        print(f"\nFinal dataset shape: {df_with_features.shape}")
        print(f"Total features: {len(df_with_features.columns)}")
        
        # Save features
        output_path = '../../data/features/temporal_features.csv'
        df_with_features.to_csv(output_path)
        print(f"Temporal features saved to: {output_path}")
        
        # Show feature summary
        temporal_cols = [col for col in df_with_features.columns 
                        if any(keyword in col for keyword in 
                              ['sin', 'cos', 'peak', 'holiday', 'festival', 'lag', 'rolling'])]
        
        print(f"\nTemporal features created ({len(temporal_cols)}):")
        for col in temporal_cols[:20]:  # Show first 20
            print(f"  {col}")
        if len(temporal_cols) > 20:
            print(f"  ... and {len(temporal_cols) - 20} more")
            
    except FileNotFoundError:
        print("Sample data not found. Please run data generation first.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()