"""
Ohrid Water Demand Synthetic Data Generator

This module generates realistic synthetic water demand data for Ohrid, North Macedonia,
incorporating local characteristics like tourism patterns, climate, and cultural factors.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import holidays
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class OhridWaterDemandGenerator:
    """
    Generate synthetic water demand data tailored to Ohrid's characteristics:
    - Tourism seasonality (UNESCO World Heritage site)
    - Mediterranean climate patterns
    - Orthodox calendar holidays
    - Lake Ohrid as water source
    - Balkan cultural patterns
    """
    
    def __init__(self, config_path: str = "config/ohrid_config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.location = self.config['location']
        self.regional = self.config['regional_characteristics']
        self.synthetic = self.config['synthetic_data']
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Initialize holidays for North Macedonia
        self.mk_holidays = holidays.NorthMacedonia()
        
        # Tourism and festival data
        self.ohrid_summer_festival = self._get_festival_dates()
        
    def _get_festival_dates(self) -> List[Tuple[datetime, datetime]]:
        """Get Ohrid Summer Festival dates (typically July-August)."""
        festival_periods = []
        for year in range(2021, 2025):
            # Ohrid Summer Festival typically runs from early July to late August
            start_date = datetime(year, 7, 1)
            end_date = datetime(year, 8, 31)
            festival_periods.append((start_date, end_date))
        return festival_periods
    
    def _is_tourist_season(self, date: datetime) -> Tuple[bool, float]:
        """
        Determine if date is in tourist season and return multiplier.
        
        Returns:
            (is_tourist_season, tourism_multiplier)
        """
        month = date.month
        
        if month in self.regional['tourism']['peak_season']:
            return True, self.regional['tourism']['peak_multiplier']
        elif month in self.regional['tourism']['shoulder_season']:
            return True, 1.5
        else:
            return False, 1.0
    
    def _is_festival_period(self, date: datetime) -> bool:
        """Check if date falls during Ohrid Summer Festival."""
        for start_date, end_date in self.ohrid_summer_festival:
            if start_date <= date <= end_date:
                return True
        return False
    
    def _get_base_demand(self, population: int, tourists: int = 0) -> float:
        """
        Calculate base water demand in cubic meters per hour.
        
        Args:
            population: Resident population
            tourists: Number of tourists
            
        Returns:
            Base demand in m³/hour
        """
        # Convert liters to cubic meters
        residential = (population * self.synthetic['base_consumption']['residential']) / 1000
        commercial = (population * self.synthetic['base_consumption']['commercial']) / 1000
        industrial = (population * self.synthetic['base_consumption']['industrial']) / 1000
        tourism_demand = (tourists * self.synthetic['base_consumption']['tourism']) / 1000
        
        # Convert daily to hourly (divide by 24)
        daily_total = residential + commercial + industrial + tourism_demand
        return daily_total / 24
    
    def _get_seasonal_multiplier(self, date: datetime) -> float:
        """Get seasonal consumption multiplier."""
        month = date.month
        
        # Winter months (Dec, Jan, Feb)
        if month in [12, 1, 2]:
            return self.synthetic['seasonal_patterns']['winter_reduction']
        # Summer months (Jun, Jul, Aug)
        elif month in [6, 7, 8]:
            return self.synthetic['seasonal_patterns']['summer_increase']
        # Spring/Fall
        else:
            return 1.0
    
    def _get_daily_pattern_multiplier(self, hour: int) -> float:
        """
        Get hourly consumption pattern multiplier.
        Based on typical Mediterranean/Balkan consumption patterns.
        """
        # Early morning low (1-5 AM)
        if 1 <= hour <= 5:
            return 0.4
        # Morning rise (6-8 AM)
        elif 6 <= hour <= 8:
            return 1.3 + 0.2 * np.sin(np.pi * (hour - 6) / 3)
        # Morning peak plateau (9-11 AM)
        elif 9 <= hour <= 11:
            return 1.1
        # Midday (12-2 PM)
        elif 12 <= hour <= 14:
            return 1.2
        # Afternoon low (3-5 PM)
        elif 15 <= hour <= 17:
            return 0.9
        # Evening peak (6-9 PM)
        elif 18 <= hour <= 21:
            return 1.4 + 0.3 * np.sin(np.pi * (hour - 18) / 4)
        # Night decline (10 PM - midnight)
        elif 22 <= hour <= 23 or hour == 0:
            return 0.7
        else:
            return 1.0
    
    def _get_weather_impact(self, temperature: float, precipitation: float) -> float:
        """
        Calculate weather impact on water demand.
        
        Args:
            temperature: Temperature in Celsius
            precipitation: Precipitation in mm/hour
            
        Returns:
            Weather multiplier
        """
        # Temperature impact
        temp_impact = 1.0
        if temperature > self.synthetic['weather_sensitivity']['temperature_threshold']:
            excess_temp = temperature - self.synthetic['weather_sensitivity']['temperature_threshold']
            temp_impact = 1 + (excess_temp * self.synthetic['weather_sensitivity']['temp_elasticity'])
        
        # Precipitation impact (reduces outdoor water use)
        rain_impact = 1.0
        if precipitation > 0.1:  # More than 0.1 mm/hour
            rain_impact = 1 - self.synthetic['weather_sensitivity']['rain_reduction']
        
        return temp_impact * rain_impact
    
    def _estimate_tourist_numbers(self, date: datetime) -> int:
        """
        Estimate number of tourists based on season and events.
        
        Ohrid typically sees:
        - Peak summer: 5,000-8,000 tourists daily
        - Shoulder season: 1,000-2,000 tourists daily
        - Off season: 200-500 tourists daily
        """
        is_tourist, multiplier = self._is_tourist_season(date)
        is_festival = self._is_festival_period(date)
        is_weekend = date.weekday() >= 5
        
        base_tourists = 300  # Off-season baseline
        
        if is_tourist:
            if multiplier > 2.0:  # Peak season
                base_tourists = np.random.randint(5000, 8000)
            else:  # Shoulder season
                base_tourists = np.random.randint(1000, 2000)
        
        # Festival boost
        if is_festival:
            base_tourists = int(base_tourists * 1.3)
        
        # Weekend boost
        if is_weekend and is_tourist:
            base_tourists = int(base_tourists * 1.2)
        
        return base_tourists
    
    def _generate_weather_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate realistic weather data for Ohrid.
        Based on Mediterranean climate with some continental influence.
        """
        dates = pd.date_range(start_date, end_date, freq='h')
        n_hours = len(dates)
        
        weather_data = []
        
        for i, dt in enumerate(dates):
            # Base temperature with seasonal variation
            day_of_year = dt.timetuple().tm_yday
            seasonal_temp = 12 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            # Daily temperature variation
            hour_of_day = dt.hour
            daily_variation = 4 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            
            # Add some noise
            temperature = seasonal_temp + daily_variation + np.random.normal(0, 2)
            
            # Humidity (higher in winter, lower in summer)
            base_humidity = 65 - 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            humidity = max(30, min(90, base_humidity + np.random.normal(0, 5)))
            
            # Precipitation (more in autumn/winter)
            precip_probability = 0.15 + 0.1 * np.sin(2 * np.pi * (day_of_year - 200) / 365)
            if np.random.random() < precip_probability:
                precipitation = np.random.exponential(2)
            else:
                precipitation = 0
            
            # Wind speed (typically light in Ohrid valley)
            wind_speed = max(0, np.random.gamma(2, 2))
            
            # Pressure
            pressure = 1013 + np.random.normal(0, 8)
            
            weather_data.append({
                'timestamp': dt,
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'precipitation': round(precipitation, 2),
                'wind_speed': round(wind_speed, 1),
                'pressure': round(pressure, 1),
                'cloud_cover': np.random.randint(0, 101)
            })
        
        return pd.DataFrame(weather_data)
    
    def generate_synthetic_data(self, 
                              start_date: str = "2021-01-01", 
                              end_date: str = "2023-12-31",
                              frequency: str = "1h") -> pd.DataFrame:
        """
        Generate comprehensive synthetic water demand data for Ohrid.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency ('1h' for hourly, '1d' for daily)
            
        Returns:
            DataFrame with synthetic water demand and feature data
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Generate timestamps
        timestamps = pd.date_range(start_dt, end_dt, freq=frequency)
        
        # Generate weather data
        weather_df = self._generate_weather_data(start_dt, end_dt)
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            # Get corresponding weather data
            weather_row = weather_df[weather_df['timestamp'] == timestamp].iloc[0]
            
            # Basic temporal features
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            month = timestamp.month
            is_weekend = day_of_week >= 5
            is_holiday = timestamp.date() in self.mk_holidays
            
            # Tourism and events
            tourists = self._estimate_tourist_numbers(timestamp)
            is_festival = self._is_festival_period(timestamp)
            is_tourist_season, tourism_multiplier = self._is_tourist_season(timestamp)
            
            # Calculate base demand
            population = self.location['population']
            base_demand = self._get_base_demand(population, tourists)
            
            # Apply multipliers
            seasonal_mult = self._get_seasonal_multiplier(timestamp)
            daily_mult = self._get_daily_pattern_multiplier(hour)
            weather_mult = self._get_weather_impact(
                weather_row['temperature'], 
                weather_row['precipitation']
            )
            
            # Weekend effect (slightly lower consumption)
            weekend_mult = 0.95 if is_weekend else 1.0
            
            # Holiday effect
            holiday_mult = 1.1 if is_holiday else 1.0
            
            # Festival effect
            festival_mult = 1.2 if is_festival else 1.0
            
            # Calculate final demand
            total_multiplier = (seasonal_mult * daily_mult * weather_mult * 
                              weekend_mult * holiday_mult * festival_mult)
            
            water_demand = base_demand * total_multiplier
            
            # Add some realistic noise (±5%)
            noise_factor = np.random.normal(1.0, 0.05)
            water_demand *= noise_factor
            
            # Ensure positive values
            water_demand = max(water_demand, base_demand * 0.3)
            
            # Simulate network losses
            actual_production = water_demand / (1 - self.regional['infrastructure']['leakage_rate'])
            
            data.append({
                'timestamp': timestamp,
                'water_demand_m3_per_hour': round(water_demand, 2),
                'water_production_m3_per_hour': round(actual_production, 2),
                'population': population,
                'tourists_estimated': tourists,
                'temperature': weather_row['temperature'],
                'humidity': weather_row['humidity'],
                'precipitation': weather_row['precipitation'],
                'wind_speed': weather_row['wind_speed'],
                'pressure': weather_row['pressure'],
                'cloud_cover': weather_row['cloud_cover'],
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'is_weekend': is_weekend,
                'is_holiday': is_holiday,
                'is_tourist_season': is_tourist_season,
                'is_festival_period': is_festival,
                'seasonal_multiplier': round(seasonal_mult, 3),
                'daily_multiplier': round(daily_mult, 3),
                'weather_multiplier': round(weather_mult, 3),
                'tourism_multiplier': round(tourism_multiplier, 3)
            })
        
        df = pd.DataFrame(data)
        
        # Add lag features
        df = self._add_lag_features(df)
        
        # Add rolling features
        df = self._add_rolling_features(df)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for water demand."""
        lag_periods = [1, 2, 7, 24, 168]  # 1h, 2h, 7h, 1day, 1week (for hourly data)
        
        for lag in lag_periods:
            df[f'demand_lag_{lag}h'] = df['water_demand_m3_per_hour'].shift(lag)
            
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features."""
        windows = [24, 168, 720]  # 1 day, 1 week, 1 month (hours)
        
        for window in windows:
            df[f'demand_rolling_mean_{window}h'] = (
                df['water_demand_m3_per_hour'].rolling(window=window).mean()
            )
            df[f'demand_rolling_std_{window}h'] = (
                df['water_demand_m3_per_hour'].rolling(window=window).std()
            )
            df[f'demand_rolling_min_{window}h'] = (
                df['water_demand_m3_per_hour'].rolling(window=window).min()
            )
            df[f'demand_rolling_max_{window}h'] = (
                df['water_demand_m3_per_hour'].rolling(window=window).max()
            )
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save generated data to CSV file."""
        df.to_csv(filename, index=False)
        print(f"Synthetic data saved to {filename}")
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Average demand: {df['water_demand_m3_per_hour'].mean():.2f} m³/hour")
        print(f"Peak demand: {df['water_demand_m3_per_hour'].max():.2f} m³/hour")


def main():
    """Generate and save synthetic data for Ohrid."""
    generator = OhridWaterDemandGenerator()
    
    # Generate 3 years of hourly data
    synthetic_data = generator.generate_synthetic_data(
        start_date="2021-01-01",
        end_date="2023-12-31",
        frequency="1h"
    )
    
    # Save the data
    generator.save_data(synthetic_data, "data/raw/ohrid_synthetic_water_demand.csv")
    
    # Display basic statistics
    print("\n=== Data Summary ===")
    print(synthetic_data.describe())
    
    print("\n=== Sample Data ===")
    print(synthetic_data.head(10))


if __name__ == "__main__":
    main()