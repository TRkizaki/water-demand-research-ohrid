#!/usr/bin/env python3
"""
Clean Weather API Testing Script
Tests OpenWeatherMap API integration without exposing credentials
"""

import os
import requests
from datetime import datetime

def test_weather_api():
    """Test OpenWeatherMap API with environment variables"""
    
    print("Weather API Testing for Ohrid Water Demand Research")
    print("=" * 55)
    
    # Get API key from environment
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("❌ OPENWEATHER_API_KEY not found in environment variables")
        print("Please set: export OPENWEATHER_API_KEY='your_api_key'")
        return False
    
    # Ohrid coordinates
    lat, lon = 41.1175, 20.8016
    location = "Ohrid, North Macedonia"
    
    print(f"Testing location: {location}")
    print(f"Coordinates: {lat}, {lon}")
    print(f"API Key: ****{api_key[-4:]}")
    print("-" * 55)
    
    # Test current weather
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Current Weather API: SUCCESS")
            print(f"   Temperature: {data['main']['temp']}°C")
            print(f"   Humidity: {data['main']['humidity']}%")
            print(f"   Description: {data['weather'][0]['description']}")
            return True
        else:
            print(f"❌ Current Weather API: FAILED ({response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ Weather API Error: {e}")
        return False

if __name__ == "__main__":
    success = test_weather_api()
    print(f"\nTest Result: {'PASSED' if success else 'FAILED'}")
    print("\nNote: This script requires OPENWEATHER_API_KEY environment variable")