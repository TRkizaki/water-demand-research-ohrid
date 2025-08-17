#!/usr/bin/env python3
"""
Comprehensive OpenWeatherMap API Testing for Water Demand Research
Location: Ohrid, North Macedonia
Purpose: Academic submission evidence for professor review
"""

import os
import json
import requests
import time
from datetime import datetime, timedelta
import pandas as pd

class WeatherAPITester:
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.onecall_url = "http://api.openweathermap.org/data/3.0/onecall"
        
        # Ohrid coordinates
        self.lat = 41.1175
        self.lon = 20.8016
        self.location_name = "Ohrid, North Macedonia"
        
        self.test_results = []
        
    def log_test(self, test_name, status, details, data=None):
        """Log test results for evidence report"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'test_name': test_name,
            'status': status,
            'details': details,
            'data_sample': data
        }
        self.test_results.append(result)
        print(f"{'✅' if status == 'SUCCESS' else '❌'} {test_name}: {details}")
        
    def test_current_weather(self):
        """Test Current Weather Data API"""
        print(f"\nTesting Current Weather for {self.location_name}")
        print("-" * 60)
        
        url = f"{self.base_url}/weather"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract key information
                weather_info = {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'description': data['weather'][0]['description'],
                    'wind_speed': data.get('wind', {}).get('speed', 0),
                    'clouds': data.get('clouds', {}).get('all', 0),
                    'visibility': data.get('visibility', 0) / 1000,  # Convert to km
                    'dt': datetime.fromtimestamp(data['dt']).isoformat()
                }
                
                self.log_test(
                    "Current Weather API",
                    "SUCCESS",
                    f"Temperature: {weather_info['temperature']}°C, Humidity: {weather_info['humidity']}%, {weather_info['description']}",
                    weather_info
                )
                
                return weather_info
                
            else:
                self.log_test(
                    "Current Weather API",
                    "FAILED",
                    f"HTTP {response.status_code}: {response.text}"
                )
                return None
                
        except Exception as e:
            self.log_test(
                "Current Weather API",
                "ERROR",
                f"Exception: {str(e)}"
            )
            return None
    
    def test_forecast_weather(self):
        """Test 5-day Weather Forecast API"""
        print(f"\nTesting 5-Day Forecast for {self.location_name}")
        print("-" * 60)
        
        url = f"{self.base_url}/forecast"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Process forecast data
                forecasts = []
                for item in data['list'][:8]:  # First 24 hours (3-hour intervals)
                    forecast = {
                        'datetime': datetime.fromtimestamp(item['dt']).isoformat(),
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'pressure': item['main']['pressure'],
                        'description': item['weather'][0]['description'],
                        'wind_speed': item.get('wind', {}).get('speed', 0),
                        'precipitation': item.get('rain', {}).get('3h', 0) + item.get('snow', {}).get('3h', 0)
                    }
                    forecasts.append(forecast)
                
                self.log_test(
                    "5-Day Forecast API",
                    "SUCCESS",
                    f"Retrieved {len(data['list'])} forecast points, showing next 24h",
                    forecasts
                )
                
                return forecasts
                
            else:
                self.log_test(
                    "5-Day Forecast API",
                    "FAILED",
                    f"HTTP {response.status_code}: {response.text}"
                )
                return None
                
        except Exception as e:
            self.log_test(
                "5-Day Forecast API",
                "ERROR",
                f"Exception: {str(e)}"
            )
            return None
    
    def test_air_quality(self):
        """Test Air Pollution API"""
        print(f"\nTesting Air Quality for {self.location_name}")
        print("-" * 60)
        
        url = f"{self.base_url}/air_pollution"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                air_quality = {
                    'aqi': data['list'][0]['main']['aqi'],
                    'co': data['list'][0]['components']['co'],
                    'no2': data['list'][0]['components']['no2'],
                    'o3': data['list'][0]['components']['o3'],
                    'pm2_5': data['list'][0]['components']['pm2_5'],
                    'pm10': data['list'][0]['components']['pm10'],
                    'datetime': datetime.fromtimestamp(data['list'][0]['dt']).isoformat()
                }
                
                aqi_levels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
                
                self.log_test(
                    "Air Quality API",
                    "SUCCESS",
                    f"AQI: {air_quality['aqi']} ({aqi_levels.get(air_quality['aqi'], 'Unknown')}), PM2.5: {air_quality['pm2_5']} μg/m³",
                    air_quality
                )
                
                return air_quality
                
            else:
                self.log_test(
                    "Air Quality API",
                    "FAILED",
                    f"HTTP {response.status_code}: {response.text}"
                )
                return None
                
        except Exception as e:
            self.log_test(
                "Air Quality API",
                "ERROR",
                f"Exception: {str(e)}"
            )
            return None
    
    def test_geocoding(self):
        """Test Geocoding API"""
        print(f"\nTesting Geocoding for {self.location_name}")
        print("-" * 60)
        
        url = f"http://api.openweathermap.org/geo/1.0/direct"
        params = {
            'q': f"Ohrid,MK",
            'limit': 1,
            'appid': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data:
                    location = data[0]
                    geo_info = {
                        'name': location['name'],
                        'country': location['country'],
                        'state': location.get('state', 'N/A'),
                        'lat': location['lat'],
                        'lon': location['lon']
                    }
                    
                    self.log_test(
                        "Geocoding API",
                        "SUCCESS",
                        f"Found: {geo_info['name']}, {geo_info['country']} at ({geo_info['lat']}, {geo_info['lon']})",
                        geo_info
                    )
                    
                    return geo_info
                else:
                    self.log_test(
                        "Geocoding API",
                        "FAILED",
                        "No location data found"
                    )
                    return None
                    
            else:
                self.log_test(
                    "Geocoding API",
                    "FAILED",
                    f"HTTP {response.status_code}: {response.text}"
                )
                return None
                
        except Exception as e:
            self.log_test(
                "Geocoding API",
                "ERROR",
                f"Exception: {str(e)}"
            )
            return None
    
    def test_api_limits(self):
        """Test API rate limits and response times"""
        print(f"\nTesting API Performance and Limits")
        print("-" * 60)
        
        url = f"{self.base_url}/weather"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        response_times = []
        
        try:
            # Test 5 consecutive calls
            for i in range(5):
                start_time = time.time()
                response = requests.get(url, params=params, timeout=10)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # Convert to ms
                response_times.append(response_time)
                
                if response.status_code != 200:
                    break
                
                time.sleep(0.1)  # Small delay between requests
            
            avg_response_time = sum(response_times) / len(response_times)
            
            performance_info = {
                'total_requests': len(response_times),
                'avg_response_time_ms': round(avg_response_time, 2),
                'min_response_time_ms': round(min(response_times), 2),
                'max_response_time_ms': round(max(response_times), 2),
                'all_response_times': [round(rt, 2) for rt in response_times]
            }
            
            self.log_test(
                "API Performance Test",
                "SUCCESS",
                f"Avg response time: {performance_info['avg_response_time_ms']}ms, {performance_info['total_requests']} requests successful",
                performance_info
            )
            
            return performance_info
            
        except Exception as e:
            self.log_test(
                "API Performance Test",
                "ERROR",
                f"Exception: {str(e)}"
            )
            return None
    
    def generate_evidence_report(self):
        """Generate comprehensive evidence report for professor"""
        print(f"\nGenerating Evidence Report")
        print("=" * 60)
        
        report = {
            'report_metadata': {
                'title': 'OpenWeatherMap API Testing Evidence',
                'project': 'Water Demand Research - Ohrid, North Macedonia',
                'student': 'Water Demand Research Team',
                'generated_at': datetime.now().isoformat(),
                'api_key_last_4_digits': self.api_key[-4:] if self.api_key else 'Not Available',
                'test_location': self.location_name,
                'coordinates': f"{self.lat}, {self.lon}"
            },
            'api_capabilities_tested': [
                'Current Weather Data',
                '5-Day Weather Forecast',
                'Air Quality Index',
                'Geocoding',
                'API Performance'
            ],
            'test_results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'successful_tests': len([r for r in self.test_results if r['status'] == 'SUCCESS']),
                'failed_tests': len([r for r in self.test_results if r['status'] in ['FAILED', 'ERROR']])
            }
        }
        
        # Add success rate
        if report['summary']['total_tests'] > 0:
            success_rate = (report['summary']['successful_tests'] / report['summary']['total_tests']) * 100
            report['summary']['success_rate_percent'] = round(success_rate, 1)
        
        # Save detailed report
        with open('weather_api_evidence_report.json', 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def print_summary(self, report):
        """Print executive summary for quick review"""
        print(f"\nEXECUTIVE SUMMARY")
        print("=" * 60)
        print(f"Project: {report['report_metadata']['project']}")
        print(f"Location: {report['report_metadata']['test_location']}")
        print(f"API Key: ****{report['report_metadata']['api_key_last_4_digits']}")
        print(f"Test Date: {report['report_metadata']['generated_at'][:19]}")
        print(f"\nResults:")
        print(f"Successful Tests: {report['summary']['successful_tests']}")
        print(f"Failed Tests: {report['summary']['failed_tests']}")
        print(f"Success Rate: {report['summary'].get('success_rate_percent', 0)}%")
        
        print(f"\nAPIs Successfully Tested:")
        for result in self.test_results:
            if result['status'] == 'SUCCESS':
                print(f"  - {result['test_name']}")
        
        print(f"\nEvidence saved to: weather_api_evidence_report.json")
        print(f"Ready for professor submission!")

def main():
    print("COMPREHENSIVE WEATHER API TESTING")
    print("=" * 60)
    print("Project: Water Demand Prediction Research")
    print("Location: Ohrid, North Macedonia")
    print("Purpose: Academic Evidence Generation")
    print("=" * 60)
    
    tester = WeatherAPITester()
    
    if not tester.api_key:
        print("ERROR: OPENWEATHER_API_KEY not found in environment variables")
        print("Please set: export OPENWEATHER_API_KEY='your_api_key'")
        return None
    
    # Run all tests
    tester.test_current_weather()
    tester.test_forecast_weather()
    tester.test_air_quality()
    tester.test_geocoding()
    tester.test_api_limits()
    
    # Generate evidence report
    report = tester.generate_evidence_report()
    tester.print_summary(report)
    
    return report

if __name__ == "__main__":
    report = main()