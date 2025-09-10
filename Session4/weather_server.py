# weather_server.py
# pip install geopy

import requests
from mcp.server.fastmcp import FastMCP
from geopy.geocoders import Nominatim

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(latitude, longitude):
    """Get current temperature for provided coordinates (latitude and longitude) in celsius."""
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    res = {"temperature": data['current']['temperature_2m'], "unit":data['current_units']['temperature_2m']}
    return res

@mcp.tool()
def get_coordinates(place_name):
    """
    지명(location)을 입력받아 위도(latitude), 경도(longitude)를 반환하는 함수
    
    Args:
        place_name (str): 찾고자 하는 지명
        
    Returns:
        tuple: (위도, 경도) 또는 None
    """
    try:
        # 지오코더 초기화
        geolocator = Nominatim(user_agent="coordinate_finder")
        
        # 지명으로 위치 검색
        location = geolocator.geocode(place_name)
        
        if location:
            return (location.latitude, location.longitude)
        else:
            print(f"'{place_name}'을(를) 찾을 수 없습니다.")
            return None
            
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

if __name__ == "__main__":
    # mcp.run(transport="stdio")
    mcp.run(transport="streamable-http")