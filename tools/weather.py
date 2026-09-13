"""
Ultron - weather tool.

Current conditions for any city using the free, keyless Open-Meteo API.
If no city is given, the public IP is geolocated first.
"""
import requests

from tools.geo import geocode, geoip
from utils.logger import get_logger

log = get_logger("tools.weather")

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 10

# WMO weather interpretation codes -> human text.
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Dense drizzle",
    56: "Freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Heavy freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light showers",
    81: "Showers",
    82: "Violent showers",
    85: "Light snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with light hail",
    99: "Thunderstorm with heavy hail",
}


def _current_weather(latitude, longitude):
    resp = requests.get(
        _FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,is_day",
            "timezone": "auto",
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def weather_now(city=None, default_city=None):
    """Return a formatted current-weather summary for a city/location."""
    name, loc = city or default_city, None
    if name:
        loc = geocode(name)
        if not loc:
            return f"I couldn't find a city matching **{name}**. Please check the spelling."
    else:
        loc = geoip()
        if not loc:
            return (
                "I couldn't determine your location automatically. "
                "Please name a city, e.g. *what's the weather in Tokyo?*"
            )

    try:
        data = _current_weather(loc["latitude"], loc["longitude"])
        current = data.get("current", {})
        if not current:
            return f"Sorry, no current weather data is available for {loc['name']}."
    except requests.exceptions.Timeout:
        log.warning("Weather fetch timed out for %s", loc["name"])
        return f"Sorry, the weather service timed out for {loc['name']}."
    except Exception as exc:
        log.warning("Weather fetch failed for %s: %s", loc["name"], exc)
        return f"Sorry, the weather service is temporarily unavailable. ({exc})"

    temp = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")
    code = current.get("weather_code", 0)
    time = (current.get("time") or "").replace("T", " at ")

    lines = [
        f"**Current weather in {loc['name']}**",
        f"- Conditions: {WEATHER_CODES.get(code, 'Unknown')}",
        f"- Temperature: {temp}°C (feels like {feels}°C)" if temp is not None else "",
        f"- Humidity: {humidity}%" if humidity is not None else "",
        f"- Wind: {wind} km/h" if wind is not None else "",
        f"- Updated: {time}" if time else "",
    ]
    return "\n".join(line for line in lines if line)
