"""
Ultron - shared geo helpers used by the time & weather tools.

Geocoding comes from the free, keyless Open-Meteo geocoding API. When the
user does not name a city, the public IP is geolocated (keyless ip-api.com)
so the answer reflects the user's actual location.
"""
import requests

from utils.logger import get_logger

log = get_logger("tools")

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_IP_URL = "http://ip-api.com/json/"
_TIMEOUT = 12


def geocode(city):
    """Return {name, latitude, longitude, timezone} for a city, or None."""
    try:
        resp = requests.get(
            _GEOCODE_URL,
            params={"name": city, "count": 1, "language": "en"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return None
        top = results[0]
        return {
            "name": top.get("name") or city,
            "latitude": top["latitude"],
            "longitude": top["longitude"],
            "timezone": top.get("timezone"),
        }
    except Exception as exc:  # network hiccup, rate limit, ...
        log.warning("Geocoding failed for %r: %s", city, exc)
        return None


def geoip():
    """Return {name, latitude, longitude, timezone} from the public IP."""
    try:
        resp = requests.get(_IP_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            return None
        return {
            "name": data.get("city") or "your location",
            "latitude": data.get("lat"),
            "longitude": data.get("lon"),
            "timezone": data.get("timezone"),
        }
    except Exception as exc:
        log.warning("IP geolocation failed: %s", exc)
        return None
