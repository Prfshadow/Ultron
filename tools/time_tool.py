"""
Ultron - time tool.

Returns the current date & time. With a city name it resolves the correct
timezone automatically (Open-Meteo geocoding + the stdlib `zoneinfo`).

Also supports historical timezone conversions: "what was the time in Tokyo
when it was 5pm in India on Sept 6".
"""
import re
import zoneinfo
from datetime import datetime

from tools.geo import geocode, geoip
from utils.logger import get_logger

log = get_logger("tools.time")


def get_time(city=None):
    """Return a human-readable current date & time string."""
    if city:
        loc = geocode(city)
        if loc and loc.get("timezone"):
            try:
                tz = zoneinfo.ZoneInfo(loc["timezone"])
                now = datetime.now(tz)
                return (
                    f"**Time in {loc['name']}**  \n"
                    f"Date: {now:%A, %d %B %Y}  \n"
                    f"Time: {now:%I:%M:%S %p} ({loc['timezone']})"
                )
            except Exception:
                log.warning("ZoneInfo failed for %s", loc.get("timezone"))
        return f"I couldn't resolve the timezone for '{city}'."

    loc = geoip()
    if loc and loc.get("timezone"):
        try:
            tz = zoneinfo.ZoneInfo(loc["timezone"])
            now = datetime.now(tz)
            return (
                f"**Local time ({loc['name']}, {loc['timezone']})**  \n"
                f"Date: {now:%A, %d %B %Y}  \n"
                f"Time: {now:%I:%M:%S %p}"
            )
        except Exception:
            pass

    now = datetime.now().astimezone()
    return (
        f"**Current date & time**  \n"
        f"Date: {now:%A, %d %B %Y}  \n"
        f"Time: {now:%I:%M:%S %p} (server local time)"
    )


def convert_time(source_time_str, source_tz_str, target_tz_str, source_date_str=None):
    """
    Convert a time from one timezone to another for a specific date.
    
    Args:
        source_time_str: Time string like "5pm", "17:00", "5:30pm"
        source_tz_str: Source timezone name (e.g., "Asia/Kolkata", "India")
        target_tz_str: Target timezone name (e.g., "Asia/Tokyo", "Tokyo")
        source_date_str: Date string like "sept 6", "2026-09-06", "today" (defaults to today)
    
    Returns:
        Formatted string with the converted time
    """
    import logging
    log = logging.getLogger("tools.time")
    log.info(f"[DEBUG time_tool] convert_time called from {__file__}")
    
    # Map common names to IANA timezone names
    tz_map = {
        "india": "Asia/Kolkata",
        "ist": "Asia/Kolkata",
        "tokyo": "Asia/Tokyo",
        "jst": "Asia/Tokyo",
        "london": "Europe/London",
        "bst": "Europe/London",
        "new york": "America/New_York",
        "est": "America/New_York",
        "edt": "America/New_York",
        "los angeles": "America/Los_Angeles",
        "pst": "America/Los_Angeles",
        "pdt": "America/Los_Angeles",
        "utc": "UTC",
        "gmt": "UTC",
    }
    
    source_tz_name = tz_map.get(source_tz_str.lower(), source_tz_str)
    target_tz_name = tz_map.get(target_tz_str.lower(), target_tz_str)
    
    try:
        source_tz = zoneinfo.ZoneInfo(source_tz_name)
    except Exception:
        return f"Unknown source timezone: {source_tz_str}"
    
    try:
        target_tz = zoneinfo.ZoneInfo(target_tz_name)
    except Exception:
        return f"Unknown target timezone: {target_tz_str}"
    
    # Parse source date
    if source_date_str:
        log.info(f"Parsing date: {repr(source_date_str)}")
        source_date_str = source_date_str.lower().strip()
        if source_date_str in ("today", "now"):
            source_date = datetime.now(source_tz).date()
        else:
            # Normalize month abbreviations (sept -> Sep, sept. -> Sep, etc.)
            import re
            month_map = {
                r"\bjan\b": "Jan", r"\bjanuary\b": "January",
                r"\bfeb\b": "Feb", r"\bfebruary\b": "February",
                r"\bmar\b": "Mar", r"\bmarch\b": "March",
                r"\bapr\b": "Apr", r"\bapril\b": "April",
                r"\bmay\b": "May",
                r"\bjun\b": "Jun", r"\bjune\b": "June",
                r"\bjul\b": "Jul", r"\bjuly\b": "July",
                r"\baug\b": "Aug", r"\baugust\b": "August",
                r"\bsep\b": "Sep", r"\bsept\b": "Sep", r"\bseptember\b": "September",
                r"\boct\b": "Oct", r"\boctober\b": "October",
                r"\bnov\b": "Nov", r"\bnovember\b": "November",
                r"\bdec\b": "Dec", r"\bdecember\b": "December",
            }
            for pattern, replacement in month_map.items():
                source_date_str = re.sub(pattern, replacement, source_date_str, flags=re.IGNORECASE)
            
            log.info(f"After month_map: {repr(source_date_str)}")
            
            # Try multiple date formats
            date_formats = [
                "%b %d", "%B %d", "%d %b", "%d %B",  # "Sep 6", "September 6"
                "%b %d %Y", "%B %d %Y", "%d %b %Y", "%d %B %Y",  # "Dec 25 2025", "December 25 2025"
                "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",  # "2026-09-06"
                "%m-%d", "%d-%m",  # "09-06"
            ]
            source_date = None
            for fmt in date_formats:
                try:
                    # Assume current year if not provided
                    if fmt in ("%b %d", "%B %d", "%d %b", "%d %B", "%m-%d", "%d-%m"):
                        dt = datetime.strptime(source_date_str + f" {datetime.now().year}", fmt + " %Y")
                    else:
                        dt = datetime.strptime(source_date_str, fmt)
                    source_date = dt.date()
                    log.info(f"Parsed date with format {fmt}: {source_date}")
                    break
                except ValueError as e:
                    log.debug(f"Format {fmt} failed: {e}")
                    continue
            if source_date is None:
                log.error(f"Could not parse date: {source_date_str}")
                return f"Could not parse date: {source_date_str}"
    else:
        source_date = datetime.now(source_tz).date()
    
    # Parse source time
    time_formats = [
        "%I:%M%p", "%I%p", "%H:%M", "%H%M",  # "5:30pm", "5pm", "17:30", "1730"
        "%I:%M %p", "%I %p",  # "5:30 pm", "5 pm"
    ]
    source_time = None
    source_time_str = source_time_str.lower().replace(".", "").strip()
    for fmt in time_formats:
        try:
            source_time = datetime.strptime(source_time_str, fmt).time()
            break
        except ValueError:
            continue
    
    if source_time is None:
        return f"Could not parse time: {source_time_str}"
    
    # Create source datetime in source timezone
    source_dt = datetime.combine(source_date, source_time)
    source_dt = source_dt.replace(tzinfo=source_tz)
    
    # Convert to target timezone
    target_dt = source_dt.astimezone(target_tz)
    
    # Format output
    return (
        f"**Time Conversion**  \n"
        f"Source: {source_dt:%A, %d %B %Y at %I:%M:%S %p} ({source_tz_name})  \n"
        f"Target: {target_dt:%A, %d %B %Y at %I:%M:%S %p} ({target_tz_name})"
    )


def parse_historical_time_query(query):
    """
    Parse queries like "what was the time in Tokyo when it was 5pm in India on sept 6"
    Returns (source_time, source_tz, target_tz, source_date) or None
    """
    # Each entry: (regex, (group1_name, group2_name, ...)).
    # Two word orders exist: "... in <source> on <date>" and "... on <date> in <source>".
    _STD = ("target_tz", "source_time", "source_tz", "source_date")
    _SWAPPED = ("target_tz", "source_time", "source_date", "source_tz")
    patterns = [
        # With explicit "on <date>" — source before date
        (r"what was the time in\s+([^,\n]+?)\s+when it was\s+([^,\n]+?)\s+in\s+([^,\n]+?)\s+on\s+([^,\n?]+)", _STD),
        (r"what time was it in\s+([^,\n]+?)\s+when it was\s+([^,\n]+?)\s+in\s+([^,\n]+?)\s+on\s+([^,\n?]+)", _STD),
        (r"time in\s+([^,\n]+?)\s+when\s+([^,\n]+?)\s+in\s+([^,\n]+?)\s+on\s+([^,\n?]+)", _STD),
        (r"convert\s+([^,\n]+?)\s+in\s+([^,\n]+?)\s+to\s+([^,\n]+?)\s+on\s+([^,\n?]+)", ("target_tz", "source_time", "source_tz", "source_date")),
        # With "at" instead of "when it was" — source before date
        (r"what was the time in\s+([^,\n]+?)\s+at\s+([^,\n]+?)\s+in\s+([^,\n]+?)\s+on\s+([^,\n?]+)", _STD),
        # Date before source: "at 6pm on 7 sept in india"
        (r"what was the time in\s+([^,\n]+?)\s+at\s+([^,\n]+?)\s+on\s+([^,\n]+?)\s+in\s+([^,\n?]+)$", _SWAPPED),
        (r"what time was it in\s+([^,\n]+?)\s+at\s+([^,\n]+?)\s+on\s+([^,\n]+?)\s+in\s+([^,\n?]+)$", _SWAPPED),
        (r"time in\s+([^,\n]+?)\s+at\s+([^,\n]+?)\s+on\s+([^,\n]+?)\s+in\s+([^,\n?]+)$", _SWAPPED),
        # Without explicit date (default to today)
        (r"what was the time in\s+([^,\n]+?)\s+when it was\s+([^,\n]+?)\s+in\s+([^,\n?]+)$", ("target_tz", "source_time", "source_tz")),
        (r"what time was it in\s+([^,\n]+?)\s+when it was\s+([^,\n]+?)\s+in\s+([^,\n?]+)$", ("target_tz", "source_time", "source_tz")),
        (r"time in\s+([^,\n]+?)\s+when\s+([^,\n]+?)\s+in\s+([^,\n?]+)$", ("target_tz", "source_time", "source_tz")),
        (r"what was the time in\s+([^,\n]+?)\s+at\s+([^,\n]+?)\s+in\s+([^,\n?]+)$", ("target_tz", "source_time", "source_tz")),
    ]

    for pattern, order in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            parts = {name: g.strip() for name, g in zip(order, match.groups())}
            parts.setdefault("source_date", "today")
            return {
                "source_time": parts["source_time"],
                "source_tz": parts["source_tz"],
                "target_tz": parts["target_tz"],
                "source_date": parts["source_date"],
            }
    return None
