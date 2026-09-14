"""Ultron smoke tests — pure tool functions only (no network, no API keys).

Run from the repo root:  pytest
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_calculator_basic():
    from tools.calculator import evaluate_expression, detect_calculation
    assert evaluate_expression("25 * 4 + 10") == 110
    assert evaluate_expression("2 + 2") == 4
    assert evaluate_expression("(5 + 3) * 2") == 16
    assert detect_calculation("calculate 10 * 5")
    assert detect_calculation("what is 3 * 4")
    assert evaluate_expression("1 / 0") == "Division by zero"


def test_time_parse_both_word_orders():
    from tools.time_tool import parse_historical_time_query
    a = parse_historical_time_query(
        "What was the time in Tokyo when it was 5:00 pm in India on September 9 2026")
    assert a == {"source_time": "5:00 pm", "source_tz": "India",
                 "target_tz": "Tokyo", "source_date": "September 9 2026"}
    b = parse_historical_time_query(
        "what was the time in london at 6pm on 7 sept in india")
    assert b == {"source_time": "6pm", "source_tz": "india",
                 "target_tz": "london", "source_date": "7 sept"}


def test_time_convert_known_offsets():
    from tools.time_tool import convert_time
    # IST (UTC+5:30) -> JST (UTC+9): +3:30
    r = convert_time("5:00 pm", "India", "Tokyo", "September 9 2026")
    assert "08:30:00 PM" in r and "Asia/Tokyo" in r
    # IST -> BST (UTC+1 in September): -4:30
    r = convert_time("6pm", "india", "london", "7 sept")
    assert "01:30:00 PM" in r and "Europe/London" in r


def test_calendar_days_left_and_until():
    from tools.calendar_tool import calendar_answer
    r = calendar_answer("how many days are left in new year?")
    assert "January 1" in r and "days" in r
    r = calendar_answer("how many days until christmas")
    assert "December 25" in r and "days" in r
    r = calendar_answer("what day is december 25 2026")
    assert "Friday" in r


def test_router_classification():
    from tools.tool_manager import create_tool_manager
    tm = create_tool_manager({"tools_enabled": True, "web_results": 4, "default_city": ""})
    assert tm.classify_intent("calculate 2 + 2").primary_tool == "calculator"
    assert tm.classify_intent("what time is it in Tokyo").primary_tool == "time"
    assert tm.classify_intent("how many days are left in new year").primary_tool == "calendar"
    assert tm.classify_intent("FIFA 2026 winner").primary_tool == "web"
    assert tm.classify_intent("hello there friend").primary_tool == "chat"


def test_provider_wiring():
    from models.providers import (
        PROVIDERS, GroqProvider, CyfutureProvider,
        _OpenAICompatibleProvider, build_provider_order,
    )
    assert set(PROVIDERS) == {"cyfuture", "groq", "gemini"}
    assert issubclass(GroqProvider, _OpenAICompatibleProvider)
    assert issubclass(CyfutureProvider, _OpenAICompatibleProvider)
    assert build_provider_order({"provider": "auto"}) == ["groq", "gemini"]
    # Cyfuture only strips real thinking blocks, never normal prose.
    c = CyfutureProvider({"cyfuture_key": "x"})
    assert c._coalesce("Hello\n\nWorld") == "Hello\n\nWorld"
    assert c._coalesce("thinking...\n</thinking>\nFinal answer") == "Final answer"


def test_search_formatter_empty():
    from tools.websearch_providers import format_search_results
    assert "no results" in format_search_results([], "xyz").lower()
