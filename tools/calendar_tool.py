"""
Ultron - calendar tool.

Tells dates, renders a month calendar grid, resolves weekdays for any date,
and does date math (days until / between / from now). Fully local, no network.

Natural-language targets:
  * "show the calendar" / "calendar for march 2027"
  * "what day of the week is december 25 2026"
  * "how many days until christmas"
  * "how many days between march 1 and march 20 2027"
  * "what date is it in 45 days"
"""
import calendar as _cal
import re
from datetime import date, timedelta

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_MONTH_ABBR = {name[:3]: num for name, num in MONTHS.items()}

# Fixed-date holidays (month, day). Floating holidays (Diwali/Easter) are skipped.
HOLIDAYS = {
    "new year's day": (1, 1), "new year": (1, 1), "new years": (1, 1),
    "republic day": (1, 26),
    "valentine's day": (2, 14), "valentine": (2, 14),
    "international women's day": (3, 8), "women's day": (3, 8),
    "april fools' day": (4, 1), "april fool": (4, 1),
    "independence day": (8, 15),
    "halloween": (10, 31),
    "christmas day": (12, 25), "christmas": (12, 25),
    "new year's eve": (12, 31),
}

_WEEKDAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _match_month(text):
    """Return (month_num, match) for a month name found in text, else None."""
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", text, re.I)
    if not m:
        return None
    return _MONTH_ABBR.get(m.group(1)[:3].lower()), m


def _parse_explicit_date(text):
    """Extract a concrete date (year optional) from free text."""
    now = date.today()
    t = text.strip()

    # Numeric: 25/12/2026, 2026-12-25, 12-25-26 ...
    num = re.search(r"\b(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{2,4})\b", t)
    if num:
        a, b, c = num.group(1), num.group(2), num.group(3)
        for combo in ((a, b, c), (c, b, a), (c, a, b)):
            try:
                y, m, d = (int(x) for x in combo)
                if 1 <= m <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100:
                    return date(y, m, d)
            except ValueError:
                continue

    # "december 25, 2026" / "dec 25 2026"
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})\b", t, re.I)
    if m:
        return date(int(m.group(3)), _MONTH_ABBR[m.group(1)[:3].lower()], int(m.group(2)))

    # "25 december 2026" / "25th of december 2026"
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?(?: of)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[,]?\s+(\d{4})\b", t, re.I)
    if m:
        return date(int(m.group(3)), _MONTH_ABBR[m.group(2)[:3].lower()], int(m.group(1)))

    # Month + day without year -> next occurrence this/next year.
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b", t, re.I)
    if m:
        month, day = _MONTH_ABBR[m.group(1)[:3].lower()], int(m.group(2))
        candidate = date(now.year, month, day)
        if candidate < now:
            candidate = candidate.replace(year=now.year + 1)
        return candidate

    return None


def _match_holiday(text):
    """Return a (month, day) for a holiday named in text, else None."""
    low = text.lower()
    for name, md in HOLIDAYS.items():
        if re.search(r"\b" + re.escape(name) + r"\b", low):
            return md
    return None


def _next_occurrence(month, day):
    """The next occurrence of (month, day) at or after today."""
    now = date.today()
    candidate = date(now.year, month, day)
    if candidate < now:
        candidate = candidate.replace(year=now.year + 1)
    return candidate


def month_grid(month=None, year=None):
    """A Markdown table calendar grid (Monday-first)."""
    now = date.today()
    month = month or now.month
    year = year or now.year
    cal = _cal.Calendar(firstweekday=0)  # Monday first
    header = " | ".join(_WEEKDAY)
    rows = ["| " + header.replace(" | ", " | ") + " |", "|" + "---|" * 7]
    weeks = cal.monthdayscalendar(year, month)
    for week in weeks:
        cells = []
        for day in week:
            if day == 0:
                cells.append(" ")
            elif day == now.day and month == now.month and year == now.year:
                cells.append(f"**{day}**")
            else:
                cells.append(str(day))
        rows.append("| " + " | ".join(cells) + " |")
    return (
        f"**{_cal.month_name[month]} {year}**\n\n"
        + "\n".join(rows)
        + (f"\n\nToday: **{now:%A, %d %B %Y}**" if (month, year) == (now.month, now.year) else "")
    )


def calendar_answer(text):
    """Dispatch a calendar question and return a formatted Markdown answer."""
    now = date.today()
    q = text.strip().lower()

    # Month grid requests.
    if re.search(r"\bcalendar\b|\bshow\b.*\bmonth\b|\bthis month\b", q):
        m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\b", q)
        if m:
            return month_grid(_MONTH_ABBR[m.group(1)[:3].lower()], int(m.group(2)))
        m = _match_month(q)
        if m:
            month, _ = m
            return month_grid(month, now.year)
        return month_grid()

    # "how many days until X" / "how long until X" / "how many days are left in X"
    until = re.search(r"\b(?:how many days|how long|days)\s+until\s+(.+)", q)
    if not until:
        left = re.search(
            r"\b(?:how many days (?:are )?(?:left|remaining)|days (?:left|remaining))"
            r"(?: in| to| until| for| till| before)?\s+(.+)", q)
        if left:
            until = left
    if until:
        target_txt = until.group(1).strip().rstrip("?.")
        target_txt = re.sub(r"^the\s+", "", target_txt)
        holiday = _match_holiday(target_txt)
        if holiday:
            target = _next_occurrence(*holiday)
            label = f"{_cal.month_name[holiday[0]]} {holiday[1]}"
        else:
            target = _parse_explicit_date(target_txt)
            if not target:
                return f"I couldn't parse '{target_txt}' as a date or holiday."
            label = f"{target:%d %B %Y}"
        days = (target - now).days
        if days < 0:
            return f"**{label}** was **{-days} days** ago (**{target:%A, %d %B %Y}**)."
        return f"**{label}** is **{days} days** away (**{target:%A, %d %B %Y}**)."

    # "days between A and B"
    between = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)\s*$", q)
    if between:
        a = _parse_explicit_date(between.group(1))
        b = _parse_explicit_date(between.group(2))
        if a and b:
            diff = abs((b - a).days)
            return f"There are **{diff} days** between **{a:%d %B %Y}** and **{b:%d %B %Y}**."
        return "I couldn't parse both dates for that calculation."

    # "what date in N days/weeks/months" / "date from now"
    ahead = re.search(r"\bin\s+(\d+)\s+(day|days|week|weeks|month|months)\b", q)
    if ahead:
        n, unit = int(ahead.group(1)), ahead.group(2).rstrip("s")
        delta = {"day": n, "week": n * 7, "month": n * 30}.get(unit, n)
        target = now + timedelta(days=delta)
        return f"**{n} {unit}(s) from today** ({now:%A, %d %B %Y}) is **{target:%A, %d %B %Y}**."

    # "what day of the week is X" / "what day is X" -> weekday of a date.
    weekday_q = re.search(r"\bwhat (?:day of the week|day|weekday) (?:is|will|would)\s+(.+)", q)
    if weekday_q:
        target_txt = weekday_q.group(1).strip().rstrip("?.")
        explicit = _parse_explicit_date(target_txt)
        if explicit:
            return f"**{explicit:%d %B %Y}** falls on a **{_WEEKDAY[explicit.weekday()]}**."
        holiday = _match_holiday(target_txt)
        if holiday:
            target = _next_occurrence(*holiday)
            return f"**{_cal.month_name[holiday[0]]} {holiday[1]}** falls on a **{_WEEKDAY[target.weekday()]}** this year."

    # Default: today's date + weekday.
    return f"**Today is {now:%A, %d %B %Y}.**\n\n{month_grid()}"
