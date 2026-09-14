"""
Ultron - Tool Manager.

A unified routing layer that:
1. Classifies intent using LLM (with pattern fallback)
2. Routes to appropriate tools (time, weather, calendar, web, imagegen, code, analysis)
3. Selects optimal model per task (vision, reasoning, speed, coding)
4. Can chain multiple tools for complex queries
"""
import json
import re
from dataclasses import dataclass
from typing import Optional
from utils.logger import get_logger

log = get_logger("tools.manager")

# Import existing tools
from tools.calendar_tool import calendar_answer
from tools.time_tool import get_time, convert_time, parse_historical_time_query
from tools.weather import weather_now
from tools.websearch_providers import multi_search, format_search_results
from tools.calculator import evaluate_expression, detect_calculation


# --- Query helpers (moved from tools.router) ---
_WEB_PREFIXES = re.compile(
    r"^(?:"
    r"search (?:the )?(?:web|online|internet|google)(?: for| and)?|"
    r"(?:web|online) search(?: for)?|"
    r"look up(?: online)?(?: for)?|"
    r"google(?: it)?(?: for)?|"
    r"find (?:out|online|information|details)(?: about)?(?: for)?|"
    r"research(?: online)?(?: about)?|"
    r"what is|what are|what was|what were|"
    r"who is|who are|who was|"
    r"when was|when did|where is|where are"
    r")\s+"
)


def _web_query(text: str) -> str:
    """Derive a clean search-engine query from a user message."""
    cleaned = _WEB_PREFIXES.sub("", text.strip(), count=1).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned).rstrip("?.")
    if not cleaned:
        return text.strip()
    # Social handles become targeted queries ("@Onryx99 youtube subscribers count").
    handle = re.search(r"@([A-Za-z0-9_.-]{2,})", cleaned)
    if handle:
        name = handle.group(1)
        if "youtube" in cleaned.lower() or "subscriber" in cleaned.lower():
            return f"{name} youtube subscribers count"
        if "instagram" in cleaned.lower() or "follower" in cleaned.lower():
            return f"{name} instagram followers count"
        return f"{name} {cleaned}"
    return cleaned


def _extract_city(text: str, pattern: str) -> Optional[str]:
    """Pull the city out of phrases like 'time/weather in <city>'."""
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        city = m.group(1).strip().rstrip("?.")
        if city and not re.fullmatch(r"(right now|now|today|outside|there|at home|it)", city, re.I):
            return city
    return None


@dataclass
class ToolResult:
    """Standardized tool result."""
    tool: str
    success: bool
    data: any = None
    error: str = None
    direct_answer: str = None
    context_for_llm: str = None
    sources: list = None


@dataclass
class RouteDecision:
    """Decision from the router."""
    primary_tool: str
    secondary_tools: list
    model_preference: str
    reasoning: str
    requires_vision: bool
    requires_reasoning: bool
    requires_code: bool


class ToolManager:
    """Manages tool routing and execution."""

    TOOLS = {
        "time": {
            "description": "Current time, timezone conversions, historical time conversions",
            "patterns": ["time", "timezone", "what time", "what was the time", "time in.*when", "convert.*to.*on"],
            "model_pref": "fast",
        },
        "weather": {
            "description": "Weather forecasts, current conditions",
            "patterns": ["weather", "forecast", "temperature", "rain", "snow"],
            "model_pref": "fast",
        },
        "calendar": {
            "description": "Date calculations, calendar views, day of week",
            "patterns": ["calendar", "date", "day of week", "days until", "days left", "days remaining", "how many days", "new year", "christmas", "schedule", "what day", "what weekday", "day of the week"],
            "model_pref": "fast",
        },
        "calculator": {
            "description": "Mathematical calculations, arithmetic",
            "patterns": ["calculate", "compute", "what is", "what's", "how much is", "math", "\\d\\s*[+\\-*/]\\s*\\d"],
            "model_pref": "fast",
        },
        "web": {
            "description": "Web search, current events, live data, facts",
            "patterns": ["search", "latest", "news", "who is", "stock", "price", "population", "who won", "winner", "champion", "won the", "won.*cup", "won.*world", "result", "score", "final"],
            "model_pref": "reasoning",
        },
        "imagegen": {
            "description": "Generate images, artwork, illustrations",
            "patterns": ["generate image", "create picture", "draw", "render", "artwork"],
            "model_pref": "vision",
        },
        "code": {
            "description": "Code execution, debugging, algorithm help",
            "patterns": ["run code", "execute", "debug", "algorithm", "function", "script"],
            "model_pref": "coding",
        },
        "analysis": {
            "description": "Document analysis, data processing, summarization",
            "patterns": ["analyze", "summarize", "extract", "process document", "pdf"],
            "model_pref": "reasoning",
        },
    }

    MODEL_PREFERENCES = {
        "fast": ["groq", "gemini"],
        "reasoning": ["groq", "gemini"],
        "coding": ["groq", "gemini"],
        "vision": ["gemini", "groq"],
        "default": ["groq", "gemini"],
    }

    def __init__(self, settings: dict):
        self.settings = settings
        self.tools_enabled = settings.get("tools_enabled") in (True, "true", "1", 1)

    def classify_intent(self, query: str, has_images: bool = False, has_files: bool = False) -> RouteDecision:
        """Classify query intent using pattern matching (LLM fallback could be added)."""
        q = query.lower().strip()

        # Quick pattern-based classification
        scores = {}
        for tool_name, config in self.TOOLS.items():
            score = 0
            for pattern in config["patterns"]:
                if re.search(pattern, q, re.IGNORECASE):
                    score += 1
            scores[tool_name] = score

        # Boost based on context
        # Images are handled directly by vision-capable models, not via a tool
        if has_files:
            # Check if query looks like a document analysis query
            is_analysis_query = any(
                re.search(pattern, q, re.IGNORECASE)
                for pattern in self.TOOLS.get("analysis", {}).get("patterns", [])
            )
            if is_analysis_query:
                scores["analysis"] = scores.get("analysis", 0) + 2
            # Suppress web search when files are attached - RAG should handle document queries
            scores["web"] = max(0, scores.get("web", 0) - 2)

        # Determine primary tool
        primary = max(scores, key=scores.get) if scores else "chat"
        if scores.get(primary, 0) == 0:
            primary = "chat"

        # Determine secondary tools (for chaining)
        secondary = [t for t, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0 and t != primary][:2]

        # Model preference - if images present, prefer vision models
        model_pref = self.TOOLS.get(primary, {}).get("model_pref", "default")
        if has_images:
            model_pref = "vision"

        return RouteDecision(
            primary_tool=primary,
            secondary_tools=secondary,
            model_preference=model_pref,
            reasoning=f"Pattern scores: {scores}",
            requires_vision=has_images or primary in ("imagegen",),
            requires_reasoning=primary in ("web", "analysis", "reasoning"),
            requires_code=primary == "code",
        )

    def get_model_order(self, decision: RouteDecision) -> list:
        """Get ordered list of model keys based on decision."""
        base_order = self.MODEL_PREFERENCES.get(decision.model_preference, self.MODEL_PREFERENCES["default"])

        if decision.requires_vision:
            # Filter to vision-capable models
            from models.providers import PROVIDERS
            base_order = [k for k in base_order if PROVIDERS.get(k, type('', (), {'supports_vision': False})()).supports_vision]

        return base_order

    def execute_tool(self, tool_name: str, query: str, **kwargs) -> ToolResult:
        """Execute a single tool."""
        if not self.tools_enabled:
            return ToolResult(tool=tool_name, success=False, error="Tools disabled")

        import logging
        log = logging.getLogger("tools.manager")
        log.info(f"[DEBUG tool_manager] execute_tool called: tool={tool_name}, query={query[:50]}")

        try:
            if tool_name == "time":
                # Check for historical time conversion query first
                hist = parse_historical_time_query(query)
                log.info(f"[DEBUG tool_manager] hist={hist}")
                if hist:
                    log.info(f"[DEBUG tool_manager] calling convert_time with {hist}")
                    answer = convert_time(
                        source_time_str=hist["source_time"],
                        source_tz_str=hist["source_tz"],
                        target_tz_str=hist["target_tz"],
                        source_date_str=hist["source_date"],
                    )
                    log.info(f"[DEBUG tool_manager] convert_time returned: {answer[:100]}")
                    return ToolResult(tool="time", success=True, direct_answer=answer, data=hist)
                
                # Regular current time query
                city = _extract_city(query, r"\b(?:time is it in|current time in|time in|time at|what time in)\s+([^\n?]+)")
                answer = get_time(city=city)
                return ToolResult(tool="time", success=True, direct_answer=answer, data={"city": city})

            elif tool_name == "weather":
                city = _extract_city(query, r"\b(?:weather|forecast|temperature|raining|snowing) (?:in|at|for) ([^\n?]+)")
                answer = weather_now(city=city, default_city=self.settings.get("default_city"))
                return ToolResult(tool="weather", success=True, direct_answer=answer, data={"city": city})

            elif tool_name == "calendar":
                answer = calendar_answer(query)
                return ToolResult(tool="calendar", success=True, direct_answer=answer)

            elif tool_name == "calculator":
                if detect_calculation(query):
                    from tools.calculator import extract_expression
                    expr = extract_expression(query)
                    answer = evaluate_expression(expr)
                    if isinstance(answer, str) and answer.startswith(("Invalid", "Division", "Empty")):
                        return ToolResult(tool="calculator", success=False, error=answer)
                    return ToolResult(tool="calculator", success=True, direct_answer=f"**Calculator**  \nResult: **{answer}**")
                return ToolResult(tool="calculator", success=False, error="No valid expression found")

            elif tool_name == "web":
                search_query = _web_query(query)
                max_results = self.settings.get("web_results", 4)
                results = multi_search(search_query, max_results=max_results)
                context = format_search_results(results, search_query)
                sources = [r.get("url", "") for r in results]
                direct = None if results else "No results found."
                return ToolResult(
                    tool="web", success=True, context_for_llm=context, direct_answer=direct, sources=sources, data={"items": results}
                )

            elif tool_name == "imagegen":
                from tools.imagegen import imagegen_answer
                result = imagegen_answer(query)
                return ToolResult(tool="imagegen", success=True, data=result, direct_answer=result.get("direct"))

            elif tool_name == "analysis":
                # Placeholder for document analysis - would use RAG
                return ToolResult(tool="analysis", success=False, error="Analysis tool not yet implemented")

            elif tool_name == "code":
                return ToolResult(tool="code", success=False, error="Code execution not yet implemented")

            else:
                return ToolResult(tool=tool_name, success=False, error=f"Unknown tool: {tool_name}")

        except Exception as exc:
            log.exception("Tool %s failed", tool_name)
            return ToolResult(tool=tool_name, success=False, error=str(exc))

    def execute_plan(self, query: str, has_images: bool = False, has_files: bool = False) -> dict:
        """Execute the full tool plan and return combined results."""
        decision = self.classify_intent(query, has_images, has_files)

        if decision.primary_tool == "chat":
            return {"tools_used": [], "model_order": self.get_model_order(decision), "decision": decision}

        results = {}
        tools_to_run = [decision.primary_tool] + decision.secondary_tools

        for tool in tools_to_run:
            result = self.execute_tool(tool, query)
            results[tool] = result

        # Build combined context for LLM
        context_parts = []
        sources = []
        direct_answers = []

        for tool, result in results.items():
            if result.success:
                if result.context_for_llm:
                    context_parts.append(f"[{tool.upper()}]\n{result.context_for_llm}")
                if result.direct_answer:
                    direct_answers.append(f"[{tool.upper()}] {result.direct_answer}")
                if result.sources:
                    sources.extend(result.sources)

        combined_context = "\n\n".join(context_parts) if context_parts else None
        combined_direct = "\n\n".join(direct_answers) if direct_answers else None

        return {
            "tools_used": list(results.keys()),
            "model_order": self.get_model_order(decision),
            "decision": decision,
            "results": results,
            "combined_context": combined_context,
            "combined_direct": combined_direct,
            "sources": sources,
        }


def create_tool_manager(settings: dict) -> ToolManager:
    """Factory function."""
    return ToolManager(settings)