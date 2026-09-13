"""
Ultron - tools subsystem.

Server-side "telling" tools available to the assistant:

  * time      - current local time, or time in any city (timezone-aware)
  * weather   - current conditions for any city (Open-Meteo, no API key)
  * calendar  - date calculations, calendar views
  * web       - web search + page scan, with snippets handed to the LLM
  * imagegen  - image generation via HF FLUX
  * tool_mgr  - unified ToolManager for multi-tool routing & model selection
"""
from tools.tool_manager import ToolManager, create_tool_manager

__all__ = ["ToolManager", "create_tool_manager"]