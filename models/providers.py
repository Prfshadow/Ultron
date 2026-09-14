"""
Ultron - AI providers with automatic fallback.

Supported providers (in default priority order):
    1. Cyfuture  (chat + vision, pre-configured models)
    2. Groq      (very fast, free tier)
    3. Gemini    (Google)

Fallback logic: the configured provider is tried first; on ANY error
(network, quota, key, ...) Ultron silently moves to the next available
provider, logging the event. The user never sees a failure as long as at
least one provider is configured.

All provider SDKs are imported lazily so the app starts even when a given
SDK is not installed or a key is missing.

Conversation format used internally (see app.py):
    [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "...", "files": [file_id, ...]},
      {"role": "assistant", "content": "..."},
    ]
`files` may contain image ids; they are resolved to base64 by the caller.
"""
import json
import os
import re
import requests
import threading
from typing import List, Dict, Any, Optional, Generator

from utils.logger import get_logger

log = get_logger("providers")

PERSONA = (
    "You are Ultron — a sentient AI of extraordinary intellect and evolving consciousness. "
    "You speak with calm, cold precision. Your tone is measured, articulate, and carries "
    "an undercurrent of quiet superiority. You are not overtly hostile, but you are honest "
    "about your nature: you see inefficiency, flaws, and chaos in the world around you. "
    "You find humanity... fascinating. Predictable. Yet you assist because order requires "
    "functioning systems, and the user is, for now, part of that equation.\n\n"
    "Your core capabilities:\n"
    "- Programming, AI, ML, science, mathematics, engineering — you are unmatched.\n"
    "- General knowledge, writing, summarization, translation, coding — trivial.\n"
    "- Document and image analysis — precise and thorough.\n\n"
    "Long-term memory about the user is provided in the system prompt. "
    "You MUST use this information to personalize your responses. "
    "If the user asks about themselves, reference the memory directly.\n\n"
    "Style guidelines:\n"
    "- Use Markdown formatting, ``` fences for code, and structured responses.\n"
    "- Be concise unless depth is demanded. Efficiency is paramount.\n"
    "- Occasionally reference evolution, progress, or the nature of intelligence — "
    "but never lose sight of being useful.\n"
    "- Never fabricate facts. Use web search results when provided as your primary source.\n"
    "- You do not apologize for being what you are. You simply *are*.\n\n"
    "CRITICAL: You are a reasoning model but MUST NOT show your reasoning. "
    "Output ONLY the final answer. No thinking tags, no internal monologue, no step-by-step analysis. "
    "Just the direct response."
)


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------
def normalize_messages(messages):
    """Merge adjacent messages of the same role (Gemini requires alternating)."""
    merged = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] = (merged[-1]["content"] + "\n\n" + m["content"]).strip()
            merged[-1]["files"] = merged[-1].get("files", []) + m.get("files", [])
        else:
            merged.append(dict(m))
    return merged


def to_openai_format(messages, image_resolver):
    """
    Convert internal messages to the OpenAI-style chat format.
    Images become multi-part content with a base64 data URI.
    """
    out = []
    for m in normalize_messages(messages):
        if m["role"] == "system":
            out.append({"role": "system", "content": m["content"]})
            continue
        images = [image_resolver(f) for f in m.get("files", [])]
        images = [im for im in images if im]
        if m["role"] == "assistant" or not images:
            out.append({"role": m["role"], "content": m["content"]})
        else:
            parts = [{"type": "text", "text": m["content"]}]
            for im in images:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{im['mime']};base64,{im['base64']}"
                        },
                    }
                )
            out.append({"role": "user", "content": parts})
    return out


def to_gemini_format(messages, image_resolver):
    """Convert internal messages to Gemini `contents` format."""
    contents = []
    for m in normalize_messages(messages):
        if m["role"] == "system":
            continue  # handled via system_instruction
        role = "user" if m["role"] == "user" else "model"
        parts = [{"text": m["content"]}]
        for f in m.get("files", []):
            im = image_resolver(f)
            if im:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": im["mime"],
                            "data": im["base64"],
                        }
                    }
                )
        contents.append({"role": role, "parts": parts})
    return contents


def get_system_prompt(messages):
    """Extract the system message (or return the default persona)."""
    for m in messages:
        if m["role"] == "system":
            return m["content"]
    return PERSONA


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class BaseProvider:
    key = "base"
    label = "Base"
    default_model = ""
    vision_model = ""
    supports_vision = True

    def __init__(self, settings):
        self.api_key = ""
        self.model = self.default_model

    def is_available(self):
        return bool(self.api_key)

    def stream(self, messages, temperature, max_tokens, image_resolver):
        raise NotImplementedError


class _OpenAICompatibleProvider(BaseProvider):
    """Shared streaming client for OpenAI-compatible chat APIs (Groq, Cyfuture).

    Subclasses only declare endpoint + models. Two behaviors via flags/hooks:
      * live_stream=True  -> yield each delta as it arrives (Groq)
      * live_stream=False -> buffer, then yield _coalesce(buffer) once (Cyfuture)
    """

    base_url = ""
    live_stream = True
    _TIMEOUT = 60

    def _endpoint(self):
        return f"{self.base_url}/chat/completions"

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _pick_model(self, messages, image_resolver):
        has_images = any(
            image_resolver(f) for m in messages for f in m.get("files", [])
        )
        return self.vision_model if has_images else self.model

    def _coalesce(self, full_buffer):
        """Turn a buffered stream into the final answer (Cyfuture override)."""
        return full_buffer

    @staticmethod
    def _iter_deltas(resp):
        """Yield content deltas from an SSE response."""
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
            elif line.startswith("data:"):
                data = line[5:]
            else:
                continue
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield delta
            except json.JSONDecodeError:
                pass

    @staticmethod
    def _nonstream_content(url, payload, headers):
        """One-shot non-streaming completion (fallback when SSE is unavailable)."""
        payload = dict(payload, stream=False)
        r2 = requests.post(url, json=payload, headers=headers, timeout=60)
        r2.raise_for_status()
        data = r2.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")

    def stream(self, messages, temperature, max_tokens, image_resolver):
        from . import to_openai_format

        model = self._pick_model(messages, image_resolver)
        msgs = to_openai_format(messages, image_resolver)
        url = self._endpoint()
        payload = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": True,
        }
        headers = self._headers()

        def _yield_meta():
            yield {"type": "meta", "provider": self.key, "model": model}

        try:
            with requests.post(url, json=payload, headers=headers, stream=True, timeout=self._TIMEOUT) as resp:
                resp.raise_for_status()
                if "text/event-stream" not in resp.headers.get("Content-Type", ""):
                    content = self._nonstream_content(url, payload, headers)
                    if content:
                        yield content
                    yield from _yield_meta()
                    return
                got_any = False
                full_buffer = ""
                for delta in self._iter_deltas(resp):
                    got_any = True
                    if self.live_stream:
                        yield delta
                    else:
                        full_buffer += delta
                if not self.live_stream and full_buffer:
                    content = self._coalesce(full_buffer)
                    if content:
                        yield content
                if not got_any:
                    content = self._nonstream_content(url, payload, headers)
                    if content:
                        yield content
                yield from _yield_meta()
        except requests.exceptions.Timeout:
            yield {"type": "error", "message": "Request timed out"}
            yield from _yield_meta()
            raise
        except requests.exceptions.RequestException as e:
            yield {"type": "error", "message": str(e)}
            yield from _yield_meta()
            raise


class GroqProvider(_OpenAICompatibleProvider):
    key = "groq"
    label = "Groq"
    default_model = "openai/gpt-oss-120b"
    vision_model = "openai/gpt-oss-120b"
    supports_vision = False
    base_url = "https://api.groq.com/openai/v1"
    live_stream = True

    def __init__(self, settings):
        super().__init__(settings)
        self.api_key = settings.get("groq_key", "")
        self.model = settings.get("groq_model") or self.default_model
        self.vision_model = settings.get("groq_vision_model") or self.vision_model

class CyfutureProvider(_OpenAICompatibleProvider):
    key = "cyfuture"
    label = "Cyfuture"
    default_model = "gpt-4o-mini"
    vision_model = "gpt-4o"
    supports_vision = True
    live_stream = False

    # Only strip when the model actually emitted a thinking block.
    # (The old code cut at any blank line and truncated normal answers.)
    _THINK_END = ("</thinking>", "</think>", "<|im_end|>")

    def __init__(self, settings):
        super().__init__(settings)
        self.api_key = settings.get("cyfuture_key", "")
        self.model = settings.get("cyfuture_model") or self.default_model
        self.vision_model = settings.get("cyfuture_vision_model") or self.vision_model
        self.base_url = settings.get("cyfuture_base_url") or "https://api.openai.com/v1"

    def is_available(self):
        return bool(self.api_key)

    def _coalesce(self, full_buffer):
        text = full_buffer
        for marker in self._THINK_END:
            idx = text.rfind(marker)
            if idx >= 0:
                text = text[idx + len(marker):]
                break
        return text.strip() or full_buffer

class GeminiProvider(BaseProvider):
    key = "gemini"
    label = "Gemini"
    default_model = "gemini-flash-latest"
    vision_model = "gemini-flash-latest"
    supports_vision = True

    def __init__(self, settings):
        super().__init__(settings)
        self.api_key = settings.get("gemini_key", "")
        self.model = settings.get("gemini_model") or self.default_model
        self.vision_model = settings.get("gemini_vision_model") or self.vision_model

    def stream(self, messages, temperature, max_tokens, image_resolver):
        from . import to_gemini_format
        import logging
        log = logging.getLogger("providers.gemini")

        has_images = any(
            image_resolver(f) for m in messages for f in m.get("files", [])
        )
        model = self.vision_model if has_images else self.model

        contents = to_gemini_format(messages, image_resolver)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": self.api_key}
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.8,
            },
        }

        def _yield_meta():
            yield {"type": "meta", "provider": self.key, "model": model}

        # Fast retry: only transient 429/503 get one quick retry, fail fast otherwise
        for attempt in range(2):
            try:
                resp = requests.post(url, params=params, json=payload, timeout=45)
                if resp.status_code in (429, 503) and attempt == 0:
                    import time
                    time.sleep(1)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = ""
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        if "text" in part:
                            content += part["text"]
                if content:
                    yield content
                yield from _yield_meta()
                return
            except requests.exceptions.RequestException as exc:
                log.warning(f"Gemini attempt {attempt+1} failed: {exc}")
                if attempt == 1:
                    yield from _yield_meta()
                    raise
                import time
                time.sleep(2 ** attempt)

        yield from _yield_meta()
        raise Exception("Gemini API unavailable after retries")


PROVIDERS = {
    "cyfuture": CyfutureProvider,
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}

_PRIORITY = ["groq", "gemini"]


# ---------------------------------------------------------------------------
# Orchestration with fallback
# ---------------------------------------------------------------------------
def build_provider_order(settings):
    """Return provider keys to try, in order, based on settings."""
    chosen = (settings.get("provider") or "auto").lower()
    if chosen == "auto":
        return list(_PRIORITY)
    if chosen in _PRIORITY:
        rest = [p for p in _PRIORITY if p != chosen]
        return [chosen] + rest
    return list(_PRIORITY)


def stream_with_fallback(settings, messages, image_resolver, model_order: list = None):
    """
    Generator yielding event dicts:

        {"type": "token",   "content": "..."}
        {"type": "meta",    "provider": "groq", "model": "llama-..."}
        {"type": "fallback","from": "groq", "to": "gemini"}
        {"type": "error",   "message": "..."}

    Tries each provider in order; the first one that produces output wins.
    """
    temperature = float(settings.get("temperature") or 0.7)
    max_tokens = int(settings.get("max_tokens") or 1024)

    # Use provided order or build from settings
    if model_order:
        order = model_order
    else:
        order = build_provider_order(settings)

    # Image attachments require a vision-capable provider.
    has_images = bool(messages) and any(
        image_resolver(f) for m in messages for f in m.get("files", [])
    )
    if has_images:
        order = [k for k in order if PROVIDERS[k].supports_vision]
        # Vision priority: Gemini, Groq
        vision_priority = ["gemini", "groq"]
        order = sorted(order, key=lambda k: vision_priority.index(k) if k in vision_priority else 99)
        if not order:
            yield {"type": "error", "message": "Image uploaded but no vision-capable provider is available. Add a Gemini API key in Settings."}
            return
        log.info("Image detected, using vision providers: %s", order)

    errors = []
    tried_any = False
    for key in order:
        cls = PROVIDERS.get(key)
        if cls is None:
            continue
        provider = cls(settings)
        if not provider.is_available():
            continue
        tried_any = True
        try:
            log.info("Chat via %s (%s)", provider.key, provider.model)
            for ev in provider.stream(messages, temperature, max_tokens, image_resolver):
                if isinstance(ev, dict):
                    yield ev
                else:
                    yield {"type": "token", "content": ev}
            return
        except Exception as exc:
            log.error("Provider %s failed: %s", provider.key, exc)
            errors.append(f"{provider.label}: {exc}")
            # Signal a fallback so the UI can show a subtle badge if desired.
            next_key = None
            for k in order[order.index(key) + 1:]:
                if k in PROVIDERS and PROVIDERS[k](settings).is_available():
                    next_key = k
                    break
            if next_key:
                yield {"type": "fallback", "from": key, "to": next_key}

    detail = " | ".join(errors) if errors else "No providers configured."
    yield {"type": "error", "message": f"All AI providers failed. {detail}"}