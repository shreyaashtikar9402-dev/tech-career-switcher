"""LLM client utilities supporting Gemini and Groq providers."""

from __future__ import annotations

import json
import os
from typing import Any

import google.generativeai as genai
from openai import OpenAI


class GeminiClient:
    """Backward-compatible wrapper that auto-selects provider by key."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        """Initializes model client and stores provider/model settings."""
        resolved_api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GROK_API_KEY", "")
        )
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        groq_model = os.getenv("GROQ_MODEL") or os.getenv("GROK_MODEL", "llama-3.3-70b-versatile")
        resolved_model_name = model_name or (gemini_model if self._looks_like_gemini_api_key(resolved_api_key) else groq_model)

        if not resolved_api_key:
            raise ValueError(
                "Missing API key. Add GEMINI_API_KEY or GROQ_API_KEY to your environment or .env file."
            )

        self.model_name = resolved_model_name
        self.provider = "gemini" if self._looks_like_gemini_api_key(resolved_api_key) else "groq"
        if self.provider == "gemini":
            genai.configure(api_key=resolved_api_key)
            self.gemini_model = genai.GenerativeModel(self.model_name)
            self.client = None
        else:
            if not self._looks_like_groq_api_key(resolved_api_key):
                raise ValueError(
                    "GROQ_API_KEY appears invalid for Groq. "
                    "Use a Groq key (typically starts with 'gsk_')."
                )
            self.client = OpenAI(api_key=resolved_api_key, base_url="https://api.groq.com/openai/v1")
            self.gemini_model = None

    def generate_text(self, prompt: str) -> str:
        """Generates a plain text response from selected provider."""
        if self.provider == "gemini":
            response = self.gemini_model.generate_content(prompt)
            return (response.text or "").strip()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    def generate_json(self, prompt: str) -> dict[str, Any]:
        """Generates JSON and parses it into a Python dictionary."""
        text = self.generate_text(prompt)
        cleaned = self._extract_json_block(text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON output: {text}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON response must be a top-level object.")
        return parsed

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """Extracts raw JSON from optional markdown code fences."""
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            # Skip opening and closing fence lines.
            core = "\n".join(lines[1:-1]).strip()
            # Remove optional language hint.
            if core.startswith("json"):
                core = "\n".join(core.splitlines()[1:]).strip()
            return core
        return stripped

    @staticmethod
    def _looks_like_groq_api_key(api_key: str) -> bool:
        """Performs a lightweight format check for Groq API keys."""
        return api_key.startswith("gsk_") and len(api_key) >= 20

    @staticmethod
    def _looks_like_gemini_api_key(api_key: str) -> bool:
        """Performs a lightweight format check for Google Gemini API keys."""
        return api_key.startswith("AIza") and len(api_key) >= 20
