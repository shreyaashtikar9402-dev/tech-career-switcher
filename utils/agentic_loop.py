"""Reusable agentic loop for Groq function calling."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI

LOGGER = logging.getLogger(__name__)

ToolExecutor = Callable[[dict[str, Any]], Any]


@dataclass(slots=True)
class AgenticLoopConfig:
    """Configuration options for running the Groq agentic loop."""

    model_name: str = "llama-3.3-70b-versatile"
    max_turns: int = 12
    temperature: float = 0.2

    def __post_init__(self) -> None:
        """Validates config bounds for safe production usage."""
        if self.max_turns < 1 or self.max_turns > 50:
            raise ValueError("max_turns must be between 1 and 50.")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")


@dataclass(slots=True)
class AgenticLoopResult:
    """Final result returned by the loop."""

    final_text: str
    turns_used: int
    tool_calls: int
    stopped_reason: str
    transcript: list[dict[str, Any]]


class GeminiAgenticLoop:
    """Reusable loop that lets the model decide tool-use and termination."""

    def __init__(
        self,
        api_key: str,
        tools: list[dict[str, Any]],
        tool_executors: dict[str, ToolExecutor],
        config: AgenticLoopConfig | None = None,
    ) -> None:
        """
        Initializes model, tool declarations, and tool executors.

        Args:
            api_key: Groq API key.
            tools: function/tool declarations.
            tool_executors: Mapping of function name -> Python callable.
            config: Optional loop configuration.
        """
        resolved_api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GROK_API_KEY")
            or os.getenv("GEMINI_API_KEY", "")
        )
        if not resolved_api_key:
            raise ValueError("GROQ_API_KEY is required.")
        if not tools:
            raise ValueError("At least one tool declaration is required.")
        if not tool_executors:
            raise ValueError("At least one tool executor is required.")

        self.config = config or AgenticLoopConfig()
        self.tools = self._normalize_tools(tools)
        self.tool_executors = tool_executors
        self.client = OpenAI(api_key=resolved_api_key, base_url="https://api.groq.com/openai/v1")

    def run(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> AgenticLoopResult:
        """
        Runs the model loop until no tool call is emitted.

        Args:
            user_prompt: Primary user request.
            system_prompt: Optional instruction for behavior constraints.
            extra_messages: Optional prior chat messages in Chat Completions format.
        """
        if not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty.")

        transcript: list[dict[str, Any]] = []
        tool_calls_count = 0
        final_text = ""

        messages: list[dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user_prompt.strip()})

        stopped_reason = "max_turns_reached"
        turns_used = 0

        for turn_idx in range(1, self.config.max_turns + 1):
            turns_used = turn_idx
            LOGGER.debug("Agentic loop turn %s starting", turn_idx)

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=self.config.temperature,
            )
            message = response.choices[0].message
            model_text = (message.content or "").strip()
            function_calls = self._extract_function_calls(message)

            transcript.append(
                {
                    "turn": turn_idx,
                    "model_text": model_text,
                    "function_calls": function_calls,
                }
            )

            if not function_calls:
                final_text = model_text or "No textual response generated."
                stopped_reason = "model_stopped_calling_tools"
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [self._to_tool_call_payload(fc) for fc in function_calls],
                }
            )

            for call in function_calls:
                tool_calls_count += 1
                tool_name = call["name"]
                args = call["args"]
                result_payload = self._execute_tool(tool_name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps({"result": result_payload}),
                    }
                )

        return AgenticLoopResult(
            final_text=final_text,
            turns_used=turns_used,
            tool_calls=tool_calls_count,
            stopped_reason=stopped_reason,
            transcript=transcript,
        )

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Executes one tool safely and returns serializable result payload."""
        executor = self.tool_executors.get(tool_name)
        if executor is None:
            LOGGER.warning("Tool not found: %s", tool_name)
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = executor(args)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Tool execution failed: %s", tool_name)
            return {"error": str(exc), "tool": tool_name}

        return self._make_json_safe(result)

    @staticmethod
    def _extract_function_calls(message: Any) -> list[dict[str, Any]]:
        """Extracts all function calls from a chat completion message."""
        calls: list[dict[str, Any]] = []
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            return calls

        for tool_call in tool_calls:
            function = getattr(tool_call, "function", None)
            if not function:
                continue
            name = str(getattr(function, "name", ""))
            raw_args = getattr(function, "arguments", "{}") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            if not isinstance(args, dict):
                args = {"value": args}
            calls.append({"id": str(getattr(tool_call, "id", "")), "name": name, "args": args})
        return calls

    @staticmethod
    def _to_tool_call_payload(function_call: dict[str, Any]) -> dict[str, Any]:
        """Converts parsed call into assistant tool call payload format."""
        return {
            "id": function_call["id"],
            "type": "function",
            "function": {
                "name": function_call["name"],
                "arguments": json.dumps(function_call["args"]),
            }
        }

    @staticmethod
    def _normalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalizes Gemini-style or OpenAI-style tools to OpenAI format."""
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                normalized.append(tool)
                continue
            declarations = tool.get("function_declarations", [])
            for decl in declarations:
                normalized.append(
                    {
                        "type": "function",
                        "function": {
                            "name": decl.get("name", ""),
                            "description": decl.get("description", ""),
                            "parameters": decl.get(
                                "parameters", {"type": "object", "properties": {}}
                            ),
                        },
                    }
                )
        return normalized

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        """Converts arbitrary Python objects into JSON-safe values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): GeminiAgenticLoop._make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [GeminiAgenticLoop._make_json_safe(v) for v in value]
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

