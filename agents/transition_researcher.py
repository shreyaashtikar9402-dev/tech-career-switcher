"""Transition researcher agent using Tavily + Gemini/Groq loop."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tavily import TavilyClient

from models.schemas import ResearchOutput, SearchInsight, UserProfile
from utils.agentic_loop import AgenticLoopConfig, GeminiAgenticLoop
from utils.gemini_client import GeminiClient


@dataclass(slots=True)
class TransitionResearchInput:
    """Structured input for transition research."""

    background: str
    target_role: str
    timeline: str


@dataclass(slots=True)
class TransitionResearchResult:
    """Structured output for transition research."""

    summary: str
    calls_used: int
    turns_used: int


class TransitionResearcher:
    """Reusable transition researcher using Gemini-directed tool calls."""

    def __init__(
        self,
        groq_api_key: str | None = None,
        tavily_api_key: str | None = None,
        max_turns: int = 12,
    ) -> None:
        """Initializes Tavily client and Gemini agentic loop configuration."""
        resolved_tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")
        resolved_groq_key = (
            groq_api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GROK_API_KEY")
            or os.getenv("GEMINI_API_KEY", "")
        )

        if not resolved_tavily_key:
            raise ValueError("TAVILY_API_KEY is missing. Add it to environment or .env.")
        resolved_gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not resolved_groq_key and not resolved_gemini_key:
            raise ValueError("Set GROQ_API_KEY or GEMINI_API_KEY in environment or .env.")

        self.tavily = TavilyClient(api_key=resolved_tavily_key)
        self.use_gemini_direct = bool(resolved_gemini_key and resolved_gemini_key.startswith("AIza"))
        if self.use_gemini_direct:
            self.text_llm = GeminiClient(api_key=resolved_gemini_key, model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
            self.gemini_loop = None
        else:
            self.text_llm = None
            self.gemini_loop = GeminiAgenticLoop(
                api_key=resolved_groq_key,
                tools=self._build_tool_declarations(),
                tool_executors={"tavily_transition_search": self._tavily_transition_search},
                config=AgenticLoopConfig(
                    model_name=os.getenv("GROQ_MODEL") or os.getenv("GROK_MODEL", "llama-3.3-70b-versatile"),
                    max_turns=max_turns,
                    temperature=0.2,
                ),
            )

    def run(self, payload: TransitionResearchInput) -> TransitionResearchResult:
        """Runs Gemini tool-calling loop and returns structured text summary."""
        if self.use_gemini_direct:
            return self._run_with_direct_search(payload)

        system_prompt = (
            "You are Transition Researcher Agent. Use the tool as needed to gather web evidence. "
            "You may call the tool multiple times. Stop once you have enough evidence."
        )
        user_prompt = f"""
Create a structured transition research summary.

Inputs:
- Background: {payload.background}
- Target role: {payload.target_role}
- Timeline: {payload.timeline}

Requirements:
- Search for (1) career transition success stories, (2) required skills, (3) realistic timelines.
- Use tool calls for evidence.
- Return final output as structured text with sections:
  1) Transition Snapshot
  2) Success Story Patterns
  3) Required Skills
  4) Timeline Reality Check
  5) Actionable Recommendations

Style constraints:
- Concise and evidence-based.
- Include 5-8 bullet points under Required Skills.
- Mention caveats where needed.
""".strip()

        result = self.gemini_loop.run(user_prompt=user_prompt, system_prompt=system_prompt)
        return TransitionResearchResult(
            summary=result.final_text,
            calls_used=result.tool_calls,
            turns_used=result.turns_used,
        )

    def _run_with_direct_search(self, payload: TransitionResearchInput) -> TransitionResearchResult:
        """Runs direct Tavily search + Gemini summarization fallback path."""
        queries = [
            f"{payload.background} to {payload.target_role} career transition success stories",
            f"{payload.target_role} required skills 2026 hiring expectations",
            f"realistic timeline to become {payload.target_role} from {payload.background}",
        ]
        insights: list[dict[str, Any]] = []
        for query in queries:
            result = self.tavily.search(query=query, max_results=4)
            for item in result.get("results", []):
                insights.append(
                    {
                        "title": str(item.get("title", "Untitled")),
                        "url": str(item.get("url", "")),
                        "snippet": str(item.get("content", ""))[:500],
                    }
                )

        sources_block = "\n".join(
            f"- Title: {item['title']}\n  URL: {item['url']}\n  Snippet: {item['snippet']}" for item in insights[:12]
        )
        prompt = f"""
Create a structured transition research summary.

Inputs:
- Background: {payload.background}
- Target role: {payload.target_role}
- Timeline: {payload.timeline}

Use the evidence below and return structured text with sections:
1) Transition Snapshot
2) Success Story Patterns
3) Required Skills
4) Timeline Reality Check
5) Actionable Recommendations

Style:
- Concise and evidence-based.
- Include 5-8 bullet points under Required Skills.
- Mention caveats where needed.

Evidence:
{sources_block}
""".strip()
        summary = self.text_llm.generate_text(prompt)
        return TransitionResearchResult(summary=summary, calls_used=0, turns_used=1)

    @staticmethod
    def _build_tool_declarations() -> list[dict[str, Any]]:
        """Builds Gemini function declarations for Tavily search."""
        return [
            {
                "function_declarations": [
                    {
                        "name": "tavily_transition_search",
                        "description": "Searches web evidence for career transitions and role expectations.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query containing role transition context.",
                                },
                                "search_focus": {
                                    "type": "string",
                                    "enum": [
                                        "success_stories",
                                        "required_skills",
                                        "timelines",
                                    ],
                                    "description": "The focus area for this search call.",
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Number of results to retrieve (1-8).",
                                },
                            },
                            "required": ["query", "search_focus"],
                        },
                    }
                ]
            }
        ]

    def _tavily_transition_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Executes Tavily search and returns normalized compact evidence."""
        query = str(args.get("query", "")).strip()
        search_focus = str(args.get("search_focus", "required_skills")).strip()
        max_results = int(args.get("max_results", 5) or 5)
        max_results = min(8, max(1, max_results))

        if not query:
            return {"error": "query is required"}

        result = self.tavily.search(query=query, max_results=max_results)
        raw_items = result.get("results", [])

        normalized = []
        for item in raw_items:
            normalized.append(
                {
                    "title": str(item.get("title", "Untitled")),
                    "url": str(item.get("url", "")),
                    "snippet": str(item.get("content", ""))[:500],
                    "focus": search_focus,
                }
            )

        return {"query": query, "focus": search_focus, "results": normalized}


class TransitionResearcherAgent:
    """
    Backward-compatible adapter for existing app pipeline.

    Converts the new structured text summary into legacy ResearchOutput fields.
    """

    def __init__(self, _unused_gemini_client: Any | None = None) -> None:
        """Initializes reusable transition researcher.

        The optional argument preserves compatibility with older wiring that passed
        a Gemini client instance into this adapter.
        """
        self.researcher = TransitionResearcher()

    def run(self, user: UserProfile) -> ResearchOutput:
        """Runs transition research and maps result into legacy schema."""
        result = self.researcher.run(
            TransitionResearchInput(
                background=user.current_background,
                target_role=user.target_role,
                timeline=user.timeline,
            )
        )

        # Lightweight extraction to preserve legacy fields used by downstream agents.
        role_requirements: list[str] = []
        lines = [line.rstrip() for line in result.summary.splitlines()]
        in_required_skills = False
        for line in lines:
            if not line:
                continue
            cleaned = line.strip()
            lower = cleaned.lower()
            if "required skills" in lower:
                in_required_skills = True
                continue
            if in_required_skills and cleaned.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.")):
                role_requirements.append(cleaned)
            elif in_required_skills and cleaned.startswith(("*", "-", "•")):
                role_requirements.append(cleaned.lstrip("*-• ").strip())
            elif in_required_skills and cleaned.endswith(":"):
                in_required_skills = False

        if not role_requirements:
            role_requirements = [
                "Role-specific technical fundamentals",
                "Portfolio-ready project execution",
                "Interview communication and problem solving",
            ]

        return ResearchOutput(
            role_market_summary=result.summary[:700],
            role_requirements=role_requirements[:10],
            insights=[
                SearchInsight(
                    title="Agentic summary",
                    url="",
                    snippet=f"Calls used: {result.calls_used}, turns used: {result.turns_used}",
                )
            ],
        )
