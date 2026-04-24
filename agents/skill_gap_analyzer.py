"""Skill gap analyzer agent using Gemini reasoning with structured output."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from models.schemas import ResearchOutput, SkillGapOutput, UserProfile
from utils.gemini_client import GeminiClient


@dataclass(slots=True)
class SkillGapAnalyzerInput:
    """Structured input for skill gap analysis."""

    background: str
    research_output: str


@dataclass(slots=True)
class SkillGapAnalyzerResult:
    """Structured JSON-like output returned by the analyzer."""

    missing_skills: list[str]
    tools: list[str]
    experience_gaps: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        """Serializes result using the required strict JSON structure."""
        return {
            "missing_skills": self.missing_skills,
            "tools": self.tools,
            "experience_gaps": self.experience_gaps,
        }


class SkillGapAnalyzer:
    """Reusable analyzer that returns strict structured JSON fields."""

    def __init__(self, gemini_api_key: str | None = None) -> None:
        """Initializes Gemini client for reasoning-based analysis."""
        resolved_key = (
            gemini_api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GROK_API_KEY")
            or os.getenv("GEMINI_API_KEY", "")
        )
        model_name = None
        if not resolved_key.startswith("AIza"):
            model_name = os.getenv("GROQ_MODEL") or os.getenv("GROK_MODEL", "llama-3.3-70b-versatile")
        self.gemini = GeminiClient(
            api_key=resolved_key,
            model_name=model_name,
        )

    def run(self, payload: SkillGapAnalyzerInput) -> SkillGapAnalyzerResult:
        """Analyzes skill gaps and returns a normalized structured result."""
        prompt = f"""
You are a technical career coach.
Analyze missing skills using reasoning and return strict JSON with exactly these keys:
{{
  "missing_skills": ["string"],
  "tools": ["string"],
  "experience_gaps": ["string"]
}}

Rules:
- missing_skills: 5-12 concise items.
- tools: practical tools/platforms/frameworks needed for the target transition.
- experience_gaps: real-world exposure or portfolio gaps to close.
- Do not add extra keys, prose, or markdown.
- Output only valid JSON object.

User background:
{payload.background}

Research output:
{payload.research_output}
""".strip()

        parsed = self.gemini.generate_json(prompt)
        return SkillGapAnalyzerResult(
            missing_skills=[str(item).strip() for item in parsed.get("missing_skills", [])],
            tools=[str(item).strip() for item in parsed.get("tools", [])],
            experience_gaps=[str(item).strip() for item in parsed.get("experience_gaps", [])],
        )


class SkillGapAnalyzerAgent:
    """Backward-compatible adapter for existing multi-agent pipeline."""

    def __init__(self, _unused_gemini_client: Any | None = None) -> None:
        """Initializes reusable skill gap analyzer."""
        self.analyzer = SkillGapAnalyzer()

    def run(self, user: UserProfile, research: ResearchOutput) -> SkillGapOutput:
        """Maps new strict JSON result to legacy schema expected by app pipeline."""
        payload = SkillGapAnalyzerInput(
            background=user.current_background,
            research_output=(
                f"Target role: {user.target_role}\n"
                f"Summary: {research.role_market_summary}\n"
                f"Requirements: {research.role_requirements}"
            ),
        )
        result = self.analyzer.run(payload)

        return SkillGapOutput(
            current_strengths=[],
            missing_skills=result.missing_skills,
            priority_skills=result.missing_skills[:5],
            rationale=(
                "Tools needed: "
                + ", ".join(result.tools[:6])
                + ". Experience gaps: "
                + ", ".join(result.experience_gaps[:6])
            ).strip(),
        )
