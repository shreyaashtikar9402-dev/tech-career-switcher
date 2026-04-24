"""Roadmap writer agent for professional motivating markdown output."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from models.schemas import PathOutput, RoadmapOutput, SkillGapOutput, UserProfile
from utils.gemini_client import GeminiClient


@dataclass(slots=True)
class RoadmapWriterInput:
    """Structured input used to generate the roadmap markdown."""

    user_name: str
    background: str
    target_role: str
    timeline: str
    json_plan: dict[str, Any]


@dataclass(slots=True)
class RoadmapWriterResult:
    """Structured output containing generated markdown roadmap."""

    markdown_roadmap: str


class RoadmapWriter:
    """Reusable roadmap writer that renders markdown from JSON plan."""

    def __init__(self, gemini_api_key: str | None = None) -> None:
        """Initializes Gemini client for roadmap writing."""
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

    def run(self, payload: RoadmapWriterInput) -> RoadmapWriterResult:
        """Generates professional and motivating markdown roadmap."""
        prompt = f"""
You are an expert technical career roadmap writer.
Write a polished markdown roadmap with a professional and motivating tone.

Input:
- Name: {payload.user_name}
- Current background: {payload.background}
- Target role: {payload.target_role}
- Timeline: {payload.timeline}
- JSON plan: {payload.json_plan}

Required sections (exactly in this order):
1) Welcome Note
2) Goals
3) Weekly Plan
4) Resources
5) Milestone Checkpoints

Writing rules:
- Keep language clear, practical, and confidence-building.
- Include concrete weekly actions aligned with the JSON plan.
- Keep recommendations role-specific.
- Use markdown headings and bullet points.
- No JSON in final output; return markdown only.
""".strip()
        markdown = self.gemini.generate_text(prompt)
        return RoadmapWriterResult(markdown_roadmap=markdown)


class RoadmapWriterAgent:
    """Backward-compatible adapter for existing app pipeline."""

    def __init__(self, _unused_gemini_client: Any | None = None) -> None:
        """Initializes reusable roadmap writer."""
        self.writer = RoadmapWriter()

    def run(self, user: UserProfile, skill_gap: SkillGapOutput, path: PathOutput) -> RoadmapOutput:
        """Maps legacy inputs into reusable roadmap writer payload."""
        # Include skill context alongside plan to improve role-specific guidance.
        composed_plan = {
            "plan": path.plan_json,
            "success_metrics": path.success_metrics,
            "priority_skills": skill_gap.priority_skills,
            "missing_skills": skill_gap.missing_skills,
        }
        result = self.writer.run(
            RoadmapWriterInput(
                user_name=user.name,
                background=user.current_background,
                target_role=user.target_role,
                timeline=user.timeline,
                json_plan=composed_plan,
            )
        )
        return RoadmapOutput(roadmap_markdown=result.markdown_roadmap)
