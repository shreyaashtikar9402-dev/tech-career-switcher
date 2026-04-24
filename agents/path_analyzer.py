"""Path analyzer agent for realistic role-specific 30/60/90 planning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from models.schemas import PathOutput, PlanMilestone, SkillGapOutput, UserProfile
from utils.gemini_client import GeminiClient


@dataclass(slots=True)
class PathAnalyzerInput:
    """Structured input for generating a role-specific path plan."""

    target_role: str
    timeline: str
    research_output: str
    skill_gaps: dict[str, list[str]]


@dataclass(slots=True)
class PathAnalyzerResult:
    """Strict JSON plan output structure."""

    plan_30_60_90: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serializes to required JSON object."""
        return self.plan_30_60_90


class PathAnalyzer:
    """Reusable planner that outputs strict 30/60/90 JSON."""

    def __init__(self, gemini_api_key: str | None = None) -> None:
        """Initializes Gemini client for planning."""
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

    def run(self, payload: PathAnalyzerInput) -> PathAnalyzerResult:
        """Builds a realistic role-specific plan in strict JSON format."""
        prompt = f"""
You are a career strategy planner.
Create a realistic role-specific 30/60/90 day plan.
Return strict JSON with this exact shape:
{{
  "30_days": {{
    "weekly_tasks": ["string"],
    "resources": ["string"],
    "milestones": ["string"]
  }},
  "60_days": {{
    "weekly_tasks": ["string"],
    "resources": ["string"],
    "milestones": ["string"]
  }},
  "90_days": {{
    "weekly_tasks": ["string"],
    "resources": ["string"],
    "milestones": ["string"]
  }}
}}

Rules:
- Make it practical for someone transitioning into the target role.
- weekly_tasks: 4-8 actionable tasks per phase.
- resources: 3-6 high-value resources (courses, docs, tools, communities).
- milestones: 3-5 measurable outcomes per phase.
- Align tasks with timeline: {payload.timeline}
- Use research insights and skill gaps directly.
- Output only valid JSON object. No markdown or extra keys.

Target Role: {payload.target_role}
Research Output:
{payload.research_output}

Skill Gaps:
{payload.skill_gaps}
""".strip()

        parsed = self.gemini.generate_json(prompt)
        result: dict[str, dict[str, Any]] = {}
        for key in ("30_days", "60_days", "90_days"):
            node = parsed.get(key, {}) if isinstance(parsed, dict) else {}
            result[key] = {
                "weekly_tasks": [str(item).strip() for item in node.get("weekly_tasks", [])],
                "resources": [str(item).strip() for item in node.get("resources", [])],
                "milestones": [str(item).strip() for item in node.get("milestones", [])],
            }
        return PathAnalyzerResult(plan_30_60_90=result)


class PathAnalyzerAgent:
    """Backward-compatible adapter for existing app pipeline."""

    def __init__(self, _unused_gemini_client: Any | None = None) -> None:
        """Initializes reusable path analyzer."""
        self.path_analyzer = PathAnalyzer()

    def run(self, user: UserProfile, skill_gap: SkillGapOutput) -> PathOutput:
        """Maps strict JSON plan into legacy schema used by the app."""
        payload = PathAnalyzerInput(
            target_role=user.target_role,
            timeline=user.timeline,
            research_output=(
                f"Role summary: transition focus toward {user.target_role} "
                f"from background {user.current_background}."
            ),
            skill_gaps={
                "missing_skills": skill_gap.missing_skills,
                "priority_skills": skill_gap.priority_skills,
                "tools_and_experience_notes": [skill_gap.rationale],
            },
        )
        result = self.path_analyzer.run(payload).to_dict()

        mapped_plan: dict[str, PlanMilestone] = {}
        for key in ("30_days", "60_days", "90_days"):
            phase = result.get(key, {})
            mapped_plan[key] = PlanMilestone(
                focus=f"{key.replace('_', ' ').title()} role transition execution",
                goals=[str(item).strip() for item in phase.get("weekly_tasks", [])][:6],
                deliverables=[str(item).strip() for item in phase.get("milestones", [])][:5],
                resources=[str(item).strip() for item in phase.get("resources", [])][:6],
            )

        return PathOutput(
            plan_json=mapped_plan,
            success_metrics=[
                "Complete each phase milestones on schedule",
                "Build portfolio artifacts linked to target role expectations",
                "Demonstrate measurable improvement in core missing skills",
            ],
        )
