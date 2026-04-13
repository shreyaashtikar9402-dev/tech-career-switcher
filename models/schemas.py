"""Dataclasses used for structured agent inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class UserProfile:
    """Stores the user profile captured from the Streamlit form."""

    name: str
    current_background: str
    target_role: str
    timeline: str


@dataclass(slots=True)
class SearchInsight:
    """Represents a single search result insight from Tavily."""

    title: str
    url: str
    snippet: str


@dataclass(slots=True)
class ResearchOutput:
    """Structured output of the transition researcher agent."""

    role_market_summary: str
    role_requirements: list[str]
    insights: list[SearchInsight] = field(default_factory=list)


@dataclass(slots=True)
class SkillGapOutput:
    """Structured output of the skill gap analyzer agent."""

    current_strengths: list[str]
    missing_skills: list[str]
    priority_skills: list[str]
    rationale: str


@dataclass(slots=True)
class PlanMilestone:
    """Represents a single day-range milestone in the action plan."""

    focus: str
    goals: list[str]
    deliverables: list[str]
    resources: list[str]


@dataclass(slots=True)
class PathOutput:
    """Structured output of the path analyzer agent."""

    plan_json: dict[str, PlanMilestone]
    success_metrics: list[str]


@dataclass(slots=True)
class RoadmapOutput:
    """Structured output of the roadmap writer agent."""

    roadmap_markdown: str


@dataclass(slots=True)
class JudgeOutput:
    """Structured output of the judge agent."""

    score: float
    max_score: float
    verdict: str
    strengths: list[str]
    improvements: list[str]
    explanation: str


def as_json_safe(value: Any) -> Any:
    """Converts dataclasses into JSON-safe dictionaries recursively."""
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: as_json_safe(getattr(value, key))
            for key in value.__dataclass_fields__.keys()
        }
    if isinstance(value, list):
        return [as_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): as_json_safe(v) for k, v in value.items()}
    return value
