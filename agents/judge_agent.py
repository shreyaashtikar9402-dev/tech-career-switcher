"""Judge agent for roadmap evaluation with structured scoring JSON."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from models.schemas import JudgeOutput, PathOutput, RoadmapOutput, SkillGapOutput, UserProfile
from utils.gemini_client import GeminiClient


@dataclass(slots=True)
class JudgeInput:
    """Structured input for judging roadmap quality."""

    background: str
    target_role: str
    timeline: str
    roadmap_markdown: str


@dataclass(slots=True)
class JudgeResult:
    """Structured evaluator output using requested JSON format."""

    scores: dict[str, float]
    overall_score: float
    summary: str
    improvement: str

    def to_dict(self) -> dict[str, Any]:
        """Serializes result to required strict JSON keys."""
        return {
            "scores": self.scores,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "improvement": self.improvement,
        }


class RoadmapJudge:
    """Reusable Gemini evaluator for roadmap quality."""

    def __init__(self, gemini_api_key: str | None = None) -> None:
        """Initializes Gemini client for evaluation."""
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

    def run(self, payload: JudgeInput) -> JudgeResult:
        """Evaluates roadmap and returns structured score output."""
        prompt = f"""
You are a strict evaluator of AI-generated career roadmaps.
Evaluate the roadmap based on:
- role_specificity
- realism
- completeness
- readability

Return strict JSON with exactly:
{{
  "scores": {{
    "role_specificity": 0,
    "realism": 0,
    "completeness": 0,
    "readability": 0
  }},
  "overall_score": 0,
  "summary": "...",
  "improvement": "..."
}}

Rules:
- Score each criterion from 0 to 10 (decimals allowed).
- overall_score should be the average of the four scores.
- summary must be concise (40-90 words).
- improvement should provide the single highest-impact recommendation.
- Output only valid JSON, no markdown.

User input:
- Background: {payload.background}
- Target role: {payload.target_role}
- Timeline: {payload.timeline}

Roadmap:
{payload.roadmap_markdown}
""".strip()

        parsed = self.gemini.generate_json(prompt)
        scores = parsed.get("scores", {})
        safe_scores = {
            "role_specificity": float(scores.get("role_specificity", 0)),
            "realism": float(scores.get("realism", 0)),
            "completeness": float(scores.get("completeness", 0)),
            "readability": float(scores.get("readability", 0)),
        }
        return JudgeResult(
            scores=safe_scores,
            overall_score=float(parsed.get("overall_score", 0)),
            summary=str(parsed.get("summary", "")).strip(),
            improvement=str(parsed.get("improvement", "")).strip(),
        )


class JudgeAgent:
    """Backward-compatible adapter for existing app pipeline."""

    def __init__(self, _unused_gemini_client: Any | None = None) -> None:
        """Initializes reusable roadmap judge."""
        self.judge = RoadmapJudge()

    def run(
        self,
        user: UserProfile,
        skill_gap: SkillGapOutput,
        path: PathOutput,
        roadmap: RoadmapOutput,
    ) -> JudgeOutput:
        """Maps new evaluator output into legacy schema used by app."""
        _ = (skill_gap, path)  # Kept for interface compatibility.
        result = self.judge.run(
            JudgeInput(
                background=user.current_background,
                target_role=user.target_role,
                timeline=user.timeline,
                roadmap_markdown=roadmap.roadmap_markdown,
            )
        )
        strengths: list[str] = []
        improvements: list[str] = []
        if result.scores["role_specificity"] >= 7:
            strengths.append("Roadmap is aligned with the target role expectations.")
        if result.scores["realism"] >= 7:
            strengths.append("Plan appears realistic within the stated timeline.")
        if result.scores["readability"] >= 7:
            strengths.append("Structure and readability are strong.")
        if not strengths:
            strengths.append("Initial structure exists and can be improved with sharper prioritization.")
        improvements.append(result.improvement or "Increase role-specific milestones and measurable outputs.")

        return JudgeOutput(
            score=float(result.overall_score),
            max_score=10.0,
            verdict="Strong" if result.overall_score >= 8 else "Promising" if result.overall_score >= 6 else "Needs Improvement",
            strengths=strengths,
            improvements=improvements,
            explanation=result.summary,
        )
