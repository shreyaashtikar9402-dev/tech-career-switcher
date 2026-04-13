"""Streamlit app entrypoint for Tech Career Switcher AI."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv
from openai import APIStatusError, AuthenticationError, PermissionDeniedError

from agents.judge_agent import JudgeAgent
from agents.path_analyzer import PathAnalyzerAgent
from agents.roadmap_writer import RoadmapWriterAgent
from agents.skill_gap_analyzer import SkillGapAnalyzerAgent
from agents.transition_researcher import TransitionResearcherAgent
from models.schemas import UserProfile, as_json_safe
from utils.gemini_client import GeminiClient


def run_pipeline(user_profile: UserProfile) -> dict:
    """Runs all five agents sequentially and returns their outputs."""
    gemini_client = GeminiClient()

    researcher = TransitionResearcherAgent(gemini_client)
    skill_gap_analyzer = SkillGapAnalyzerAgent(gemini_client)
    path_analyzer = PathAnalyzerAgent(gemini_client)
    roadmap_writer = RoadmapWriterAgent(gemini_client)
    judge_agent = JudgeAgent(gemini_client)

    research_output = researcher.run(user_profile)
    skill_gap_output = skill_gap_analyzer.run(user_profile, research_output)
    path_output = path_analyzer.run(user_profile, skill_gap_output)
    roadmap_output = roadmap_writer.run(user_profile, skill_gap_output, path_output)
    judge_output = judge_agent.run(user_profile, skill_gap_output, path_output, roadmap_output)

    return {
        "research": research_output,
        "skill_gap": skill_gap_output,
        "path": path_output,
        "roadmap": roadmap_output,
        "judge": judge_output,
    }


def main() -> None:
    """Renders Streamlit UI and executes multi-agent workflow."""
    load_dotenv()
    st.set_page_config(page_title="Tech Career Switcher AI", page_icon="🚀", layout="wide")
    st.title("Tech Career Switcher AI")
    st.caption("Multi-agent career transition planner powered by Groq + Tavily")

    with st.form("career_form"):
        name = st.text_input("Name", placeholder="e.g., Priya")
        current_background = st.text_area(
            "Current Background",
            placeholder="e.g., 2 years in support engineering with basic Python and SQL",
        )
        target_role = st.text_input("Target Role", placeholder="e.g., Data Analyst")
        timeline = st.selectbox("Timeline", options=["3 months", "6 months", "9 months", "12 months"])
        submitted = st.form_submit_button("Generate Career Roadmap")

    if submitted:
        if not name.strip() or not current_background.strip() or not target_role.strip():
            st.error("Please fill in name, current background, and target role.")
            return

        user_profile = UserProfile(
            name=name.strip(),
            current_background=current_background.strip(),
            target_role=target_role.strip(),
            timeline=timeline.strip(),
        )

        with st.spinner("Running multi-agent pipeline..."):
            try:
                outputs = run_pipeline(user_profile)
            except PermissionDeniedError:
                st.error(
                    "Groq API access denied. Check your plan/limits and try again."
                )
                st.info("Go to your Groq dashboard and verify usage limits for this key.")
                return
            except AuthenticationError:
                st.error("Invalid Groq API key. Update `GROQ_API_KEY` in your `.env` file.")
                return
            except APIStatusError as exc:
                st.error(f"Groq API request failed with status {exc.status_code}. Please try again.")
                st.caption(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
                return

        st.subheader("Career Roadmap")
        st.markdown(outputs["roadmap"].roadmap_markdown)

        st.subheader("Judge Score")
        judge = outputs["judge"]
        st.metric("Score", f"{judge.score:.1f} / {judge.max_score:.1f}")
        st.write(f"**Verdict:** {judge.verdict}")
        st.write(f"**Explanation:** {judge.explanation}")
        if judge.strengths:
            st.write("**Strengths**")
            for item in judge.strengths:
                st.write(f"- {item}")
        if judge.improvements:
            st.write("**Improvements**")
            for item in judge.improvements:
                st.write(f"- {item}")

        with st.expander("Debug: Structured Agent Outputs"):
            st.json(
                {
                    "user": as_json_safe(user_profile),
                    "research": as_json_safe(outputs["research"]),
                    "skill_gap": as_json_safe(outputs["skill_gap"]),
                    "path": as_json_safe(outputs["path"]),
                    "judge": as_json_safe(outputs["judge"]),
                }
            )


if __name__ == "__main__":
    main()
