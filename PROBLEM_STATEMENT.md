# Tech Career *Switcher* AI — Problem Statement Report

> **Stack:** Python 3.11 · Streamlit · Groq · Tavily &nbsp;|&nbsp; **Agents:** 5 specialized agents &nbsp;|&nbsp; **Report Date:** April 24, 2026

---

## 01 — The Problem Being Solved

Career transitions in the technology industry are among the most high-stakes decisions a professional can make. Someone with 2 years of support engineering experience trying to pivot to data analytics faces a labyrinth of overlapping concerns: *what skills do I actually need?*, *is my timeline realistic?*, *where do I even start?*

Generic career advice is abundantly available on the internet, but it is exactly that — generic. It is rarely calibrated to an individual's specific background, target role, or available timeline. Humans offering personalized career coaching are expensive and inaccessible to most professionals. Self-guided research is fragmented, time-consuming, and often outdated.

**Core problem:** There is no accessible, personalized, and evidence-based system that can take a professional's specific background and generate a concrete, time-bound roadmap toward a new tech role.

`Career guidance` · `Skill gap identification` · `Transition planning` · `Personalization at scale`

---

## 02 — Who It Is Built For

The system is designed for working professionals who are actively considering or actively pursuing a role change within the technology sector. The target user provides four inputs:

| Input | Purpose |
|---|---|
| 👤 **Name** | Personalizes the generated roadmap output for a more direct and motivating experience. |
| 📋 **Current Background** | A free-text description of their current role, experience level, and existing skills. This anchors the gap analysis. |
| 🎯 **Target Role** | The specific tech role they are aiming for — e.g., Data Analyst, DevOps Engineer, Product Manager. |
| ⏱️ **Timeline** | A self-selected window: 3, 6, 9, or 12 months. Shapes the realism and pacing of the generated plan. |

---

## 03 — How the System Approaches the Problem

The project's core architectural bet is that a single LLM prompt cannot produce a high-quality, personalized career plan. Instead, the problem is decomposed into distinct reasoning steps, each handled by a specialized AI agent. The five agents run **sequentially in a pipeline**, with each agent's output flowing into the next as structured input.

### Agent 01 — Transition Researcher
`transition_researcher.py` · Tavily + Groq/Gemini

Performs live web searches via the Tavily API to gather real-world evidence — career transition success stories, current role requirements, and realistic timeline data. It uses an agentic tool-calling loop, letting the LLM decide when it has enough evidence to stop searching. Produces a structured 5-section research summary.

### Agent 02 — Skill Gap Analyzer
`skill_gap_analyzer.py` · Groq/Gemini JSON

Compares the user's stated background against the research findings to produce strict JSON output: a list of missing skills, required tools/platforms, and real-world experience gaps. Structured JSON output ensures downstream agents receive reliable, machine-parseable data rather than free text.

### Agent 03 — Path Analyzer
`path_analyzer.py` · Groq/Gemini JSON

Takes the skill gaps and timeline as input and produces a strict 30/60/90-day JSON plan. Each phase includes weekly tasks (4–8 items), high-value resources (3–6), and measurable milestones (3–5). The plan is explicitly calibrated to the user's timeline constraint.

### Agent 04 — Roadmap Writer
`roadmap_writer.py` · Groq/Gemini text

Converts the structured JSON plan into a polished, human-readable markdown roadmap with five required sections: Welcome Note, Goals, Weekly Plan, Resources, and Milestone Checkpoints. The writing style is intentionally professional and confidence-building — designed to motivate, not just inform.

### Agent 05 — Judge Agent
`judge_agent.py` · Groq/Gemini JSON

Acts as a strict independent evaluator of the generated roadmap. Scores the output across four criteria — role specificity, realism, completeness, and readability — each on a 0–10 scale. Produces a verdict ("Strong", "Promising", or "Needs Improvement"), a list of strengths, and a single highest-impact improvement recommendation.

---

## 04 — Key Technical Design Choices

Several architectural decisions reflect deliberate tradeoffs made in the implementation:

**🔄 Dual LLM Provider Support**
The `GeminiClient` wrapper auto-detects whether a Groq key (`gsk_…`) or Gemini key (`AIza…`) is configured, routing to the appropriate provider. This means the system works with either API without code changes.

**🔁 Agentic Tool-Calling Loop**
The researcher uses a multi-turn loop (up to 12 turns) where the LLM autonomously decides when to call the Tavily search tool again and when to stop — rather than a fixed number of searches.

**📐 Strict JSON Contracts**
Agents 2, 3, and 5 demand exact JSON schemas from the LLM with no extra keys, no markdown, and no prose — enforced by prompt design. This makes the pipeline robust to format drift.

**🧩 Adapter Pattern**
Each agent has two classes: a reusable core (e.g., `SkillGapAnalyzer`) and a backward-compatible adapter (e.g., `SkillGapAnalyzerAgent`). This decouples internal implementation from the pipeline interface.

**📦 Dataclass Schemas**
All inter-agent data is passed as typed Python dataclasses (`ResearchOutput`, `SkillGapOutput`, `PathOutput`, etc.), providing type safety and a clear contract between pipeline stages.

**⚡ Streamlit UI**
A minimal form-based UI captures the four user inputs and renders the markdown roadmap and judge scores inline. A debug expander reveals the full structured JSON output from all agents.

---

## 05 — Observed Limitations

- The pipeline is **fully sequential** with no parallelism. With up to 12 agentic turns in the researcher plus 4 additional LLM calls, latency can be substantial on complex transitions.
- The **30/60/90-day plan is always used** regardless of the user's selected timeline (3, 6, 9, or 12 months). The path analyzer acknowledges the timeline in its prompt, but the structural output remains a 30/60/90 framework — which maps poorly to a 12-month plan.
- The researcher's skill extraction logic in the adapter (parsing bullet points from free text) is fragile — if the LLM response format varies, the `role_requirements` list falls back to three generic placeholder strings.
- There is **no persistent storage**. Every session re-runs the full 5-agent pipeline from scratch. Generated plans cannot be saved, iterated on, or compared.
- The judge agent scores the roadmap it reviews but **there is no feedback loop**. A low judge score does not trigger revision. The pipeline produces one output regardless of quality.
- API key handling allows typos in environment variable names (`GROK_API_KEY` instead of `GROQ_API_KEY`), which can cause silent fallback behavior that is difficult to debug.

---

## 06 — Summary

| Dimension | Detail |
|---|---|
| **Problem Domain** | Personalized tech career transition planning |
| **Approach** | Sequential 5-agent LLM pipeline |
| **Data Source** | Live web search via Tavily API |
| **LLM Providers** | Groq (Llama 3.3 70B) or Google Gemini |
| **User Interface** | Streamlit web app, 4-field form |
| **Key Output** | Markdown roadmap + quality score/verdict |
| **Strongest Design Choice** | Strict JSON agent contracts + agentic search loop |
| **Primary Gap** | No revision loop despite judge scoring |

The project successfully demonstrates that decomposing a complex reasoning task into specialized agents — each with a strict input/output contract — produces more structured, evaluable, and improvable outputs than a single monolithic prompt. The inclusion of a Judge Agent that scores but does not yet revise represents the most natural next evolution of the system.
