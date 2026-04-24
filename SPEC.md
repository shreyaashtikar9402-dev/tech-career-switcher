# Tech Career Switcher AI — Task Decomposition & Spec Document

> **Project:** `tech-career-switcher`
> **Stack:** Python 3.11 · Streamlit · Groq (LLaMA 3.3-70B) · Tavily Search · Google Gemini (optional)
> **Purpose:** A multi-agent AI system that generates a personalised career transition roadmap for users switching into a tech role.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Summary](#2-architecture-summary)
3. [Module Breakdown & Task Decomposition](#3-module-breakdown--task-decomposition)
   - 3.1 Entry Point — `app.py`
   - 3.2 Data Models — `models/schemas.py`
   - 3.3 LLM Client — `utils/gemini_client.py`
   - 3.4 Agentic Loop — `utils/agentic_loop.py`
   - 3.5 Agent 1 — Transition Researcher
   - 3.6 Agent 2 — Skill Gap Analyzer
   - 3.7 Agent 3 — Path Analyzer
   - 3.8 Agent 4 — Roadmap Writer
   - 3.9 Agent 5 — Judge Agent
4. [Data Flow](#4-data-flow)
5. [Environment & Configuration](#5-environment--configuration)
6. [Known Issues & Improvement Areas](#6-known-issues--improvement-areas)
7. [Suggested Next Tasks](#7-suggested-next-tasks)

---

## 1. Project Overview

Tech Career Switcher AI accepts four user inputs — name, current professional background, target tech role, and a transition timeline — and runs them through a five-agent pipeline. Each agent builds on the previous agent's output. The final deliverables are a structured markdown roadmap and a quality score with actionable feedback.

**Key APIs used:**

| Service | Purpose | Key Env Var |
|---|---|---|
| Groq (LLaMA 3.3-70B) | LLM inference for all agents | `GROQ_API_KEY` |
| Tavily | Real-time web search for market research | `TAVILY_API_KEY` |
| Google Gemini (optional) | Alternative LLM provider | `GEMINI_API_KEY` |

---

## 2. Architecture Summary

```
User Input (Streamlit Form)
        │
        ▼
  app.py :: run_pipeline()
        │
        ├──► Agent 1: TransitionResearcher     → ResearchOutput
        ├──► Agent 2: SkillGapAnalyzer         → SkillGapOutput
        ├──► Agent 3: PathAnalyzer             → PathOutput
        ├──► Agent 4: RoadmapWriter            → RoadmapOutput
        └──► Agent 5: JudgeAgent               → JudgeOutput
                │
                ▼
        Streamlit renders Roadmap Markdown + Judge Score
```

Each agent follows the same pattern: a **reusable core class** (e.g., `TransitionResearcher`) that handles all logic, wrapped by a **backward-compatible adapter** (e.g., `TransitionResearcherAgent`) that maps inputs/outputs to the shared schema defined in `models/schemas.py`.

---

## 3. Module Breakdown & Task Decomposition

---

### 3.1 Entry Point — `app.py`

**File:** `app.py`

**Responsibility:** Renders the Streamlit UI, collects user inputs, orchestrates the agent pipeline, and displays results.

**Key functions:**

- `run_pipeline(user_profile: UserProfile) -> dict` — Instantiates all five agents and calls them sequentially. Returns a dict with keys: `research`, `skill_gap`, `path`, `roadmap`, `judge`.
- `main()` — Sets up Streamlit page config, renders the input form, validates inputs, calls `run_pipeline`, and renders all outputs including roadmap markdown, judge score, and a debug expander with raw JSON.

**Error handling covers:**
- `PermissionDeniedError` — Groq plan/limits issue
- `AuthenticationError` — Invalid API key
- `APIStatusError` — HTTP-level Groq failures
- Generic `Exception` — Displayed via `st.exception()`

**Inputs collected from UI:**

| Field | Type | Example |
|---|---|---|
| Name | Text | Priya |
| Current Background | Text area | "2 years in support engineering with basic Python and SQL" |
| Target Role | Text | Data Analyst |
| Timeline | Select | 3 / 6 / 9 / 12 months |

---

### 3.2 Data Models — `models/schemas.py`

**File:** `models/schemas.py`

**Responsibility:** Defines all dataclasses used as typed contracts between agents. All classes use `@dataclass(slots=True)` for memory efficiency.

**Dataclasses defined:**

| Class | Purpose |
|---|---|
| `UserProfile` | Stores raw user inputs: `name`, `current_background`, `target_role`, `timeline` |
| `SearchInsight` | A single Tavily search result: `title`, `url`, `snippet` |
| `ResearchOutput` | Output of Agent 1: `role_market_summary`, `role_requirements`, `insights` |
| `SkillGapOutput` | Output of Agent 2: `current_strengths`, `missing_skills`, `priority_skills`, `rationale` |
| `PlanMilestone` | One phase of the plan: `focus`, `goals`, `deliverables`, `resources` |
| `PathOutput` | Output of Agent 3: `plan_json` (dict of `PlanMilestone`), `success_metrics` |
| `RoadmapOutput` | Output of Agent 4: `roadmap_markdown` (plain markdown string) |
| `JudgeOutput` | Output of Agent 5: `score`, `max_score`, `verdict`, `strengths`, `improvements`, `explanation` |

**Utility function:**

- `as_json_safe(value)` — Recursively converts dataclass instances, lists, and dicts to plain JSON-serialisable types. Used by the debug expander in `app.py`.

---

### 3.3 LLM Client — `utils/gemini_client.py`

**File:** `utils/gemini_client.py`

**Responsibility:** Unified LLM client that auto-selects the provider based on the API key prefix. Abstracts prompt execution from all agent classes.

**Provider detection logic:**

- Key starts with `AIza` → Google Gemini (uses `google.generativeai` SDK)
- Key starts with `gsk_` → Groq (uses OpenAI-compatible client pointed at `https://api.groq.com/openai/v1`)

**Public methods:**

| Method | Returns | Notes |
|---|---|---|
| `generate_text(prompt)` | `str` | Sends prompt, returns raw text response |
| `generate_json(prompt)` | `dict` | Calls `generate_text`, strips markdown fences, parses JSON |

**JSON extraction logic (`_extract_json_block`):** Strips triple-backtick fences and optional `json` language hint before parsing. Raises `ValueError` if the result is not a valid JSON object.

**Env vars resolved (in order):** `GEMINI_API_KEY` → `GROQ_API_KEY` → `GROK_API_KEY`

---

### 3.4 Agentic Loop — `utils/agentic_loop.py`

**File:** `utils/agentic_loop.py`

**Responsibility:** Implements a reusable multi-turn tool-calling loop over the Groq OpenAI-compatible API. Used exclusively by `TransitionResearcher` for web search orchestration.

**Classes:**

**`AgenticLoopConfig`** — Configuration dataclass:
- `model_name` (default: `llama-3.3-70b-versatile`)
- `max_turns` (1–50, default: 12)
- `temperature` (0–2, default: 0.2)

**`AgenticLoopResult`** — Result dataclass:
- `final_text` — The model's last plain-text response
- `turns_used` — Number of loop iterations executed
- `tool_calls` — Total tool invocations made
- `stopped_reason` — Either `"model_stopped_calling_tools"` or `"max_turns_reached"`
- `transcript` — Full per-turn log with model text and function calls

**`GeminiAgenticLoop`** — Main loop class:
- Accepts tool declarations (Gemini-style or OpenAI-style) and normalises them to OpenAI function-calling format via `_normalize_tools()`.
- On each turn: calls the model → checks for tool calls → executes them via registered `tool_executors` → appends results → repeats until no tool call is emitted.
- `_execute_tool()` wraps executors in try/except and returns `{"error": ...}` on failure instead of raising.
- `_make_json_safe()` recursively sanitises tool results before serialisation.

---

### 3.5 Agent 1 — Transition Researcher

**File:** `agents/transition_researcher.py`

**Responsibility:** Gathers real-world evidence about the target role transition using Tavily web search, then summarises findings into a structured text report.

**Core class: `TransitionResearcher`**

Supports two execution paths depending on which API keys are present:

**Path A — Groq + Agentic Loop (default):**
- Creates a `GeminiAgenticLoop` with the `tavily_transition_search` tool registered.
- Sends a structured user prompt asking for five output sections: Transition Snapshot, Success Story Patterns, Required Skills, Timeline Reality Check, Actionable Recommendations.
- The model decides autonomously how many times to call Tavily (up to `max_turns=12`).

**Path B — Gemini direct (fallback):**
- Issues three fixed Tavily queries: success stories, required skills, realistic timeline.
- Concatenates up to 12 search result snippets.
- Sends the combined evidence block to Gemini via `generate_text()` for summarisation.

**Tavily tool declaration (`tavily_transition_search`):**
- Parameters: `query` (required), `search_focus` (`success_stories` | `required_skills` | `timelines`), `max_results` (1–8)
- Returns: normalised list of `{title, url, snippet, focus}` objects

**Adapter class: `TransitionResearcherAgent`**
- Calls `TransitionResearcher.run()` and maps the result to `ResearchOutput`.
- Extracts `role_requirements` by line-scanning for bullet points under the "Required Skills" section heading.
- Falls back to three generic requirement strings if none are found.

**Output schema:** `ResearchOutput` — `role_market_summary` (first 700 chars of summary), `role_requirements` (up to 10 items), `insights` (single agentic metadata entry).

---

### 3.6 Agent 2 — Skill Gap Analyzer

**File:** `agents/skill_gap_analyzer.py`

**Responsibility:** Compares the user's background against the research findings to identify missing skills, required tools, and experience gaps.

**Core class: `SkillGapAnalyzer`**

- Sends a structured prompt to `GeminiClient.generate_json()`.
- Prompt instructs the model to return strict JSON with exactly three keys:
  - `missing_skills` — 5–12 concise skill labels
  - `tools` — Practical tools, platforms, and frameworks
  - `experience_gaps` — Portfolio or real-world exposure gaps

**Input:** `SkillGapAnalyzerInput` with `background` and `research_output` strings.

**Output:** `SkillGapAnalyzerResult` with `missing_skills`, `tools`, `experience_gaps` lists.

**Adapter class: `SkillGapAnalyzerAgent`**
- Constructs the `research_output` string from `ResearchOutput` fields.
- Maps result to `SkillGapOutput`:
  - `current_strengths` → always empty (not derived by this agent)
  - `missing_skills` → direct from analyzer
  - `priority_skills` → first 5 items of `missing_skills`
  - `rationale` → concatenated tools and experience gaps string

---

### 3.7 Agent 3 — Path Analyzer

**File:** `agents/path_analyzer.py`

**Responsibility:** Generates a structured 30/60/90-day action plan tailored to the user's target role and skill gaps.

**Core class: `PathAnalyzer`**

- Sends a structured prompt to `GeminiClient.generate_json()`.
- Prompt instructs the model to return strict JSON with three top-level keys: `30_days`, `60_days`, `90_days`.
- Each phase contains:
  - `weekly_tasks` — 4–8 actionable tasks
  - `resources` — 3–6 high-value resources (courses, docs, communities)
  - `milestones` — 3–5 measurable outcomes

**Input:** `PathAnalyzerInput` with `target_role`, `timeline`, `research_output`, `skill_gaps` dict.

**Output:** `PathAnalyzerResult` wrapping `plan_30_60_90` dict.

**Adapter class: `PathAnalyzerAgent`**
- Constructs input from `UserProfile` and `SkillGapOutput`.
- Maps each phase to a `PlanMilestone` dataclass:
  - `focus` → auto-generated label (e.g., "30 Days role transition execution")
  - `goals` → first 6 weekly tasks
  - `deliverables` → first 5 milestones
  - `resources` → first 6 resources
- Appends three fixed `success_metrics` strings to `PathOutput`.

---

### 3.8 Agent 4 — Roadmap Writer

**File:** `agents/roadmap_writer.py`

**Responsibility:** Converts the structured JSON plan and skill context into a polished, motivating markdown roadmap document.

**Core class: `RoadmapWriter`**

- Sends a structured prompt to `GeminiClient.generate_text()` (not JSON — returns markdown directly).
- Prompt requires five sections in exact order:
  1. Welcome Note
  2. Goals
  3. Weekly Plan
  4. Resources
  5. Milestone Checkpoints
- Tone: professional, practical, and confidence-building.

**Input:** `RoadmapWriterInput` with `user_name`, `background`, `target_role`, `timeline`, `json_plan`.

**Output:** `RoadmapWriterResult` with `markdown_roadmap` string.

**Adapter class: `RoadmapWriterAgent`**
- Composes a richer `json_plan` dict that merges `PathOutput.plan_json`, `success_metrics`, `priority_skills`, and `missing_skills` to give the writer agent more context.
- Returns `RoadmapOutput(roadmap_markdown=...)`.

---

### 3.9 Agent 5 — Judge Agent

**File:** `agents/judge_agent.py`

**Responsibility:** Evaluates the generated roadmap across four quality dimensions and returns a numeric score with a verdict and improvement suggestions.

**Core class: `RoadmapJudge`**

- Sends a structured prompt to `GeminiClient.generate_json()`.
- Prompt instructs the model to score the roadmap on four criteria (0–10 each):
  - `role_specificity`
  - `realism`
  - `completeness`
  - `readability`
- Expected JSON schema:
  ```json
  {
    "scores": { "role_specificity": 0, "realism": 0, "completeness": 0, "readability": 0 },
    "overall_score": 0,
    "summary": "40-90 word summary",
    "improvement": "single highest-impact recommendation"
  }
  ```

**Input:** `JudgeInput` with `background`, `target_role`, `timeline`, `roadmap_markdown`.

**Output:** `JudgeResult` with `scores` dict, `overall_score`, `summary`, `improvement`.

**Adapter class: `JudgeAgent`**
- Maps `JudgeResult` to `JudgeOutput` used by `app.py`:
  - Derives `strengths` list by checking if individual dimension scores are ≥ 7.
  - Sets `verdict` to `"Strong"` (≥ 8), `"Promising"` (≥ 6), or `"Needs Improvement"` (< 6).
  - `improvements` list always contains one item: the model's top recommendation.
  - `max_score` is hardcoded to `10.0`.

---

## 4. Data Flow

```
UserProfile
    │
    ▼
TransitionResearcherAgent.run(user)
    → Tavily search (1–3 queries, up to 12 agentic turns)
    → LLM summarisation
    → ResearchOutput

    │
    ▼
SkillGapAnalyzerAgent.run(user, research_output)
    → LLM JSON generation
    → SkillGapOutput

    │
    ▼
PathAnalyzerAgent.run(user, skill_gap_output)
    → LLM JSON generation (30/60/90-day plan)
    → PathOutput

    │
    ▼
RoadmapWriterAgent.run(user, skill_gap_output, path_output)
    → LLM markdown generation
    → RoadmapOutput

    │
    ▼
JudgeAgent.run(user, skill_gap_output, path_output, roadmap_output)
    → LLM JSON scoring
    → JudgeOutput

    │
    ▼
Streamlit renders: roadmap_markdown + judge score + debug JSON
```

---

## 5. Environment & Configuration

**Required env vars:**

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes (unless using Gemini) | Groq LLM API key (prefix: `gsk_`) |
| `TAVILY_API_KEY` | Yes | Tavily search API key |
| `GROQ_MODEL` | No | Default: `llama-3.3-70b-versatile` |
| `GEMINI_API_KEY` | No | Google Gemini API key (prefix: `AIza`) |
| `GEMINI_MODEL` | No | Default: `gemini-1.5-flash` |

**Provider selection:** All agent classes check keys in this order: `GROQ_API_KEY` → `GROK_API_KEY` → `GEMINI_API_KEY`. If `GEMINI_API_KEY` is set and starts with `AIza`, Gemini is used; otherwise Groq is assumed.

**Setup steps:**

```bash
# 1. Install Python 3.11
# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 5. Run
streamlit run app.py
```

---

## 6. Known Issues & Improvement Areas

**1. `current_strengths` is always empty**
`SkillGapAnalyzerAgent` sets `current_strengths=[]` unconditionally. The LLM prompt does not ask for strengths, so this field is never populated even though it exists in the schema.

**2. Provider key resolution is duplicated**
Every agent class contains its own copy of the key-resolution logic (`GROQ_API_KEY` → `GROK_API_KEY` → `GEMINI_API_KEY`). This should be extracted into a shared utility function to avoid drift.

**3. `agentic_loop.py` is named `GeminiAgenticLoop` but uses Groq**
The class name is misleading — it uses the OpenAI-compatible Groq endpoint, not the Gemini SDK. Renaming to `GroqAgenticLoop` would reduce confusion.

**4. `TransitionResearcherAgent` uses a brittle text parser**
The adapter parses `role_requirements` by scanning for bullet characters and numbered list prefixes in the LLM's free-text output. This is fragile and will silently fall back to generic defaults if the LLM changes its formatting.

**5. No streaming support**
All five agents run sequentially and only display results after the entire pipeline finishes. Long pipelines (especially with 12 agentic loop turns in Agent 1) will show a spinner for 30–90 seconds with no intermediate feedback.

**6. `max_score` is hardcoded**
`JudgeAgent` always sets `max_score=10.0`. If the scoring rubric changes, this needs a manual update.

---

## 7. Suggested Next Tasks

The following improvements are scoped as individual, independently deliverable tasks suitable for GitHub issues.

| # | Task | Priority | Effort |
|---|---|---|---|
| T-01 | Extract provider key resolution into `utils/api_keys.py` shared helper | High | Small |
| T-02 | Populate `current_strengths` in `SkillGapAnalyzer` prompt and output | Medium | Small |
| T-03 | Rename `GeminiAgenticLoop` → `GroqAgenticLoop` across all files | Low | Small |
| T-04 | Add Streamlit `st.status()` or per-agent progress updates to the UI | High | Medium |
| T-05 | Replace brittle text parser in `TransitionResearcherAgent` with a structured JSON sub-call | High | Medium |
| T-06 | Add unit tests for `GeminiClient._extract_json_block()` and `AgenticLoopConfig` validation | Medium | Medium |
| T-07 | Add a Dockerfile and `docker-compose.yml` for one-command local setup | Medium | Medium |
| T-08 | Implement roadmap export as a downloadable `.md` or `.pdf` file via Streamlit | Medium | Medium |
| T-09 | Add retry logic with exponential backoff for Groq/Tavily API rate-limit errors | High | Medium |
| T-10 | Support parallel execution of Agents 2–4 (they do not depend on each other) | Low | Large |

---

*Document generated from source analysis of `tech-career-switcher-main.zip`.*
