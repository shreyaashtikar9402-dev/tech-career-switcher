# Tech Career Switcher AI

A full multi-agent AI system built with Python 3.11, Streamlit, Groq API, and Tavily API.

## Architecture

- **Agent 1: Transition Researcher** (`agents/transition_researcher.py`)
  - Uses Tavily search to gather role transition insights and requirements.
- **Agent 2: Skill Gap Analyzer** (`agents/skill_gap_analyzer.py`)
  - Compares current background and target role requirements.
- **Agent 3: Path Analyzer** (`agents/path_analyzer.py`)
  - Produces a structured 30/60/90-day plan in JSON format.
- **Agent 4: Roadmap Writer** (`agents/roadmap_writer.py`)
  - Converts structured outputs into a practical markdown roadmap.
- **Agent 5: Judge Agent** (`agents/judge_agent.py`)
  - Scores quality and provides improvement feedback.

## Folder Structure

```text
Tech Career Switcher AI/
├─ app.py
├─ requirements.txt
├─ runtime.txt
├─ .env.example
├─ .gitignore
├─ README.md
├─ agents/
│  ├─ __init__.py
│  ├─ transition_researcher.py
│  ├─ skill_gap_analyzer.py
│  ├─ path_analyzer.py
│  ├─ roadmap_writer.py
│  └─ judge_agent.py
├─ models/
│  ├─ __init__.py
│  └─ schemas.py
└─ utils/
   ├─ __init__.py
   └─ gemini_client.py
```

## Setup

1. Install Python 3.11.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create `.env` from `.env.example` and add your API keys:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

## Run

```bash
streamlit run app.py
```

## Streamlit Inputs

- Name
- Current Background
- Target Role
- Timeline

## Outputs

- Generated markdown roadmap
- Judge score and feedback

## Live App
https://tech-career-switcher-763modqfewb8ya2sz5xgfx.streamlit.app/
