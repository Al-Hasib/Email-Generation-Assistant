# Email Generation Assistant

AI-powered email generation with LangGraph workflows, custom evaluation metrics, and multi-model comparison.

## Project Structure

```
├── src/                    # Core generation logic
│   ├── config.py           # YAML + .env config (Groq API, model settings)
│   ├── email_generator.py  # Orchestrates model → generation
│   └── prompt_templates.py # LangChain prompts + LangGraph workflow defs
├── models/                 # Model implementations
│   ├── base.py             # Abstract base class (EmailInput, EmailResult)
│   ├── model_a.py          # Model A: COT + Review Loop (LangGraph)
│   ├── model_b.py          # Model B: Role-Playing Only (LangChain)
│   └── model_c.py          # Model C: Self-Reflection Agent (LangGraph)
├── evaluation/             # Evaluation framework
│   ├── test_scenarios.py   # 10 test scenarios with reference emails
│   ├── metrics.py          # 3 custom evaluation metrics
│   ├── evaluator.py        # Sync + async scenario runner
│   └── report.py           # CSV, JSON, and HTML reports
├── results/                # Output reports
├── run_evaluation.py       # Main entry point (all 3 models)
├── config.yaml             # Optional YAML config (overrides .env)
├── Dockerfile              # Containerised execution
├── docker-compose.yml      # Orchestration with env & volume mounts
├── .env.example            # Environment variable template
└── requirements.txt
```

## Tech Stack

- **LLM Provider**: Groq (via `langchain-groq` / `ChatGroq`)
- **Orchestration**: LangChain (`ChatPromptTemplate`, LCEL chains)
- **Workflow Engine**: LangGraph (`StateGraph` with conditional edges)
- **Evaluation Judge**: Groq LLM-as-a-Judge (TAS & OQS metrics)

## Models

| Model | Graph | Nodes | Description |
|-------|-------|-------|-------------|
| **A** | Plan → Write → Review → (accept / rewrite →) | 4 | COT with one review pass; catches tone/fact errors |
| **B** | (single chain) | 1 | Role-Playing only; no graph, no review |
| **C** | Plan → Write → Review → (accept / rewrite up to 2×) | 4 | Strict self-reflection with up to 2 rewrites |

## Setup

```bash
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your GROQ_API_KEY (get one at https://console.groq.com)
```

Optionally edit `config.yaml` to override model names, temperature, or other settings.

## Usage

```bash
python run_evaluation.py
```

### Docker

```bash
docker compose up --build
```

Results (CSV, JSON, HTML) appear in `./results/`.

## Custom Evaluation Metrics

| Metric | Method | Description |
|--------|--------|-------------|
| **Fact Recall Accuracy (FRA)** | Automated keyword matching (50% threshold) | Fraction of required key facts present in the output |
| **Tone Adherence Score (TAS)** | LLM-as-a-Judge (1–5, normalized to 0–1) | How well the email matches the requested tone |
| **Overall Quality Score (OQS)** | LLM-as-a-Judge (1–5, normalized to 0–1) | Structure, clarity, grammar, CTA, fact integration |
