# 🎯 Appetite Matcher

**Solving the "Spray and Pray" Problem in Commercial Insurance**

Brokers often blast submissions to 20+ carriers, most of whom have zero interest. Underwriters waste ~30% of their day declining out-of-appetite deals. This AI agent fixes that.

## What It Does

1. **Reads** a carrier's complex Appetite Guide (PDF) and extracts structured appetite criteria
2. **Scans** a broker's Submission Email (unstructured text) and extracts risk details
3. **Matches** the submission against appetite using industry keyword matching and risk profile alignment
4. **Reasons** about edge cases: explains *why* something is flagged, not just pass/fail

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                     │
│   Upload Appetite PDF + Paste Submission Email    │
└──────────────────────┬──────────────────────────┘
                       │
              ┌────────▼────────┐
              │    LangGraph    │
              │   Orchestrator  │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐ ┌──────────┐ ┌──────────────┐
   │ Parse    │ │ Parse    │ │ Reasoning    │
   │ Appetite │ │ Submission│ │ Gate         │
   │ Guide    │ │ Email    │ │ (Match +     │
   │ (PDF→   │ │ (Text→   │ │  Explain)    │
   │ Pydantic)│ │ Pydantic)│ │              │
   └──────────┘ └──────────┘ └──────────────┘
```

## Tech Stack

| Component       | Tool       | Why                                              |
|----------------|------------|--------------------------------------------------|
| Orchestration  | LangGraph  | Multi-step agent workflows with conditional edges |
| Validation     | Pydantic   | Structured LLM output that won't break downstream |
| UI / Demo      | Streamlit  | Non-technical users can "play" with the agent     |
| Observability  | LangSmith  | Debugging, hallucination tracking, tracing        |
| LLM            | Groq (Llama 3.3 70B) | Fast inference via Groq API             |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export GROQ_API_KEY="your-key"
export LANGSMITH_API_KEY="your-key"        # optional
export LANGSMITH_TRACING=true              # optional

# 3. Run the app
streamlit run app.py
```

## Project Structure

```
appetite-matcher/
├── app.py                  # Streamlit UI
├── graph.py                # LangGraph workflow
├── appetite_schemas.py     # Pydantic models for appetite criteria
├── submission_schemas.py   # Pydantic models for submission data
├── pdf_parser.py           # PDF text extraction
├── data/
│   ├── sample_guides/      # Example appetite guide PDFs
│   └── sample_submissions/ # Example submission emails
├── requirements.txt
└── README.md
```

## Sample Usage

The app ships with a **sample Appetite Guide** (Keystone Mutual) and a **sample Submission Email** (roofing contractor). Try it to see the Reasoning Gate in action — it will flag the building age limit violation and recommend a premium load.

## Key Design Decisions

- **Pydantic everywhere**: Every LLM output is validated through Pydantic models. If the LLM hallucinates a field or returns bad types, we catch it immediately.
- **LangGraph conditional edges**: The graph routes to different nodes based on parsing confidence. Low-confidence parses trigger a re-extraction step.
- **Reasoning Gate**: Not just pass/fail — the agent explains each criterion match/mismatch and provides actionable recommendations (premium loads, exclusions, or rejection).
