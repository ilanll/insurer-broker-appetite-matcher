"""
Appetite Matcher — Streamlit UI
Solves the "Spray and Pray" problem in commercial insurance.
"""

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

from appetite_schemas import AppetiteCriteria
from submission_schemas import MatchResult, MatchVerdict, SubmissionData
from pdf_parser import extract_text_from_pdf
from graph import build_graph, run_appetite_match

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Appetite Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container */
    .block-container { max-width: 1200px; padding-top: 2rem; }

    /* Verdict badges */
    .verdict-accept {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white; padding: 1rem 2rem; border-radius: 12px;
        font-size: 1.3rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 15px rgba(5,150,105,0.3);
    }
    .verdict-reject {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white; padding: 1rem 2rem; border-radius: 12px;
        font-size: 1.3rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 15px rgba(220,38,38,0.3);
    }
    .verdict-refer {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white; padding: 1rem 2rem; border-radius: 12px;
        font-size: 1.3rem; font-weight: 700; text-align: center;
        box-shadow: 0 4px 15px rgba(217,119,6,0.3);
    }

    /* Criterion cards */
    .criterion-pass {
        border-left: 4px solid #10b981;
        padding: 0.75rem 1rem; margin: 0.5rem 0;
        background: #f0fdf4; border-radius: 0 8px 8px 0;
    }
    .criterion-fail-hard {
        border-left: 4px solid #ef4444;
        padding: 0.75rem 1rem; margin: 0.5rem 0;
        background: #fef2f2; border-radius: 0 8px 8px 0;
    }
    .criterion-fail-soft {
        border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem; margin: 0.5rem 0;
        background: #fffbeb; border-radius: 0 8px 8px 0;
    }

    /* Risk score gauge */
    .risk-gauge {
        text-align: center; padding: 1.5rem;
        background: #f8fafc; border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .risk-score-number {
        font-size: 3rem; font-weight: 800; line-height: 1;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sample data loader
# ---------------------------------------------------------------------------
SAMPLE_DIR = Path(__file__).parent / "data"


def load_sample_appetite() -> str:
    path = SAMPLE_DIR / "sample_guides" / "keystone_mutual_appetite.txt"
    if path.exists():
        return path.read_text()
    return ""


def load_sample_submissions() -> dict[str, str]:
    sub_dir = SAMPLE_DIR / "sample_submissions"
    subs = {}
    if sub_dir.exists():
        for f in sorted(sub_dir.glob("*.txt")):
            subs[f.stem.replace("_", " ").title()] = f.read_text()
    return subs


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Required to run the agent. Set GROQ_API_KEY env var or enter here.",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()

    langsmith_key = st.text_input(
        "LangSmith API Key (optional)",
        type="password",
        value=os.environ.get("LANGSMITH_API_KEY", ""),
        help="Enable LangSmith tracing for observability.",
    )
    if langsmith_key:
        os.environ["LANGSMITH_API_KEY"] = langsmith_key
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = "appetite-matcher"
        st.success("✓ LangSmith tracing enabled")

    st.divider()
    st.markdown("### About")
    st.markdown(
        "**Appetite Matcher** uses a LangGraph agent pipeline to parse "
        "carrier appetite guides, extract submission details, and provide "
        "structured underwriting recommendations with full reasoning."
    )
    st.markdown(
        "**Stack:** LangGraph · Pydantic · Groq (Llama 3.3 70B) · Streamlit · LangSmith"
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown("# 🎯 Appetite Matcher")
st.markdown("*Stop spraying submissions. Start matching them.*")
st.markdown("---")

# --- Input tabs ---
tab_appetite, tab_submission = st.tabs(["📋 Carrier Appetite Guide", "📧 Broker Submission"])

with tab_appetite:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### Upload or paste the carrier's Appetite Guide")
    with col2:
        use_sample_appetite = st.button("Load Sample Guide", type="secondary")

    uploaded_pdf = st.file_uploader(
        "Upload Appetite Guide PDF",
        type=["pdf", "txt"],
        help="Upload the carrier's appetite guide as PDF or plain text.",
    )

    appetite_text = st.text_area(
        "Or paste the appetite guide text below:",
        height=300,
        key="appetite_input",
        value=load_sample_appetite() if use_sample_appetite else "",
    )

    # Process uploaded file
    if uploaded_pdf is not None:
        if uploaded_pdf.name.endswith(".pdf"):
            with st.spinner("Extracting text from PDF..."):
                appetite_text = extract_text_from_pdf(uploaded_pdf.read())
            st.success(f"✓ Extracted {len(appetite_text):,} characters from PDF")
        else:
            appetite_text = uploaded_pdf.read().decode("utf-8")
            st.success(f"✓ Loaded {len(appetite_text):,} characters")

with tab_submission:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### Paste the broker's submission email")
    with col2:
        samples = load_sample_submissions()
        sample_choice = st.selectbox(
            "Load sample:",
            ["—"] + list(samples.keys()),
            label_visibility="collapsed",
        )

    default_sub = samples.get(sample_choice, "") if sample_choice != "—" else ""
    submission_text = st.text_area(
        "Submission email text:",
        height=300,
        key="submission_input",
        value=default_sub,
    )

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
st.markdown("---")
run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
with run_col2:
    run_button = st.button(
        "🔍  Run Appetite Match",
        type="primary",
        use_container_width=True,
        disabled=not (appetite_text and submission_text),
    )

if not appetite_text or not submission_text:
    st.info("Upload or paste both an Appetite Guide and a Submission Email to begin.")
    st.stop()

if not run_button and "match_result" not in st.session_state:
    st.stop()

if run_button:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please provide a Groq API key in the sidebar.")
        st.stop()

    # Run the agent pipeline
    with st.status("Running Appetite Match pipeline...", expanded=True) as status:
        st.write("**Step 1/4:** Parsing appetite guide...")
        t0 = time.time()

        try:
            match_result, full_state = run_appetite_match(appetite_text, submission_text)
            elapsed = time.time() - t0
            status.update(label=f"Analysis complete ({elapsed:.1f}s)", state="complete")

            # Cache results
            st.session_state["match_result"] = match_result.model_dump()
            st.session_state["full_state"] = full_state
        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Pipeline error: {str(e)}")
            st.stop()

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "match_result" not in st.session_state:
    st.stop()

result = MatchResult(**st.session_state["match_result"])
state = st.session_state.get("full_state", {})

st.markdown("---")
st.markdown("## 📊 Analysis Results")

# --- Top-level verdict ---
verdict_map = {
    MatchVerdict.ACCEPT: ("verdict-accept", "✅ ACCEPT — In Appetite"),
    MatchVerdict.REJECT: ("verdict-reject", "❌ REJECT — Out of Appetite"),
    MatchVerdict.REFER: ("verdict-refer", "⚠️ REFER — Needs Underwriter Review"),
}
css_class, verdict_label = verdict_map[result.verdict]
st.markdown(f'<div class="{css_class}">{verdict_label}</div>', unsafe_allow_html=True)

# --- Metrics row ---
st.markdown("")
m1, m2, m3, m4 = st.columns(4)
with m1:
    confidence_pct = f"{result.confidence:.0%}"
    st.metric("Confidence", confidence_pct)
with m2:
    risk_color = "🟢" if result.risk_score < 30 else "🟡" if result.risk_score < 60 else "🔴"
    st.metric("Risk Score", f"{risk_color} {result.risk_score:.0f}/100")
with m3:
    adj = result.premium_adjustment_pct
    if adj is not None:
        st.metric("Premium Adjustment", f"{adj:+.0f}%")
    else:
        st.metric("Premium Adjustment", "None")
with m4:
    passed = sum(1 for cr in result.criteria_results if cr.passed)
    total = len(result.criteria_results)
    st.metric("Criteria Passed", f"{passed}/{total}")

# --- Executive summary ---
st.markdown("### 💡 Executive Summary")
st.info(result.overall_summary)

# --- Criteria detail ---
st.markdown("### 🔎 Criterion-by-Criterion Breakdown")

for cr in result.criteria_results:
    if cr.passed:
        css = "criterion-pass"
        icon = "✅"
    elif cr.severity == "hard_fail":
        css = "criterion-fail-hard"
        icon = "❌"
    else:
        css = "criterion-fail-soft"
        icon = "⚠️"

    rec_html = ""
    if cr.recommendation:
        rec_html = f'<br><strong>💊 Recommendation:</strong> {cr.recommendation}'

    limits_html = ""
    if cr.carrier_limit or cr.submission_value:
        limits_html = (
            f'<br><small style="color:#64748b;">'
            f'Carrier: {cr.carrier_limit or "N/A"} · '
            f'Submission: {cr.submission_value or "N/A"}'
            f'</small>'
        )

    st.markdown(
        f'<div class="{css}">'
        f'<strong>{icon} {cr.criterion_name}</strong><br>'
        f'{cr.explanation}'
        f'{limits_html}'
        f'{rec_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

# --- Recommended actions ---
if result.recommended_actions:
    st.markdown("### 📋 Recommended Actions")
    for i, action in enumerate(result.recommended_actions, 1):
        st.markdown(f"**{i}.** {action}")

# --- Debug: parsed data ---
with st.expander("🔧 Debug: Parsed Appetite Criteria"):
    if state.get("appetite_criteria"):
        st.json(state["appetite_criteria"])
    else:
        st.warning("Appetite criteria not available")

with st.expander("🔧 Debug: Parsed Submission Data"):
    if state.get("submission_data"):
        st.json(state["submission_data"])
    else:
        st.warning("Submission data not available")

with st.expander("🔧 Debug: Full Match Result JSON"):
    st.json(result.model_dump())
