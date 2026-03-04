"""
LangGraph workflow for the Appetite Matcher agent.

Graph Structure:
    parse_appetite -> parse_submission -> match_criteria -> reasoning_gate -> END

Conditional edges route low-confidence parses back for re-extraction.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from appetite_schemas import AppetiteCriteria
from submission_schemas import (
    CriterionResult,
    MatchResult,
    MatchVerdict,
    SubmissionData,
)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """State that flows through the LangGraph nodes."""

    appetite_text: str
    submission_text: str
    appetite_criteria: dict | None
    submission_data: dict | None
    match_result: dict | None
    parse_appetite_attempts: int
    parse_submission_attempts: int
    error: str | None


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def get_llm() -> ChatGroq:
    """Get the Groq LLM instance."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Node: Parse Appetite Guide
# ---------------------------------------------------------------------------

APPETITE_EXTRACTION_PROMPT = """You are an expert insurance underwriting analyst. Extract structured appetite criteria from this carrier's Appetite Guide text.

Return ONLY valid JSON matching this schema (no markdown, no explanation):
{{
    "carrier_name": "string",
    "effective_date": "string or null",
    "target_industries": ["string"],
    "excluded_industries": ["string"],
    "target_states": ["string (2-letter)"],
    "excluded_states": ["string (2-letter)"],
    "risk_tolerance": "low | medium | high",
    "building": {{
        "max_age_years": "int or null",
        "max_stories": "int or null",
        "accepted_construction_types": ["string"],
        "min_fire_protection_grade": "int or null",
        "sprinkler_required": "bool or null"
    }},
    "financial": {{
        "min_annual_revenue": "float or null",
        "max_annual_revenue": "float or null",
        "min_years_in_business": "int or null",
        "max_loss_ratio_3yr": "float or null"
    }},
    "coverage": {{
        "max_tiv": "float or null",
        "max_single_location_tiv": "float or null",
        "lines_offered": ["string"]
    }},
    "preferred_premium_range_min": "float or null",
    "preferred_premium_range_max": "float or null",
    "notes": ["string"]
}}

APPETITE GUIDE TEXT:
{text}
"""


def parse_appetite_node(state: AgentState) -> dict[str, Any]:
    """Extract structured criteria from the appetite guide text."""
    llm = get_llm()
    attempts = state.get("parse_appetite_attempts", 0) + 1

    prompt = APPETITE_EXTRACTION_PROMPT.format(text=state["appetite_text"][:15000])
    response = llm.invoke(prompt)
    content = response.content

    try:
        # Strip markdown fences if present
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

        data = json.loads(text)
        # Validate through Pydantic
        criteria = AppetiteCriteria(**data)
        return {
            "appetite_criteria": criteria.model_dump(),
            "parse_appetite_attempts": attempts,
            "error": None,
        }
    except Exception as e:
        return {
            "appetite_criteria": None,
            "parse_appetite_attempts": attempts,
            "error": f"Appetite parse error (attempt {attempts}): {str(e)[:200]}",
        }


# ---------------------------------------------------------------------------
# Node: Parse Submission Email
# ---------------------------------------------------------------------------

SUBMISSION_EXTRACTION_PROMPT = """You are an expert insurance broker analyst. Extract structured submission data from this broker's email.

Return ONLY valid JSON matching this schema (no markdown, no explanation):
{{
    "business_name": "string",
    "industry_description": "string",
    "years_in_business": "int or null",
    "annual_revenue": "float or null",
    "employee_count": "int or null",
    "state": "string (2-letter) or null",
    "address": "string or null",
    "building_age_years": "int or null",
    "building_stories": "int or null",
    "construction_type": "string or null",
    "has_sprinklers": "bool or null",
    "total_insurable_value": "float or null",
    "lines_requested": ["string"],
    "effective_date_requested": "string or null",
    "loss_ratio_3yr": "float or null",
    "claims_last_5yr": "int or null",
    "largest_claim_amount": "float or null"
}}

If information is not provided, use null.

SUBMISSION EMAIL:
{text}
"""


def parse_submission_node(state: AgentState) -> dict[str, Any]:
    """Extract structured data from the submission email."""
    llm = get_llm()
    attempts = state.get("parse_submission_attempts", 0) + 1

    prompt = SUBMISSION_EXTRACTION_PROMPT.format(text=state["submission_text"][:8000])
    response = llm.invoke(prompt)
    content = response.content

    try:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

        data = json.loads(text)
        data["raw_email_text"] = state["submission_text"][:2000]
        submission = SubmissionData(**data)
        return {
            "submission_data": submission.model_dump(),
            "parse_submission_attempts": attempts,
            "error": None,
        }
    except Exception as e:
        return {
            "submission_data": None,
            "parse_submission_attempts": attempts,
            "error": f"Submission parse error (attempt {attempts}): {str(e)[:200]}",
        }


# ---------------------------------------------------------------------------
# Node: Match Criteria (deterministic checks)
# ---------------------------------------------------------------------------

def match_criteria_node(state: AgentState) -> dict[str, Any]:
    """Run deterministic matching of submission against appetite criteria."""
    appetite = AppetiteCriteria(**state["appetite_criteria"])
    submission = SubmissionData(**state["submission_data"])
    results: list[dict] = []

    # 1. Industry match
    sub_industry = submission.industry_description.lower()
    industry_match = any(
        ind.lower() in sub_industry or sub_industry in ind.lower()
        for ind in appetite.target_industries
    )
    industry_excluded = any(
        exc.lower() in sub_industry or sub_industry in exc.lower()
        for exc in appetite.excluded_industries
    )
    if industry_excluded:
        industry_pass = False
        industry_explanation = (
            f"Industry '{submission.industry_description}' matches an excluded class. "
            f"Excluded: {', '.join(appetite.excluded_industries)}"
        )
        industry_severity = "hard_fail"
    elif industry_match:
        industry_pass = True
        industry_explanation = (
            f"Industry '{submission.industry_description}' matches carrier's target classes."
        )
        industry_severity = "info"
    else:
        industry_pass = False
        industry_explanation = (
            f"Industry '{submission.industry_description}' does not clearly match "
            f"any target classes: {', '.join(appetite.target_industries[:5])}. "
            f"Manual review recommended."
        )
        industry_severity = "soft_fail"
    results.append(
        CriterionResult(
            criterion_name="Industry Match",
            passed=industry_pass,
            severity=industry_severity,
            explanation=industry_explanation,
            carrier_limit=f"Targets: {', '.join(appetite.target_industries[:5])}",
            submission_value=submission.industry_description,
        ).model_dump()
    )

    # 2. Geographic match
    if appetite.target_states and submission.state:
        geo_pass = submission.state in appetite.target_states
        results.append(
            CriterionResult(
                criterion_name="Geographic Appetite",
                passed=geo_pass,
                severity="hard_fail" if not geo_pass else "info",
                explanation=(
                    f"State {submission.state} is {'within' if geo_pass else 'outside'} "
                    f"carrier's target states."
                ),
                carrier_limit=f"Target states: {', '.join(appetite.target_states[:10])}",
                submission_value=submission.state,
            ).model_dump()
        )
    if appetite.excluded_states and submission.state:
        if submission.state in appetite.excluded_states:
            results.append(
                CriterionResult(
                    criterion_name="Excluded State",
                    passed=False,
                    severity="hard_fail",
                    explanation=f"State {submission.state} is explicitly excluded.",
                    carrier_limit=f"Excluded: {', '.join(appetite.excluded_states)}",
                    submission_value=submission.state,
                ).model_dump()
            )

    # 3. Building age
    if appetite.building.max_age_years and submission.building_age_years:
        age_pass = submission.building_age_years <= appetite.building.max_age_years
        overage = submission.building_age_years - appetite.building.max_age_years
        rec = None
        if not age_pass:
            if overage <= 10:
                rec = f"Apply {overage * 3}% premium surcharge for building age overage"
            else:
                rec = "Recommend rejection — building age significantly exceeds limit"
        results.append(
            CriterionResult(
                criterion_name="Building Age",
                passed=age_pass,
                severity="hard_fail" if not age_pass and overage > 10 else "soft_fail",
                explanation=(
                    f"Building age is {submission.building_age_years} years. "
                    f"Carrier limit is {appetite.building.max_age_years} years. "
                    f"{'Within limit.' if age_pass else f'Exceeds limit by {overage} years.'}"
                ),
                carrier_limit=f"{appetite.building.max_age_years} years",
                submission_value=f"{submission.building_age_years} years",
                recommendation=rec,
            ).model_dump()
        )

    # 4. Building stories
    if appetite.building.max_stories and submission.building_stories:
        stories_pass = submission.building_stories <= appetite.building.max_stories
        results.append(
            CriterionResult(
                criterion_name="Building Height",
                passed=stories_pass,
                severity="hard_fail" if not stories_pass else "info",
                explanation=(
                    f"Building has {submission.building_stories} stories. "
                    f"Carrier max is {appetite.building.max_stories}. "
                    f"{'OK.' if stories_pass else 'Exceeds maximum.'}"
                ),
                carrier_limit=f"{appetite.building.max_stories} stories",
                submission_value=f"{submission.building_stories} stories",
            ).model_dump()
        )

    # 5. Sprinkler requirement
    if appetite.building.sprinkler_required and submission.has_sprinklers is not None:
        sprinkler_pass = submission.has_sprinklers
        results.append(
            CriterionResult(
                criterion_name="Sprinkler System",
                passed=sprinkler_pass,
                severity="soft_fail" if not sprinkler_pass else "info",
                explanation=(
                    f"Sprinklers {'present' if submission.has_sprinklers else 'absent'}. "
                    f"Carrier {'requires' if appetite.building.sprinkler_required else 'does not require'} sprinklers."
                ),
                carrier_limit="Required" if appetite.building.sprinkler_required else "Not required",
                submission_value="Yes" if submission.has_sprinklers else "No",
                recommendation="Apply 10% surcharge for no sprinkler system" if not sprinkler_pass else None,
            ).model_dump()
        )

    # 6. Financial: loss ratio
    if appetite.financial.max_loss_ratio_3yr and submission.loss_ratio_3yr is not None:
        lr_pass = submission.loss_ratio_3yr <= appetite.financial.max_loss_ratio_3yr
        results.append(
            CriterionResult(
                criterion_name="3-Year Loss Ratio",
                passed=lr_pass,
                severity="hard_fail" if not lr_pass else "info",
                explanation=(
                    f"Submission loss ratio: {submission.loss_ratio_3yr:.0%}. "
                    f"Carrier max: {appetite.financial.max_loss_ratio_3yr:.0%}."
                ),
                carrier_limit=f"{appetite.financial.max_loss_ratio_3yr:.0%}",
                submission_value=f"{submission.loss_ratio_3yr:.0%}",
            ).model_dump()
        )

    # 7. TIV limits
    if appetite.coverage.max_tiv and submission.total_insurable_value:
        tiv_pass = submission.total_insurable_value <= appetite.coverage.max_tiv
        results.append(
            CriterionResult(
                criterion_name="Total Insurable Value",
                passed=tiv_pass,
                severity="hard_fail" if not tiv_pass else "info",
                explanation=(
                    f"TIV: ${submission.total_insurable_value:,.0f}. "
                    f"Carrier max: ${appetite.coverage.max_tiv:,.0f}."
                ),
                carrier_limit=f"${appetite.coverage.max_tiv:,.0f}",
                submission_value=f"${submission.total_insurable_value:,.0f}",
            ).model_dump()
        )

    # 8. Years in business
    if appetite.financial.min_years_in_business and submission.years_in_business is not None:
        yib_pass = submission.years_in_business >= appetite.financial.min_years_in_business
        results.append(
            CriterionResult(
                criterion_name="Years in Business",
                passed=yib_pass,
                severity="soft_fail" if not yib_pass else "info",
                explanation=(
                    f"Business has operated {submission.years_in_business} years. "
                    f"Carrier minimum: {appetite.financial.min_years_in_business} years."
                ),
                carrier_limit=f"{appetite.financial.min_years_in_business} years",
                submission_value=f"{submission.years_in_business} years",
            ).model_dump()
        )

    # 9. Lines of coverage
    if appetite.coverage.lines_offered and submission.lines_requested:
        offered_set = {l.upper() for l in appetite.coverage.lines_offered}
        requested_set = {l.upper() for l in submission.lines_requested}
        unavailable = requested_set - offered_set
        if unavailable:
            results.append(
                CriterionResult(
                    criterion_name="Lines of Coverage",
                    passed=False,
                    severity="soft_fail",
                    explanation=(
                        f"Requested lines not offered: {', '.join(unavailable)}. "
                        f"Carrier offers: {', '.join(appetite.coverage.lines_offered)}."
                    ),
                    carrier_limit=", ".join(appetite.coverage.lines_offered),
                    submission_value=", ".join(submission.lines_requested),
                    recommendation="Remove unavailable lines or find additional markets.",
                ).model_dump()
            )
        else:
            results.append(
                CriterionResult(
                    criterion_name="Lines of Coverage",
                    passed=True,
                    severity="info",
                    explanation="All requested lines are offered by the carrier.",
                    carrier_limit=", ".join(appetite.coverage.lines_offered),
                    submission_value=", ".join(submission.lines_requested),
                ).model_dump()
            )

    return {"match_result": {"criteria_results": results, "error": None}}


# ---------------------------------------------------------------------------
# Node: Reasoning Gate (LLM-powered final judgment)
# ---------------------------------------------------------------------------

REASONING_PROMPT = """You are a senior insurance underwriter reviewing a submission match analysis.

CARRIER APPETITE (structured):
{appetite_json}

SUBMISSION DATA (structured):
{submission_json}

CRITERIA CHECK RESULTS:
{criteria_json}

Based on the above, provide your final underwriting judgment. Return ONLY valid JSON:
{{
    "verdict": "accept" | "reject" | "refer",
    "confidence": 0.0 to 1.0,
    "overall_summary": "2-3 sentence executive summary for the underwriter",
    "recommended_actions": ["action 1", "action 2"],
    "premium_adjustment_pct": float or null,
    "risk_score": 0.0 to 100.0
}}

Rules:
- If ANY criterion has severity "hard_fail", the verdict should be "reject" or "refer"
- If only "soft_fail" items exist, verdict should be "refer" with premium adjustments
- Explain your reasoning naturally, as if briefing a senior underwriter
- Be specific: cite exact numbers, limits, and overages
- If recommending premium adjustment, explain the math
"""


def reasoning_gate_node(state: AgentState) -> dict[str, Any]:
    """LLM-powered reasoning gate that synthesizes all checks into a final verdict."""
    llm = get_llm()

    criteria_results = state["match_result"]["criteria_results"]

    prompt = REASONING_PROMPT.format(
        appetite_json=json.dumps(state["appetite_criteria"], indent=2),
        submission_json=json.dumps(state["submission_data"], indent=2, default=str),
        criteria_json=json.dumps(criteria_results, indent=2),
    )

    response = llm.invoke(prompt)
    content = response.content

    try:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

        reasoning = json.loads(text)

        # Merge criteria results into the final match result
        match_result = MatchResult(
            verdict=MatchVerdict(reasoning["verdict"]),
            confidence=reasoning["confidence"],
            overall_summary=reasoning["overall_summary"],
            criteria_results=[CriterionResult(**cr) for cr in criteria_results],
            recommended_actions=reasoning.get("recommended_actions", []),
            premium_adjustment_pct=reasoning.get("premium_adjustment_pct"),
            risk_score=reasoning.get("risk_score", 50.0),
        )

        return {"match_result": match_result.model_dump(), "error": None}
    except Exception as e:
        # Fallback: build result from criteria alone
        hard_fails = [
            cr for cr in criteria_results if cr.get("severity") == "hard_fail" and not cr.get("passed")
        ]
        soft_fails = [
            cr for cr in criteria_results if cr.get("severity") == "soft_fail" and not cr.get("passed")
        ]

        if hard_fails:
            verdict = MatchVerdict.REJECT
        elif soft_fails:
            verdict = MatchVerdict.REFER
        else:
            verdict = MatchVerdict.ACCEPT

        match_result = MatchResult(
            verdict=verdict,
            confidence=0.6,
            overall_summary=f"Automated analysis (LLM reasoning failed: {str(e)[:100]}). "
            f"Found {len(hard_fails)} hard failures and {len(soft_fails)} soft failures.",
            criteria_results=[CriterionResult(**cr) for cr in criteria_results],
            recommended_actions=["Manual review recommended — LLM reasoning gate encountered an error."],
            risk_score=min(100, len(hard_fails) * 30 + len(soft_fails) * 15),
        )
        return {"match_result": match_result.model_dump(), "error": str(e)}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def should_retry_appetite(state: AgentState) -> str:
    """Route: retry appetite parse or proceed."""
    if state.get("appetite_criteria") is not None:
        return "parse_submission"
    if state.get("parse_appetite_attempts", 0) < 2:
        return "parse_appetite"  # retry
    return "error_end"


def should_retry_submission(state: AgentState) -> str:
    """Route: retry submission parse or proceed."""
    if state.get("submission_data") is not None:
        return "match_criteria"
    if state.get("parse_submission_attempts", 0) < 2:
        return "parse_submission"  # retry
    return "error_end"


def error_end_node(state: AgentState) -> dict[str, Any]:
    """Terminal node for unrecoverable errors."""
    return {
        "match_result": MatchResult(
            verdict=MatchVerdict.REFER,
            confidence=0.0,
            overall_summary=f"Unable to complete analysis: {state.get('error', 'Unknown error')}",
            criteria_results=[],
            recommended_actions=["Manual review required — automated analysis failed."],
            risk_score=50.0,
        ).model_dump()
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("parse_appetite", parse_appetite_node)
    graph.add_node("parse_submission", parse_submission_node)
    graph.add_node("match_criteria", match_criteria_node)
    graph.add_node("reasoning_gate", reasoning_gate_node)
    graph.add_node("error_end", error_end_node)

    # Entry point
    graph.set_entry_point("parse_appetite")

    # Conditional edges with retry logic
    graph.add_conditional_edges(
        "parse_appetite",
        should_retry_appetite,
        {
            "parse_submission": "parse_submission",
            "parse_appetite": "parse_appetite",
            "error_end": "error_end",
        },
    )

    graph.add_conditional_edges(
        "parse_submission",
        should_retry_submission,
        {
            "match_criteria": "match_criteria",
            "parse_submission": "parse_submission",
            "error_end": "error_end",
        },
    )

    # Linear edges
    graph.add_edge("match_criteria", "reasoning_gate")
    graph.add_edge("reasoning_gate", END)
    graph.add_edge("error_end", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_appetite_match(
    appetite_text: str,
    submission_text: str,
) -> tuple[MatchResult, dict]:
    """
    Run the full appetite matching pipeline.

    Args:
        appetite_text: Raw text from the carrier's appetite guide PDF.
        submission_text: Raw text from the broker's submission email.

    Returns:
        (MatchResult, full_state_dict) for inspection / debugging.
    """
    app = build_graph()

    initial_state: AgentState = {
        "appetite_text": appetite_text,
        "submission_text": submission_text,
        "appetite_criteria": None,
        "submission_data": None,
        "match_result": None,
        "parse_appetite_attempts": 0,
        "parse_submission_attempts": 0,
        "error": None,
    }

    final_state = app.invoke(initial_state)
    match_result = MatchResult(**final_state["match_result"])
    return match_result, final_state
