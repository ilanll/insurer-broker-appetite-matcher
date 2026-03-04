"""Pydantic models for broker submission data."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MatchVerdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REFER = "refer"  # needs human review


class SubmissionData(BaseModel):
    """
    Structured data extracted from a broker's submission email.
    """

    # Business info
    business_name: str = Field(..., description="Name of the insured business")
    industry_description: str = Field(
        ..., description="Description of what the business does"
    )
    years_in_business: Optional[int] = Field(
        None, description="How long the business has been operating"
    )
    annual_revenue: Optional[float] = Field(
        None, description="Annual revenue in USD"
    )
    employee_count: Optional[int] = Field(
        None, description="Number of employees"
    )

    # Location
    state: Optional[str] = Field(
        None, description="Primary state of operations (2-letter code)"
    )
    address: Optional[str] = Field(None, description="Business address")

    # Building / property info
    building_age_years: Optional[int] = Field(
        None, description="Age of the primary building in years"
    )
    building_stories: Optional[int] = Field(
        None, description="Number of stories"
    )
    construction_type: Optional[str] = Field(
        None, description="Construction type (e.g., masonry, frame, fire-resistive)"
    )
    has_sprinklers: Optional[bool] = Field(
        None, description="Whether the building has sprinkler systems"
    )
    total_insurable_value: Optional[float] = Field(
        None, description="Total Insurable Value (TIV) in USD"
    )

    # Coverage requested
    lines_requested: list[str] = Field(
        default_factory=list,
        description="Lines of coverage requested (GL, Property, WC, etc.)",
    )
    effective_date_requested: Optional[str] = Field(
        None, description="Desired policy effective date"
    )

    # Loss history
    loss_ratio_3yr: Optional[float] = Field(
        None, description="3-year loss ratio if provided"
    )
    claims_last_5yr: Optional[int] = Field(
        None, description="Number of claims in the last 5 years"
    )
    largest_claim_amount: Optional[float] = Field(
        None, description="Largest single claim amount"
    )

    # Raw text
    raw_email_text: Optional[str] = Field(
        None, description="Original email text for reference"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "business_name": "Summit Roofing LLC",
                "industry_description": "Commercial roofing contractor specializing in flat-roof systems for high-rise buildings",
                "building_age_years": 45,
                "state": "TX",
            }
        }


class CriterionResult(BaseModel):
    """Result of checking a single appetite criterion."""

    criterion_name: str = Field(..., description="Name of the criterion checked")
    passed: bool = Field(..., description="Whether the submission meets this criterion")
    severity: str = Field(
        "info",
        description="Severity if failed: 'hard_fail', 'soft_fail', 'info'",
    )
    explanation: str = Field(
        ..., description="Human-readable explanation of the check result"
    )
    carrier_limit: Optional[str] = Field(
        None, description="What the carrier's limit is"
    )
    submission_value: Optional[str] = Field(
        None, description="What the submission's value is"
    )
    recommendation: Optional[str] = Field(
        None,
        description="Recommended action (e.g., 'Apply 15% premium load')",
    )


class MatchResult(BaseModel):
    """Full result of matching a submission against appetite."""

    verdict: MatchVerdict = Field(..., description="Overall verdict")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the verdict (0-1)"
    )
    overall_summary: str = Field(
        ...,
        description="Executive summary of the match result for the underwriter",
    )
    criteria_results: list[CriterionResult] = Field(
        default_factory=list,
        description="Detailed results for each criterion checked",
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="List of recommended actions for the underwriter",
    )
    premium_adjustment_pct: Optional[float] = Field(
        None,
        description="Suggested premium adjustment percentage (positive = surcharge)",
    )
    risk_score: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Overall risk score (0 = perfect fit, 100 = terrible fit)",
    )
