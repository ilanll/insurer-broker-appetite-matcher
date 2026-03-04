"""Pydantic models for carrier Appetite Guide criteria."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskTolerance(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BuildingCriteria(BaseModel):
    """Hard limits on building characteristics."""

    max_age_years: Optional[int] = Field(
        None, description="Maximum building age in years"
    )
    max_stories: Optional[int] = Field(None, description="Maximum number of stories")
    accepted_construction_types: list[str] = Field(
        default_factory=list,
        description="Accepted construction types (e.g., 'fire-resistive', 'masonry')",
    )
    min_fire_protection_grade: Optional[int] = Field(
        None, description="Minimum fire protection class (1-10, lower is better)"
    )
    sprinkler_required: Optional[bool] = Field(
        None, description="Whether sprinkler systems are required"
    )


class FinancialCriteria(BaseModel):
    """Financial thresholds for acceptable submissions."""

    min_annual_revenue: Optional[float] = Field(
        None, description="Minimum annual revenue in USD"
    )
    max_annual_revenue: Optional[float] = Field(
        None, description="Maximum annual revenue in USD"
    )
    min_years_in_business: Optional[int] = Field(
        None, description="Minimum years the business must have been operating"
    )
    max_loss_ratio_3yr: Optional[float] = Field(
        None,
        description="Maximum acceptable 3-year loss ratio (e.g., 0.65 = 65%)",
    )


class CoverageLimits(BaseModel):
    """Limits on coverage the carrier will write."""

    max_tiv: Optional[float] = Field(
        None, description="Maximum Total Insurable Value in USD"
    )
    max_single_location_tiv: Optional[float] = Field(
        None, description="Max TIV for any single location"
    )
    lines_offered: list[str] = Field(
        default_factory=list,
        description="Lines of coverage offered (GL, Property, WC, Auto, Umbrella, etc.)",
    )


class AppetiteCriteria(BaseModel):
    """
    Structured representation of a carrier's appetite guide.
    Extracted from PDF via LLM.
    """

    carrier_name: str = Field(..., description="Name of the insurance carrier")
    effective_date: Optional[str] = Field(
        None, description="Date the appetite guide is effective"
    )

    # Target classes
    target_industries: list[str] = Field(
        default_factory=list,
        description="Industry descriptions the carrier targets (human-readable)",
    )
    excluded_industries: list[str] = Field(
        default_factory=list,
        description="Industries the carrier explicitly excludes",
    )

    # Geographic appetite
    target_states: list[str] = Field(
        default_factory=list,
        description="US states the carrier writes in (empty = all states)",
    )
    excluded_states: list[str] = Field(
        default_factory=list,
        description="US states the carrier avoids",
    )

    # Risk criteria
    risk_tolerance: RiskTolerance = Field(
        RiskTolerance.MEDIUM, description="Overall risk tolerance level"
    )
    building: BuildingCriteria = Field(default_factory=BuildingCriteria)
    financial: FinancialCriteria = Field(default_factory=FinancialCriteria)
    coverage: CoverageLimits = Field(default_factory=CoverageLimits)

    # Soft preferences
    preferred_premium_range_min: Optional[float] = Field(
        None, description="Minimum premium the carrier prefers"
    )
    preferred_premium_range_max: Optional[float] = Field(
        None, description="Maximum premium the carrier prefers"
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Additional underwriting notes or preferences from the guide",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "carrier_name": "Keystone Mutual",
                "target_industries": [
                    "Roofing Contractors",
                    "Commercial Building Construction",
                ],
                "excluded_industries": ["Asbestos Abatement", "Demolition"],
                "building": {
                    "max_age_years": 40,
                    "max_stories": 6,
                    "sprinkler_required": True,
                },
            }
        }
