"""Request and response models. Validation happens here, before any model code runs."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.fraud.data import CLAIM_TYPES, REGIONS

Region = Literal[tuple(REGIONS)]
ClaimType = Literal[tuple(CLAIM_TYPES)]


class ClaimIn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "claim_amount": 9500, "vehicle_age": 4, "driver_age": 27,
            "policy_tenure_months": 5, "previous_claims": 2, "days_to_report": 21,
            "injury_claimed": True, "witness_present": False, "police_report": False,
            "late_night": True, "third_party_involved": True, "flagged_garage": True,
            "region": "London", "claim_type": "collision",
        }},
    )

    claim_amount: float = Field(gt=0, le=1_000_000, description="Claimed amount in GBP")
    vehicle_age: int = Field(ge=0, le=40)
    driver_age: int = Field(ge=17, le=100)
    policy_tenure_months: int = Field(ge=0, le=600)
    previous_claims: int = Field(ge=0, le=50)
    days_to_report: int = Field(ge=0, le=365)
    injury_claimed: bool
    witness_present: bool
    police_report: bool
    late_night: bool
    third_party_involved: bool
    flagged_garage: bool
    region: Region
    claim_type: ClaimType

    def to_features(self) -> dict:
        """Plain dict in the form the model was trained on (booleans as 0 or 1)."""
        data = self.model_dump()
        return {k: int(v) if isinstance(v, bool) else v for k, v in data.items()}


class Reason(BaseModel):
    feature: str
    impact: float
    direction: Literal["raises risk", "lowers risk"]


class ScoreOut(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    model: str


class ExplainOut(ScoreOut):
    reasons: list[Reason]
