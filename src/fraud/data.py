"""Synthetic UK motor insurance claims with a realistic, learnable fraud signal.

The data is generated, not real. The data card in the README states this clearly.
Fraud is rare (about 6 per cent) and driven by a mix of weak signals plus noise,
so the model has to work for its score rather than finding a trivial rule.
"""
import numpy as np
import pandas as pd

REGIONS = ["London", "South East", "Midlands", "North West", "Yorkshire", "Scotland", "Wales"]
CLAIM_TYPES = ["collision", "theft", "windscreen", "vandalism", "fire"]

NUMERIC = [
    "claim_amount", "vehicle_age", "driver_age", "policy_tenure_months",
    "previous_claims", "days_to_report", "injury_claimed", "witness_present",
    "police_report", "late_night", "third_party_involved", "flagged_garage",
]
CATEGORICAL = ["region", "claim_type"]
TARGET = "is_fraud"


def generate_claims(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "claim_amount": np.round(rng.lognormal(mean=7.9, sigma=0.8, size=n), 2),
        "vehicle_age": rng.integers(0, 18, n),
        "driver_age": rng.integers(18, 80, n),
        "policy_tenure_months": rng.integers(1, 120, n),
        "previous_claims": rng.poisson(0.6, n),
        "days_to_report": rng.integers(0, 45, n),
        "injury_claimed": rng.binomial(1, 0.12, n),
        "witness_present": rng.binomial(1, 0.35, n),
        "police_report": rng.binomial(1, 0.5, n),
        "late_night": rng.binomial(1, 0.15, n),
        "third_party_involved": rng.binomial(1, 0.45, n),
        "flagged_garage": rng.binomial(1, 0.07, n),
        "region": rng.choice(REGIONS, n, p=[0.22, 0.18, 0.15, 0.15, 0.12, 0.10, 0.08]),
        "claim_type": rng.choice(CLAIM_TYPES, n, p=[0.50, 0.15, 0.17, 0.10, 0.08]),
    })

    logit = (
        -4.3
        + 0.00018 * df["claim_amount"]
        + 0.35 * df["previous_claims"]
        + 0.03 * df["days_to_report"]
        - 0.012 * df["policy_tenure_months"]
        + 0.9 * df["injury_claimed"]
        - 0.7 * df["witness_present"]
        - 0.6 * df["police_report"]
        + 0.6 * df["late_night"]
        + 1.6 * df["flagged_garage"]
        + 0.5 * (df["claim_type"] == "theft")
        + 0.4 * (df["claim_type"] == "fire")
        # Interaction a linear model cannot capture: new policy, large claim, slow reporting
        + 1.8 * ((df["policy_tenure_months"] < 12) & (df["claim_amount"] > 6000)
                 & (df["days_to_report"] > 14))
        + rng.normal(0, 0.6, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df[TARGET] = rng.binomial(1, prob)
    return df


if __name__ == "__main__":
    data = generate_claims()
    data.to_csv("data/claims.csv", index=False)
    print(f"Saved {len(data)} claims, fraud rate {data[TARGET].mean():.1%}")
