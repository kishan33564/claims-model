"""Same payout + decision rules as the website (src/lib/insurance.ts).

payout = clamp(loss * coverage - deductible, 0, min(cap, insured_value))
decision: no damage / wrong subject -> rejected; fraud not low or confidence < 0.6 -> review; else approved
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent


@dataclass
class Company:
    id: str
    name: str
    coverage: float
    deductible_factor: float
    cap_factor: float
    settlement_days: int
    covers: set[str]


def load_claim_types() -> dict[str, dict]:
    with open(HERE / "claim_types.csv", newline="") as f:
        return {r["claim_type"]: r for r in csv.DictReader(f)}


def load_companies() -> dict[str, Company]:
    all_types = set(load_claim_types())
    out = {}
    with open(HERE / "payout_rules.csv", newline="") as f:
        for r in csv.DictReader(f):
            covers = all_types if r["covers"] == "all" else set(r["covers"].split("|"))
            out[r["company_id"]] = Company(
                r["company_id"], r["company_name"], float(r["coverage"]), float(r["deductible_factor"]),
                float(r["cap_factor"]), int(r["settlement_days"]), covers,
            )
    return out


CLAIM_TYPES = load_claim_types()
COMPANIES = load_companies()


def compute_payout(company_id: str, claim_type: str, loss: float, insured_value: float = 0) -> dict:
    c, t = COMPANIES.get(company_id), CLAIM_TYPES.get(claim_type)
    if not c or not t or claim_type not in c.covers:
        return {"covered": False, "payout": 0}
    deductible = round(float(t["base_deductible"]) * c.deductible_factor)
    cap = round(float(t["base_cap"]) * c.cap_factor)
    if insured_value > 0 and t["category"] in ("vehicle", "property"):
        cap = min(cap, insured_value)
    payout = round(min(max(loss * c.coverage - deductible, 0), cap))
    return {"covered": True, "payout": payout, "coverage": c.coverage, "deductible": deductible, "cap": cap}


def decide(matches_claim_type: bool, severity: str, fraud_risk: str, confidence: float) -> str:
    if not matches_claim_type or severity == "none":
        return "rejected"
    if fraud_risk != "low" or confidence < 0.6:
        return "review"
    return "approved"


def compare_companies(claim_type: str, loss: float, insured_value: float = 0) -> list[dict]:
    rows = [{"company": c.name, **compute_payout(c.id, claim_type, loss, insured_value)} for c in COMPANIES.values()]
    return sorted(rows, key=lambda r: r["payout"], reverse=True)


if __name__ == "__main__":
    assert compute_payout("icici", "car_accident", 100_000)["payout"] == 88_000
    assert compute_payout("star", "car_accident", 100_000)["covered"] is False
    assert compute_payout("bajaj", "bike_accident", 100_000, 40_000)["payout"] == 40_000
    assert decide(True, "moderate", "low", 0.8) == "approved"
    assert decide(True, "none", "low", 0.9) == "rejected"
    print("payout rules OK")
    for row in compare_companies("car_accident", 85_000, 600_000):
        print(row)
