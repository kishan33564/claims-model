"""Claim-photo API using your trained models.

  uvicorn serve:app --host 0.0.0.0 --port 8000

POST /predict  (multipart form)
  image          file   claim photo
  claim_type     str    car_accident | bike_accident | bus_accident | truck_accident | auto_rickshaw | ...
  company_id     str    icici | hdfc | bajaj | tata | newindia | star
  claimed_amount float  rupees (optional)
  insured_value  float  vehicle IDV / property value (optional)

Returns the same fields the website shows: decision, severity, fraud_risk, estimated_loss_inr, payout, comparison.
"""
from __future__ import annotations

import io
from pathlib import Path

import timm
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from payout_rules import CLAIM_TYPES, compare_companies, compute_payout, decide

MODELS = Path("models")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
device = "cuda" if torch.cuda.is_available() else "cpu"

# COCO classes the pretrained YOLO detector already knows -> claim types
VEHICLE_FOR_CLAIM = {
    "car_accident": {"car"},
    "bike_accident": {"motorcycle", "bicycle"},
    "bus_accident": {"bus"},
    "truck_accident": {"truck"},
    "auto_rickshaw": {"car", "motorcycle", "truck"},  # no auto-rickshaw class in COCO; fine-tune YOLO for better results
}
ACCIDENT_CLAIM_TYPES = set(VEHICLE_FOR_CLAIM)
# Share of insured value lost, by severity (tune with your own settled-claims history)
LOSS_SHARE = {"minor": 0.04, "moderate": 0.12, "severe": 0.35}
DEFAULT_VALUE = {"car_accident": 600_000, "bike_accident": 80_000, "bus_accident": 2_500_000, "truck_accident": 2_000_000, "auto_rickshaw": 200_000}


def load_classifier(name: str):
    ckpt = torch.load(MODELS / f"{name}.pt", map_location=device)
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    tf = transforms.Compose([transforms.Resize(int(ckpt["img"] * 1.14)), transforms.CenterCrop(ckpt["img"]), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return model, ckpt["classes"], tf


damage_model, damage_classes, damage_tf = load_classifier("damage")
severity_model, severity_classes, severity_tf = load_classifier("severity")
detector = YOLO("yolov8n.pt")  # downloads automatically the first time

app = FastAPI(title="ZeroTouch Claims model")


@torch.no_grad()
def classify(model, classes, tf, img: Image.Image) -> tuple[str, float]:
    probs = model(tf(img).unsqueeze(0).to(device)).softmax(1)[0]
    i = int(probs.argmax())
    return classes[i], float(probs[i])


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    claim_type: str = Form(...),
    company_id: str = Form(...),
    claimed_amount: float = Form(0),
    insured_value: float = Form(0),
):
    if claim_type not in CLAIM_TYPES or claim_type not in ACCIDENT_CLAIM_TYPES:
        raise HTTPException(400, "claim_type must be a supported vehicle accident")
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "not a readable image")

    # 1. does the photo show the claimed kind of vehicle?
    detections = detector(img, verbose=False)[0]
    names = {detector.names[int(c)] for c in detections.boxes.cls.tolist()}
    wanted = VEHICLE_FOR_CLAIM[claim_type]
    matches = bool(names & wanted)

    # 2. is it damaged, and how badly?
    dmg_label, dmg_conf = classify(damage_model, damage_classes, damage_tf, img)
    if dmg_label == "not_damaged":
        severity, sev_conf = "none", dmg_conf
    else:
        severity, sev_conf = classify(severity_model, severity_classes, severity_tf, img)
    confidence = round(min(dmg_conf, sev_conf), 3)

    has_damage = dmg_label != "not_damaged"
    is_accident_photo = matches and has_damage
    if not is_accident_photo:
        if not names:
            message = "No supported vehicle was detected. Upload a clear vehicle accident photo."
        elif not matches:
            message = "The detected vehicle does not match the selected accident type."
        else:
            message = "No vehicle damage was detected. Upload a photo showing accident damage."
        return {
            "is_accident_photo": False,
            "matches_claim_type": matches,
            "has_damage": has_damage,
            "detected_objects": sorted(names),
            "severity": severity,
            "confidence": confidence,
            "message": message,
            "decision": "rejected",
            "fraud_risk": "not_assessed",
            "estimated_loss_inr": 0,
            "assessed_loss_inr": 0,
            "payout": 0,
            "comparison": [],
        }

    # 3. loss estimate + fraud heuristics (swap in models/fraud.joblib when you have claim metadata)
    value = insured_value or DEFAULT_VALUE.get(claim_type, 500_000)
    estimate = round(value * LOSS_SHARE.get(severity, 0))
    signals = []
    if claimed_amount and estimate and claimed_amount > 2.5 * estimate:
        signals.append("Claimed amount is far above what the visible damage supports")
    if not matches:
        signals.append(f"Photo shows {sorted(names) or 'no vehicle'}, not a {claim_type.replace('_', ' ')}")
    fraud_risk = "high" if len(signals) >= 2 else "medium" if signals else "low"

    decision = decide(matches, severity, fraud_risk, confidence)
    loss = min(claimed_amount, estimate) if claimed_amount and estimate else max(claimed_amount, estimate)
    payout = compute_payout(company_id, claim_type, loss, insured_value)
    return {
        "is_accident_photo": True,
        "matches_claim_type": matches,
        "has_damage": has_damage,
        "detected_objects": sorted(names),
        "severity": severity,
        "confidence": confidence,
        "fraud_risk": fraud_risk,
        "fraud_signals": signals,
        "estimated_loss_inr": estimate,
        "assessed_loss_inr": loss,
        "decision": decision,
        "payout": 0 if decision == "rejected" else payout.get("payout", 0),
        "payout_terms": payout,
        "comparison": compare_companies(claim_type, loss, insured_value),
    }
