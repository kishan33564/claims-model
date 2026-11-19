<<<<<<< HEAD
=======
# claims-model
>>>>>>> e561cf560ab199f2687636a8f5e5bb6577185a7e
# ZeroTouch Claims — model training kit

Trains your own models that decide, from a claim photo, whether a person can get insurance and how much
each company pays. It mirrors what the website's officer desk does.

```
photo ──► 1. vehicle check (YOLOv8, pretrained)   car / motorcycle / bus / truck? matches claim type?
      ──► 2. damage model (EfficientNet, trained)  damaged vs not damaged
      ──► 3. severity model (EfficientNet, trained) minor / moderate / severe
      ──► 4. fraud model (gradient boosting, trained on claim records)  low / medium / high
      ──► 5. payout_rules.py   loss × coverage − deductible, capped at limit and insured value
      ──► decision: approved / review / rejected  +  payout per company
```

## 1. Datasets to train on

| Purpose | Dataset | Size | Get it |
|---|---|---|---|
| Damaged vs not damaged, and severity (minor / moderate / severe) | **Car Damage Detection** (Anuj Shah) — folders `data1a` (damage / whole) and `data3a` (minor / moderate / severe) | ~2,300 + ~1,400 images | `kaggle datasets download -d anujms/car-damage-detection` |
| Damage *type* (dent, scratch, crack, glass shatter, broken lamp, flat tyre) — optional upgrade | **CarDD** (USTC) | 4,000 images, 9,000+ labelled damages | Apply at https://cardd-ustc.github.io |
| Fraud risk from claim details | **Vehicle Insurance Claim Fraud Detection** (`fraud_oracle.csv`) | 15,420 claims, ~6% fraud | `kaggle datasets download -d shivamb/vehicle-claim-fraud-detection` |
| Vehicle type (car, bike, bus, truck) | Already learned by the pretrained **YOLOv8 COCO** detector — no training needed | 80 classes | downloads automatically |

The public damage datasets are mostly cars. For bikes, buses, trucks and auto-rickshaws, add your own
labelled insurer photos (even 200–300 per class helps a lot) — see step 3.

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Kaggle token: kaggle.com → Settings → Create New Token → save as ~/.kaggle/kaggle.json
```

A GPU (NVIDIA or Apple Silicon) is recommended; training also runs on CPU, just slower.
Free option: run these scripts in a Google Colab / Kaggle notebook with a GPU.

## 3. Prepare the images

```bash
python prepare_dataset.py
# optional: add your own photos, one sub-folder per class
#   my_photos/damaged/  my_photos/not_damaged/  my_photos/minor/  my_photos/moderate/  my_photos/severe/
python prepare_dataset.py --extra my_photos
```

Creates `data/damage/{train,val,test}/…` and `data/severity/{train,val,test}/…` (75 / 15 / 10 split).

## 4. Train

```bash
python train_image_model.py --task damage   --epochs 15
python train_image_model.py --task severity --epochs 25

kaggle datasets download -d shivamb/vehicle-claim-fraud-detection -p raw/fraud --unzip
python train_fraud_model.py --csv raw/fraud/fraud_oracle.csv
```

Each run prints test accuracy, a per-class report and a confusion matrix, and saves to `models/`.
Typical results: damage ≈ 90%+, severity ≈ 70–80% (severity is subjective — your own labelled photos improve it most).

## 5. Run the prediction API

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/predict \
  -F image=@damaged_car.jpg -F claim_type=car_accident -F company_id=icici \
  -F claimed_amount=45000 -F insured_value=650000
```

Response (same fields the officer desk shows):

```json
{ "matches_claim_type": true, "severity": "moderate", "confidence": 0.86, "fraud_risk": "low",
  "estimated_loss_inr": 78000, "assessed_loss_inr": 45000, "decision": "approved", "payout": 38500,
  "comparison": [{"company": "ICICI Lombard", "payout": 38500}, …] }
```

## 6. Payout rules

`payout_rules.csv` (companies) and `claim_types.csv` (limits, deductibles) hold the plan figures — the same
ones the website uses. Edit them to match real policy wording. Formula:

```
payout = min( max(loss × coverage − deductible, 0), min(claim limit, insured value) )
decision: photo doesn't match / no damage → rejected
          fraud risk medium/high or confidence < 60% → needs review
          otherwise → approved
```

Run `python payout_rules.py` to self-check the rules.

## 7. Run the claims frontend

```bash
streamlit run app.py --server.port 8501
```

Set `CLAIMS_API_URL` to use a deployed prediction endpoint instead of the default
`http://localhost:8000/predict`. The frontend requires a live API and does not produce demo
assessments. It only offers vehicle accident claim types; the API rejects photos without a
matching detected vehicle and visible damage.

## 8. Connecting to the website

The website currently uses a built-in AI vision model for the photo check, so it works without hosting
anything. Once your trained API is deployed (Render, Railway, a GPU VM, etc.), the website's photo
check can be switched to call `POST /predict` instead — ask for that change and provide the API URL.

> The company plan figures are sample numbers for demonstration, not real insurer tariffs.
