from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

from payout_rules import CLAIM_TYPES, COMPANIES


st.set_page_config(
    page_title="ZeroTouch Claims",
    page_icon=":material/health_and_safety:",
    layout="wide",
)

st.html("""
<style>
.st-key-page_header {
    padding: 1.35rem 1.6rem;
    margin-bottom: 1.2rem;
    border: 1px solid #7c3aed;
    border-radius: 1.25rem;
    background: linear-gradient(115deg, #4f46e5 0%, #7c3aed 58%, #c026d3 100%);
}

.st-key-page_header h1 {
    margin-bottom: 0.25rem;
    color: #ffffff;
    letter-spacing: -0.035em;
}

.st-key-page_header p {
    color: #e0e7ff;
    font-size: 1rem;
}

.st-key-claim_panel {
    border-color: #ddd6fe;
    border-radius: 1.1rem;
    background: linear-gradient(145deg, #ffffff 0%, #f0fdf4 58%, #effcf8 100%);
    box-shadow: 0 8px 28px rgba(109, 40, 217, 0.08);
}

.st-key-claim_panel h2 {
    color: #6d28d9;
}

.st-key-photo_placeholder {
    border: 1px dashed #86bca3;
    border-radius: 1rem;
    background: linear-gradient(145deg, #ecfdf5, #ecfeff);
}

.st-key-assess_claim button {
    min-height: 3rem;
    border-radius: 0.8rem;
    font-weight: 650;
    transition: transform 120ms ease, box-shadow 120ms ease;
}

.st-key-assess_claim button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(124, 58, 237, 0.3);
}

.st-key-metrics_panel {
    gap: 0.9rem;
}

@media (max-width: 700px) {
    .st-key-page_header {
        padding: 1rem 1.1rem;
    }

    .st-key-page_header h1 {
        font-size: 1.8rem;
    }
}
</style>
""")

if "recent_claims" not in st.session_state:
    st.session_state.recent_claims = []


@st.cache_data(ttl="30s", max_entries=20)
def check_api(api_url: str) -> bool:
    try:
        response = requests.get(api_url.rsplit("/predict", 1)[0] + "/docs", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def format_inr(value: int | float) -> str:
    value = int(round(value))
    return f"₹{value:,}"


with st.sidebar:
    st.markdown("### :material/health_and_safety: ZeroTouch")
    st.caption("Insurance claim")
    st.space("medium")

    api_url = st.text_input(
        "Prediction API",
        value=os.getenv("CLAIMS_API_URL", "http://localhost:8000/predict"),
    )
    api_live = check_api(api_url)
    if api_live:
        st.success("AI service connected", icon=":material/check_circle:")
    else:
        st.warning("AI service offline — assessment unavailable", icon=":material/warning:")

    st.space("medium")
    st.caption("Recent claims")
    if not st.session_state.recent_claims:
        st.caption("No claims submitted yet")
    else:
        for claim in st.session_state.recent_claims[:4]:
            st.markdown(
                f"**{claim['type']}**  \n"
                f"{claim['decision'].title()} · {format_inr(claim['payout'])}"
            )


with st.container(key="page_header"):
    st.title("Claims assessment")
    st.caption("Upload a claim photo and compare plan outcomes")

accident_claim_types = {
    "car_accident",
    "bike_accident",
    "bus_accident",
    "truck_accident",
    "auto_rickshaw",
}
claim_options = {
    claim_type: details["label"]
    for claim_type, details in CLAIM_TYPES.items()
    if claim_type in accident_claim_types
}
company_options = {
    company_id: company.name
    for company_id, company in COMPANIES.items()
}

with st.container(border=True, key="claim_panel"):
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Claim details")
        claim_type = st.selectbox(
            "Claim type",
            options=list(claim_options),
            format_func=lambda value: claim_options[value],
        )
        company_id = st.selectbox(
            "Primary insurer",
            options=list(company_options),
            format_func=lambda value: company_options[value],
        )
        st.session_state.selected_company = company_id

        claimed_amount = st.number_input(
            "Claimed amount",
            min_value=0,
            value=45000,
            step=5000,
            format="%d",
        )
        insured_value = st.number_input(
            "Insured value",
            min_value=0,
            value=650000,
            step=50000,
            format="%d",
        )

        photo = st.file_uploader(
            "Claim photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="Only photos showing a damaged vehicle can be assessed.",
        )
        if not api_live:
            st.info("Train the models and start the prediction API to assess photos.")

        submitted = st.button(
            "Assess claim",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
            disabled=photo is None or not api_live,
            key="assess_claim",
        )

    with right:
        st.subheader("Photo preview")
        if photo:
            st.image(photo, caption=photo.name, width="stretch")
        else:
            with st.container(
                border=True,
                horizontal_alignment="center",
                height=360,
                key="photo_placeholder",
            ):
                st.markdown(":material/add_photo_alternate:")
                st.caption("Upload a claim photo")


if submitted:
    result = None
    with st.spinner("Assessing the claim...", show_time=True):
        try:
            response = requests.post(
                api_url,
                data={
                    "claim_type": claim_type,
                    "company_id": company_id,
                    "claimed_amount": claimed_amount,
                    "insured_value": insured_value,
                },
                files={"image": (photo.name, photo.getvalue(), photo.type)},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"The prediction API returned an error: {exc}")

    if result is None:
        st.stop()
    if not result.get("is_accident_photo", False):
        st.warning(
            result.get(
                "message",
                "This image was not verified as a matching damaged-vehicle photo.",
            )
        )
        if result.get("detected_objects"):
            st.caption(
                "Detected objects: "
                + ", ".join(result["detected_objects"])
            )
        st.stop()

    st.session_state.recent_claims.insert(
        0,
        {
            "type": claim_options[claim_type],
            "decision": result["decision"],
            "payout": result["payout"],
        },
    )
    st.session_state.recent_claims = st.session_state.recent_claims[:8]

    st.space("small")
    decision = result["decision"]
    decision_color = {
        "approved": "green",
        "review": "orange",
        "rejected": "red",
    }[decision]

    st.subheader("Assessment")
    with st.container(horizontal=True, key="metrics_panel"):
        st.metric(
            "Decision",
            decision.title(),
            border=True,
            chart_data=None,
        )
        st.metric(
            "Estimated payout",
            format_inr(result["payout"]),
            border=True,
        )
        st.metric(
            "Damage severity",
            str(result["severity"]).title(),
            border=True,
        )
        st.metric(
            "Fraud risk",
            str(result["fraud_risk"]).title(),
            border=True,
        )

    details_col, comparison_col = st.columns([0.85, 1.15], gap="large")
    with details_col:
        with st.container(border=True):
            st.markdown(f":{decision_color}-badge[{decision.title()}]")
            st.write(f"Photo confidence: **{result['confidence']:.0%}**")
            st.write(f"Claim-to-loss: **{format_inr(result['assessed_loss_inr'])}**")
            st.write(f"Estimated loss: **{format_inr(result['estimated_loss_inr'])}**")

            if result.get("fraud_signals"):
                st.markdown("**Assessment signals**")
                for signal in result["fraud_signals"]:
                    st.caption(f"• {signal}")

    with comparison_col:
        with st.container(border=True):
            st.subheader("Insurer comparison")
            comparison = [
                row
                for row in result["comparison"]
                if row.get("covered")
            ]
            if comparison:
                table = pd.DataFrame(
                    [
                        {
                            "Insurer": row["company"],
                            "Payout": format_inr(row["payout"]),
                            "Coverage": f"{row['coverage']:.0%}",
                            "Deductible": format_inr(row["deductible"]),
                        }
                        for row in comparison
                    ]
                )
                st.dataframe(table, hide_index=True, width="stretch")
            else:
                st.caption("No insurer plans cover this claim type.")
