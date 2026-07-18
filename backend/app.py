from __future__ import annotations

import json
import os
import re
from typing import Any

import boto3
import httpx
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
SEGMENTATION_SERVICE_URL = os.getenv("SEGMENTATION_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")

SEGMENT_STRATEGY = {
    "Affluent Investors": {
        "behavior_label": "Investment Momentum",
        "customer_story": "customers with strong balances and investment activity",
        "tone": "confident, premium, opportunity-led",
    },
    "Emerging Digital Everyday": {
        "behavior_label": "Digital Everyday Builder",
        "customer_story": "mobile-first customers who use the app often and respond to simple digital journeys",
        "tone": "friendly, clear, low-friction",
    },
    "High Spend Credit Active": {
        "behavior_label": "High Spend Credit Active",
        "customer_story": "customers with high card activity who may value a cleaner repayment plan",
        "tone": "practical, transparent, control-focused",
    },
    "High Spend and Frequent Overdraft": {
        "behavior_label": "Cashflow Pressure Window",
        "customer_story": "customers showing repeated overdraft events and short-term cashflow pressure",
        "tone": "supportive, non-judgemental, stabilising",
    },
    "Low Digital Engagement and Cash-Oriented": {
        "behavior_label": "Assisted Banking Preference",
        "customer_story": "customers who use cash more often and may need simpler guidance before acting digitally",
        "tone": "plain, reassuring, step-by-step",
    },
    "Stable Salary Builders": {
        "behavior_label": "Stable Salary Builder",
        "customer_story": "salary customers with stable inflow, growing relationship depth, and planning potential",
        "tone": "encouraging, future-planning, trustworthy",
    },
}

FALLBACK_SEGMENTS = [
    {"segment_id": 1, "segment_name": "Affluent Investors", "customer_count": 142, "average_confidence": 0.72},
    {"segment_id": 2, "segment_name": "Emerging Digital Everyday", "customer_count": 188, "average_confidence": 0.68},
    {"segment_id": 3, "segment_name": "High Spend Credit Active", "customer_count": 164, "average_confidence": 0.70},
    {"segment_id": 4, "segment_name": "High Spend and Frequent Overdraft", "customer_count": 151, "average_confidence": 0.66},
    {"segment_id": 5, "segment_name": "Low Digital Engagement and Cash-Oriented", "customer_count": 177, "average_confidence": 0.64},
    {"segment_id": 6, "segment_name": "Stable Salary Builders", "customer_count": 178, "average_confidence": 0.71},
]

app = FastAPI(title="BNZ AI Marketing Hybrid API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CampaignRequest(BaseModel):
    product: str = Field(min_length=3, description="Any product, campaign, or banking offer typed by the presenter")
    brief: str = Field(default="", description="Optional campaign intent in plain human language")
    objective: str = "generate personalised compliant marketing and support conversion"
    segment_id: int | None = None


class AssistantRequest(BaseModel):
    question: str = Field(min_length=2)
    product: str = Field(min_length=3)
    segment_name: str = ""
    behavior_label: str = ""
    message: str = ""
    customer_id: str | None = None


def bedrock_client():
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        config=Config(read_timeout=120, connect_timeout=10, retries={"max_attempts": 2}),
    )


async def fetch_json(path: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=4.0) as client:
        response = await client.get(f"{SEGMENTATION_SERVICE_URL}{path}")
        response.raise_for_status()
        return response.json()


async def get_segments() -> tuple[list[dict[str, Any]], str]:
    try:
        data = await fetch_json("/v1/segments")
        return data.get("segments", []), "live-ml-api"
    except Exception:
        return FALLBACK_SEGMENTS, "fallback-segments"


async def get_customers(segment_id: int, limit: int = 8) -> tuple[list[dict[str, Any]], str]:
    try:
        data = await fetch_json(f"/v1/customers?segment_id={segment_id}&limit={limit}")
        return data.get("customers", []), "live-ml-api"
    except Exception:
        fallback = [
            {
                "customer_id": f"DEMO-{segment_id}-{i + 1:03d}",
                "segment_id": segment_id,
                "segment_name": next((s["segment_name"] for s in FALLBACK_SEGMENTS if s["segment_id"] == segment_id), "Demo Segment"),
                "assignment_confidence": round(0.82 - i * 0.03, 2),
            }
            for i in range(limit)
        ]
        return fallback, "fallback-customers"


async def get_model_features() -> tuple[list[str], str]:
    try:
        data = await fetch_json("/v1/model/features")
        return data.get("required_features", []), "live-ml-api"
    except Exception:
        return [
            "avg_monthly_inflow_6m",
            "salary_inflow_ratio",
            "inflow_cv_6m",
            "avg_balance_6m",
            "min_balance_6m",
            "avg_monthly_spend_6m",
            "monthly_txn_count_6m",
            "digital_txn_ratio",
            "cash_withdrawal_ratio",
            "discretionary_spend_ratio",
            "travel_spend_ratio",
            "investment_contribution_ratio",
            "credit_card_utilisation",
            "days_since_last_txn",
            "monthly_app_logins_3m",
            "products_held",
            "overdraft_events_6m",
        ], "fallback-features"


def strategy_for(segment_name: str) -> dict[str, str]:
    return SEGMENT_STRATEGY.get(
        segment_name,
        {
            "behavior_label": segment_name or "Behavioural Segment",
            "customer_story": "customers grouped by bank-owned behavioural signals",
            "tone": "clear, helpful, compliant",
        },
    )


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def invoke_bedrock_text(prompt: str, max_tokens: int = 700, temperature: float = 0.6) -> str:
    response = bedrock_client().converse(
        modelId=BEDROCK_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    return response["output"]["message"]["content"][0]["text"]


def local_copy(segment_name: str, product: str) -> dict[str, Any]:
    strategy = strategy_for(segment_name)
    label = strategy["behavior_label"]
    return {
        "headline": f"A BNZ option for {label}",
        "message": f"BNZ can present this exact offer: {product}. The app can show repayment estimates and next steps before the customer decides.",
        "app_notification": "Review this exact BNZ offer in the app.",
        "lex_reply": "I can answer questions about this specific offer, including repayment estimates, eligibility steps, and what happens before applying.",
        "compliance_note": "Uses behavioural segments only. Do not infer protected identity, occupation, or life status.",
        "cta": "Review in BNZ App",
        "send_reason": f"Matched to {label} based on the ML segmentation API output.",
        "segment_label": label,
        "bedrock_status": "fallback-local-copy",
    }


def generate_with_bedrock(segment_name: str, product: str, brief: str, objective: str) -> dict[str, Any]:
    strategy = strategy_for(segment_name)
    prompt = f"""
You are writing REAL segment-specific BNZ marketing copy from the exact product text typed by the presenter.
Do not use a generic banking template. Do not replace the product with a generic loan.
The audience is selected by an ML behavioural segmentation API, not by identity labels.

EXACT product / offer typed by presenter:
<<<{product}>>>

Optional bank employee brief:
<<<{brief or 'No extra brief. Use the exact product text and adapt only the angle for the segment.'}>>>

Objective:
{objective}

ML behavioural segment name:
{segment_name}
Behavioural label shown in the demo:
{strategy['behavior_label']}
Observed behavioural story:
{strategy['customer_story']}
Recommended tone:
{strategy['tone']}

Return ONLY valid JSON with these exact keys:
headline, message, app_notification, lex_reply, compliance_note, cta, send_reason.

Strict grounding rules:
- Write in English.
- Use the presenter's product details directly. If the product says "trip loan", the copy must say trip loan or travel/trip funding.
- Preserve concrete details from the typed product: amount, product purpose, app repayment estimates, timing, channel, and any conditions.
- Do not silently correct or invent numbers. If the typed amount is unusual, either keep it exactly or avoid repeating the number.
- Do not use generic phrases like "manage your expenses with ease", "financial freedom", "unlock your dreams", or "tailored solution".
- Do not claim approval unless the typed product explicitly says pre-approved.
- Do not mention protected identity, ethnicity, family status, student status, income class, or occupation.
- Create a different reason and angle for this behavioural segment using only behavioural signals.
- Keep headline under 12 words.
- Keep message under 45 words.
- Keep app_notification under 20 words.
- lex_reply should answer like a BNZ app assistant for THIS exact product, not for a generic loan.
""".strip()

    try:
        parsed = extract_json(invoke_bedrock_text(prompt))
        parsed["segment_label"] = strategy["behavior_label"]
        parsed["bedrock_status"] = "generated-by-bedrock"
        return parsed
    except (BotoCoreError, ClientError, KeyError, json.JSONDecodeError, ValueError) as exc:
        fallback = local_copy(segment_name, product)
        fallback["bedrock_status"] = f"bedrock-error-fallback: {type(exc).__name__}"
        return fallback


def answer_with_bedrock(request: AssistantRequest) -> dict[str, str]:
    prompt = f"""
You are a concise BNZ in-app financial assistant for a demo.
Answer the customer's question based on the campaign context.
Do not invent exact rates, approvals, or legal terms. If needed, say the app can show estimates or eligibility steps.

Product or offer:
{request.product}

Current behavioural segment:
{request.behavior_label} / {request.segment_name}

Ad message shown to customer:
{request.message}

Customer ID, if shown in demo:
{request.customer_id or 'not provided'}

Customer question:
{request.question}

Return ONLY valid JSON with this exact key: answer.
Keep answer under 55 words.
""".strip()
    try:
        parsed = extract_json(invoke_bedrock_text(prompt, max_tokens=300, temperature=0.4))
        return {"answer": parsed.get("answer", "I can help compare options and show the next step in the BNZ app."), "bedrock_status": "generated-by-bedrock"}
    except (BotoCoreError, ClientError, KeyError, json.JSONDecodeError, ValueError) as exc:
        return {
            "answer": "I can explain repayment estimates, eligibility steps, fees, and what happens next before you decide to continue.",
            "bedrock_status": f"bedrock-error-fallback: {type(exc).__name__}",
        }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    segmentation_ok = False
    try:
        data = await fetch_json("/health")
        segmentation_ok = data.get("status") == "ok"
    except Exception:
        pass
    return {
        "status": "ok",
        "aws_profile": AWS_PROFILE,
        "aws_region": AWS_REGION,
        "bedrock_model_id": BEDROCK_MODEL_ID,
        "segmentation_service_url": SEGMENTATION_SERVICE_URL,
        "segmentation_endpoints_used": ["GET /v1/segments", "GET /v1/customers?segment_id={id}"],
        "segmentation_ok": segmentation_ok,
    }


@app.get("/api/ml-evidence")
async def ml_evidence() -> dict[str, Any]:
    segments, segment_source = await get_segments()
    features, feature_source = await get_model_features()
    evidence_segments = []
    for segment in segments:
        customers, customer_source = await get_customers(int(segment["segment_id"]), limit=5)
        evidence_segments.append({
            "segment_id": int(segment["segment_id"]),
            "segment_name": segment["segment_name"],
            "behavior_label": strategy_for(segment["segment_name"])["behavior_label"],
            "customer_count": segment.get("customer_count"),
            "average_confidence": segment.get("average_confidence"),
            "sample_customer_ids": [customer["customer_id"] for customer in customers],
            "customer_source": customer_source,
        })
    return {
        "service": "bank-segmentation-service",
        "base_url": SEGMENTATION_SERVICE_URL,
        "source": segment_source,
        "model_version": "1.0.0-demo",
        "algorithm": "K-Means clustering over bank-owned behavioural features",
        "selected_k": len(segments),
        "training_customer_count": 1000,
        "features_source": feature_source,
        "features_used": features,
        "identity_fields_excluded": [
            "customer name",
            "phone number",
            "passport",
            "address document",
            "occupation inference",
            "student / family / income-class labels",
        ],
        "endpoints_used": [
            "GET /health",
            "GET /v1/segments",
            "GET /v1/model/features",
            "GET /v1/customers?segment_id={id}&limit=5",
        ],
        "segments": evidence_segments,
    }


@app.get("/api/audience")
async def audience() -> dict[str, Any]:
    segments, segment_source = await get_segments()
    enriched = []
    for segment in segments:
        name = segment["segment_name"]
        customers, customer_source = await get_customers(int(segment["segment_id"]), limit=8)
        enriched.append({
            **segment,
            "behavior_label": strategy_for(name)["behavior_label"],
            "customer_story": strategy_for(name)["customer_story"],
            "customers": customers,
            "ml_source": {"segments": segment_source, "customers": customer_source},
        })
    return {"segments": enriched, "ml_api_url": SEGMENTATION_SERVICE_URL}


@app.post("/api/generate")
async def generate_campaign(request: CampaignRequest) -> dict[str, Any]:
    segments, segment_source = await get_segments()
    if request.segment_id:
        segments = [s for s in segments if int(s["segment_id"]) == request.segment_id]
    if not segments:
        raise HTTPException(status_code=404, detail="No matching segment found")

    outputs = []
    for segment in segments[:6]:
        segment_id = int(segment["segment_id"])
        segment_name = segment["segment_name"]
        customers, customer_source = await get_customers(segment_id, limit=8)
        creative = generate_with_bedrock(segment_name, request.product, request.brief, request.objective)
        outputs.append({
            "segment_id": segment_id,
            "segment_name": segment_name,
            "customer_count": segment.get("customer_count"),
            "average_confidence": segment.get("average_confidence"),
            "behavior_label": strategy_for(segment_name)["behavior_label"],
            "customer_story": strategy_for(segment_name)["customer_story"],
            "customers": customers,
            "creative": creative,
            "delivery": {
                "send_to_customer_ids": [customer["customer_id"] for customer in customers],
                "channel": "BNZ App push / in-app card",
                "ml_source": {"segments": segment_source, "customers": customer_source},
            },
        })
    return {
        "product": request.product,
        "campaign": outputs,
        "ml_api_url": SEGMENTATION_SERVICE_URL,
        "ml_endpoints_used": ["GET /v1/segments", "GET /v1/customers?segment_id={id}"],
    }


@app.post("/api/assistant")
async def assistant(request: AssistantRequest) -> dict[str, str]:
    return answer_with_bedrock(request)
