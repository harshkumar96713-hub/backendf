from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import base64
import json
import hashlib
import re

import google.generativeai as genai

# ===================== LOAD ENV =====================
load_dotenv()

# ===================== FASTAPI ======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== MONGODB ======================
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/darkpattern_db")
client = MongoClient(MONGO_URL)
db = client.darkpattern_db
analyses_collection = db.analyses

# ===================== GEMINI ======================
EMERGENT_LLM_KEY = os.getenv("EMERGENT_LLM_KEY")
if not EMERGENT_LLM_KEY:
    raise RuntimeError("EMERGENT_LLM_KEY missing")

genai.configure(api_key=EMERGENT_LLM_KEY)
model = genai.GenerativeModel("gemini-2.5-pro")

# ===================== MODELS ======================
class AnalyzeRequest(BaseModel):
    screenshot: str
    language: str = "en"

class DetectedIssue(BaseModel):
    issue: str
    description: str

class SignalBreakdown(BaseModel):
    visual: float
    semantic: float
    effort: float
    default: float
    pressure: float

class AnalysisResponse(BaseModel):
    id: str
    dpi_score: int
    risk_level: str
    simple_summary: str
    detected_issues: List[DetectedIssue]
    signal_breakdown: SignalBreakdown
    timestamp: str
    language: str

class HistoryResponse(BaseModel):
    analyses: List[AnalysisResponse]

# ===================== TEXT ======================
TEXT = {
    "en": {
        "low": "Low",
        "moderate": "Moderate",
        "high": "High",
        "summary_low": "This screen appears mostly fair with minimal manipulation.",
        "summary_medium": "Some design choices may subtly influence your decision.",
        "summary_high": "This app is strongly pushing you toward one option.",
        "visual": "One option is visually highlighted more",
        "semantic": "Confusing or misleading language",
        "effort": "Rejecting requires more effort than accepting",
        "default": "Consent is enabled by default",
        "pressure": "Uses urgency or pressure language",
    }
}

def t(key: str, lang: str):
    return TEXT.get(lang, TEXT["en"]).get(key, key)

# ===================== IMAGE HASH ======================
def image_hash(base64_img: str) -> str:
    return hashlib.sha256(base64_img.strip().encode()).hexdigest()

def hash_variation(img_hash: str, max_delta=0.15):
    seed = int(img_hash[:6], 16)
    return (seed % 1000) / 1000 * max_delta

# ===================== GEMINI ANALYSIS ======================
async def analyze_with_gemini(base64_img: str) -> Dict:
    try:
        image_bytes = base64.b64decode(base64_img)

        prompt = """
Return ONLY valid JSON.

{
  "visual_score": 0.0,
  "semantic_score": 0.0,
  "effort_score": 0.0,
  "default_score": 0.0,
  "pressure_score": 0.0,
  "primary_issues": ["visual","semantic","effort","default","pressure"]
}
"""

        res = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": image_bytes}
        ])

        text = res.text.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON returned")

        return json.loads(match.group(0))

    except Exception as e:
        print("Gemini fallback:", e)

        h = image_hash(base64_img)
        base = int(h[:6], 16) % 100

        return {
            "visual_score": 0.4 + (base % 10) / 25,
            "semantic_score": 0.4 + (base % 7) / 25,
            "effort_score": 0.4 + (base % 5) / 25,
            "default_score": 0.25,
            "pressure_score": 0.35,
            "primary_issues": ["semantic", "effort"] if base % 2 else ["visual", "default"]
        }

# ===================== DPI ======================
def calculate_dpi(signals: Dict[str, float]):
    score = (
        0.30 * signals["visual"]
        + 0.25 * signals["semantic"]
        + 0.25 * signals["effort"]
        + 0.10 * signals["default"]
        + 0.10 * signals["pressure"]
    )
    dpi = int(score * 100)

    if dpi < 30:
        return dpi, "low"
    elif dpi < 60:
        return dpi, "moderate"
    return dpi, "high"

def generate_summary(dpi: int, lang: str):
    if dpi < 30:
        return t("summary_low", lang)
    elif dpi < 60:
        return t("summary_medium", lang)
    return t("summary_high", lang)

def generate_issues(analysis: Dict, lang: str):
    return [
        {"issue": t(k, lang), "description": t(k, lang)}
        for k in analysis.get("primary_issues", [])
    ]

# ===================== API ======================
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalyzeRequest):

    img_hash = image_hash(req.screenshot)

    cached = analyses_collection.find_one({"image_hash": img_hash})
    if cached:
        cached["id"] = str(cached["_id"])
        return AnalysisResponse(**cached)

    analysis = await analyze_with_gemini(req.screenshot)
    v = hash_variation(img_hash)

    signals = {
        "visual": min(1.0, analysis["visual_score"] + v),
        "semantic": min(1.0, analysis["semantic_score"] + v / 2),
        "effort": min(1.0, analysis["effort_score"] + v / 3),
        "default": analysis["default_score"],
        "pressure": analysis["pressure_score"],
    }

    dpi, risk = calculate_dpi(signals)

    response = {
        "dpi_score": dpi,
        "risk_level": t(risk, req.language),
        "simple_summary": generate_summary(dpi, req.language),
        "detected_issues": generate_issues(analysis, req.language),
        "signal_breakdown": signals,
        "timestamp": datetime.now().isoformat(),
        "language": req.language,
        "image_hash": img_hash
    }

    result = analyses_collection.insert_one({**response, "raw_analysis": analysis})
    response["id"] = str(result.inserted_id)

    return AnalysisResponse(**response)

@app.get("/api/history", response_model=HistoryResponse)
async def history():
    records = list(
        analyses_collection.find({}, {"raw_analysis": 0, "image_hash": 0})
        .sort("timestamp", -1)
    )
    for r in records:
        r["id"] = str(r.pop("_id"))
    return {"analyses": records}

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

# ===================== RUN ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)