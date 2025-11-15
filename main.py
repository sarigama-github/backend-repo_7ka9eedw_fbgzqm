import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents
from schemas import Drug, InteractionRule, QuizQuestion, PaperSummary

app = FastAPI(title="Pharmacy EdTech Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Pharmacy EdTech Backend is running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = getattr(db, 'name', '✅ Connected')
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:20]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:120]}"

    # Check environment variables
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    response["openai_api_key"] = "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not Set"
    return response


# ----------------------- Schema Introspection -----------------------
@app.get("/schema")
def get_schema_definitions():
    """Expose simplified schema definitions for the frontend viewer"""
    return {
        "drug": {
            "fields": [
                "name", "brand_names", "class_name", "indications",
                "contraindications", "side_effects", "mechanisms"
            ]
        },
        "interactionrule": {
            "fields": [
                "drug_a", "drug_b", "severity", "description", "management"
            ]
        },
        "quizquestion": {
            "fields": [
                "topic", "question", "options", "answer_index", "explanation", "difficulty"
            ]
        },
        "papersummary": {
            "fields": [
                "query", "title", "url", "abstract", "summary"
            ]
        }
    }


# ----------------------- Drug Database -----------------------
class DrugCreate(Drug):
    pass


@app.post("/api/drugs")
def add_drug(drug: DrugCreate):
    doc_id = create_document("drug", drug)
    return {"id": doc_id}


@app.get("/api/drugs/search")
def search_drugs(q: str, limit: int = 10):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    # Basic case-insensitive search across common fields
    import re
    regex = {"$regex": re.escape(q), "$options": "i"}
    filter_dict = {"$or": [
        {"name": regex},
        {"brand_names": regex},
        {"class_name": regex},
        {"indications": regex},
        {"side_effects": regex},
        {"mechanisms": regex},
    ]}
    results = get_documents("drug", filter_dict, limit)
    # Convert ObjectId to string if present
    for r in results:
        if "_id" in r:
            r["id"] = str(r.pop("_id"))
    return {"items": results}


# ----------------------- Interaction Simulation -----------------------
class InteractionRequest(BaseModel):
    drugs: List[str] = Field(..., description="List of generic drug names")


@app.post("/api/interactions/simulate")
def simulate_interactions(payload: InteractionRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    names = [d.strip().lower() for d in payload.drugs if d.strip()]
    pair_results: List[Dict[str, Any]] = []

    # Fetch all rules that match any of the provided names
    rules = list(db["interactionrule"].find({
        "$or": [
            {"drug_a": {"$in": names}},
            {"drug_b": {"$in": names}},
        ]
    }))

    def normalized_pair(a: str, b: str):
        return "::".join(sorted([a.lower(), b.lower()]))

    # Index rules for quick lookup
    rule_map = {normalized_pair(r.get("drug_a", ""), r.get("drug_b", "")): r for r in rules}

    # Evaluate all unique pairs
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = normalized_pair(names[i], names[j])
            rule = rule_map.get(key)
            if rule:
                item = {
                    "drug_a": rule.get("drug_a"),
                    "drug_b": rule.get("drug_b"),
                    "severity": rule.get("severity"),
                    "description": rule.get("description"),
                    "management": rule.get("management"),
                }
            else:
                item = {
                    "drug_a": names[i],
                    "drug_b": names[j],
                    "severity": "unknown",
                    "description": "No interaction rule found in the database.",
                    "management": "Consult reliable references and monitor as clinically indicated.",
                }
            pair_results.append(item)

    return {"pairs": pair_results}


# ----------------------- Chatbot -----------------------
class ChatPayload(BaseModel):
    message: str
    system_prompt: Optional[str] = "You are a helpful pharmacology assistant for pharmacy students. Give concise, evidence-based answers with references when appropriate."
    model: Optional[str] = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@app.post("/api/chat")
def chat(payload: ChatPayload):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Graceful fallback if no API key is provided
        return {
            "reply": "AI is not configured. Please set OPENAI_API_KEY to enable the chatbot.",
            "configured": False
        }
    try:
        # OpenAI SDK v1 pattern
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=payload.model,
            messages=[
                {"role": "system", "content": payload.system_prompt or ""},
                {"role": "user", "content": payload.message},
            ],
            temperature=0.3,
        )
        reply = completion.choices[0].message.content
        return {"reply": reply, "configured": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)[:200]}")


# ----------------------- Quiz Generation -----------------------
class QuizGenPayload(BaseModel):
    topic: str
    count: int = 5
    difficulty: Optional[str] = None


@app.post("/api/quizzes/generate")
def generate_quiz(payload: QuizGenPayload):
    api_key = os.getenv("OPENAI_API_KEY")
    questions: List[Dict[str, Any]] = []

    if not api_key:
        # Fallback: simple template questions without AI
        for i in range(payload.count):
            q = QuizQuestion(
                topic=payload.topic,
                question=f"Placeholder question {i+1} about {payload.topic}?",
                options=["Option A", "Option B", "Option C", "Option D"],
                answer_index=0,
                explanation="Set OPENAI_API_KEY to enable AI-generated questions.",
                difficulty=payload.difficulty or "easy"
            )
            create_document("quizquestion", q)
            questions.append(q.model_dump())
        return {"items": questions, "configured": False}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Create "
            f"{payload.count} multiple-choice questions for pharmacy students on {payload.topic}. "
            "Return strict JSON with an array named items. Each item must have: "
            "question (string), options (array of 4), answer_index (0-3), explanation (string). "
            f"Difficulty: {payload.difficulty or 'mixed'}. Keep them concise."
        )
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You output only JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        import json
        content = completion.choices[0].message.content
        data = json.loads(content)
        for item in data.get("items", []):
            q = QuizQuestion(
                topic=payload.topic,
                question=item.get("question"),
                options=item.get("options", [])[:4],
                answer_index=int(item.get("answer_index", 0)),
                explanation=item.get("explanation"),
                difficulty=payload.difficulty
            )
            create_document("quizquestion", q)
            questions.append(q.model_dump())
        return {"items": questions, "configured": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation error: {str(e)[:200]}")


# ----------------------- Research Assistant -----------------------
class ResearchPayload(BaseModel):
    query: str
    text: Optional[str] = None
    urls: Optional[List[str]] = None


@app.post("/api/research/summarize")
def summarize_research(payload: ResearchPayload):
    api_key = os.getenv("OPENAI_API_KEY")

    base_summary = ""
    if payload.text:
        base_summary = payload.text[:4000]
    elif payload.urls:
        base_summary = "\n".join(payload.urls)

    if not api_key:
        summary = (
            "AI summarization is not configured. Provide OPENAI_API_KEY to enable. "
            f"Query: {payload.query}. Provided context: {base_summary[:200]}"
        )
        ps = PaperSummary(query=payload.query, title=payload.query, summary=summary)
        create_document("papersummary", ps)
        return {"summary": summary, "configured": False}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Summarize the following content for pharmacy students. "
            "Focus on mechanisms, efficacy, safety, dosing, and key trial outcomes when relevant. "
            "Return a clear, bullet-style summary within 200-250 words.\n\n"
            f"Query: {payload.query}\n\nContext:\n{base_summary}"
        )
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You create concise, accurate medical summaries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        summary = completion.choices[0].message.content
        ps = PaperSummary(query=payload.query, title=payload.query, summary=summary)
        create_document("papersummary", ps)
        return {"summary": summary, "configured": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research summary error: {str(e)[:200]}")


# ----------------------- Seed Demo Data -----------------------
@app.post("/api/seed")
def seed_demo_data():
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    # Seed a couple of drugs and interactions for demo
    demo_drugs = [
        Drug(name="warfarin", brand_names=["Coumadin"], class_name="Anticoagulant", indications=["AF", "DVT"], side_effects=["Bleeding"], mechanisms="Vitamin K antagonist"),
        Drug(name="fluconazole", brand_names=["Diflucan"], class_name="Azole antifungal", indications=["Candidiasis"], side_effects=["Hepatotoxicity"], mechanisms="Inhibits fungal CYP450"),
        Drug(name="metoprolol", brand_names=["Lopressor"], class_name="Beta-1 blocker", indications=["HTN", "CHF"], side_effects=["Bradycardia"], mechanisms="Selective β1 blockade"),
    ]
    for d in demo_drugs:
        create_document("drug", d)

    rules = [
        InteractionRule(
            drug_a="warfarin",
            drug_b="fluconazole",
            severity="major",
            description="Fluconazole inhibits CYP2C9 increasing warfarin levels → bleeding risk.",
            management="Avoid or monitor INR closely; reduce warfarin dose as needed."
        ),
        InteractionRule(
            drug_a="metoprolol",
            drug_b="warfarin",
            severity="minor",
            description="No clinically significant PK interaction; monitor as usual.",
            management="No change typically required."
        ),
    ]
    for r in rules:
        create_document("interactionrule", r)

    return {"status": "ok", "seeded": {"drugs": len(demo_drugs), "rules": len(rules)}}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
