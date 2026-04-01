from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, traceback, uuid

from agent_explain import agent_executor

app = FastAPI(title="Agentic AI Data Scientist")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

sessions = {}

def normalize_result(r):

    if not isinstance(r, dict):
        return {"type": "text", "content": str(r)}

    r_type = r.get("type")

    if r_type == "plot" and "b64" in r:
        return r

    if r_type == "text":
        return {
            "type": "text",
            "content": r.get("content", "")
        }

    if r_type == "table":
        return {
            "type": "table",
            "data": r.get("data", {})
        }

    # fallback → treat as table
    return {
        "type": "table",
        "data": r
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{file.filename}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "file_path": file_path
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze")
async def analyze(data: dict):

    try:
        session_id = data.get("session_id")
        file_path = data.get("file_path")
        query = data.get("query")

        if not session_id or not file_path or not query:
            return {"status": "error", "message": "Missing fields"}

        if session_id not in sessions:
            sessions[session_id] = {"history": []}

        sessions[session_id]["history"].append({
            "role": "user",
            "content": query
        })

        history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in sessions[session_id]["history"]]
        )

        prompt = f"""
You are an AI Data Scientist.

Dataset: {file_path}

Conversation:
{history_text}

User Query:
{query}
"""

        response = agent_executor.invoke({"input": prompt})

        insights = response.get("output", "No insights generated")
        steps = response.get("intermediate_steps", [])

        results = []

        for step in steps:
            try:
                observation = step[1]
                normalized = normalize_result(observation)
                results.append(normalized)
            except:
                results.append({
                    "type": "text",
                    "content": "⚠️ Error processing step"
                })

        sessions[session_id]["history"].append({
            "role": "assistant",
            "content": insights
        })

        return {
            "status": "success",
            "insights": insights,
            "results": results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }