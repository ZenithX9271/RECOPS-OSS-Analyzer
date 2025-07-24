# === FILE: render_backend.py ===
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import uuid

from oss_power_analyser import analyze_multiple_repos_with_logs, save_all
from history_tracker import save_history_entry

app = FastAPI()

class RepoList(BaseModel):
    repo_urls: List[str]

@app.post("/analyze")
async def analyze_repos(payload: RepoList):
    logs = []
    def logger(msg):
        logs.append(msg)

    try:
        results = analyze_multiple_repos_with_logs(payload.repo_urls, log_fn=logger)
        timestamp = save_all(results)
        save_history_entry(timestamp)
        return {
            "status": "success",
            "logs": logs,
            "features": results,
            "timestamp": timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("render_backend:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")

# === FILE: requirements.txt ===
fastapi
uvicorn[standard]
PyGithub
python-dotenv
pandas
sentence-transformers
faiss-cpu
streamlit
requests

# === FILE: render.yaml ===
services:
  - type: web
    name: oss-analyzer-api
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn render_backend:app --host 0.0.0.0 --port 10000"
    plan: free

# === FILE: streamlit_ui.py ===
import os
import streamlit as st
import requests

st.title("OSS Power Analyzer (via Render Backend)")

api_url = os.getenv("API_URL", "http://localhost:10000/analyze")

urls_input = st.text_area("Enter one or more GitHub repo URLs (one per line)")

if st.button("Analyze"):
    repo_urls = [url.strip() for url in urls_input.strip().splitlines() if url.strip()]
    with st.spinner("Analyzing..."):
        try:
            response = requests.post(api_url, json={"repo_urls": repo_urls})
            response.raise_for_status()
            res = response.json()
            for log in res.get("logs", []):
                st.text(log)
            st.success("✅ Analysis complete. Download results below.")
            timestamp = res.get("timestamp")
            if timestamp:
                st.markdown(f"**Timestamp:** {timestamp}")
                st.markdown(f"Download CSV: `/features_output_{timestamp}.csv` or JSON: `/features_output_{timestamp}.json`")
        except Exception as e:
            st.error(f"❌ Failed: {e}")
