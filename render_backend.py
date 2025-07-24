# === FILE: render_backend.py ===
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import uvicorn
import oss_power_analyser
import os
import uuid

app = FastAPI()

class RepoList(BaseModel):
    repo_urls: List[str]

@app.post("/analyze")
async def analyze_repos(payload: RepoList):
    logs = []
    def logger(msg):
        logs.append(msg)

    try:
        results = oss_power_analyser.analyze_multiple_repos_with_logs(payload.repo_urls, log_fn=logger)
        oss_power_analyser.save_all(results)
        return {
            "status": "success",
            "logs": logs,
            "features": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === FILE: requirements.txt ===
fastapi
uvicorn
github
python-dotenv
langchain
groq
pandas
python-dotenv
streamlit

# === FILE: render.yaml ===
services:
  - type: web
    name: oss-analyzer-api
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn render_backend:app --host 0.0.0.0 --port 10000"
    plan: free

# === FILE: streamlit_ui.py ===
import streamlit as st
import requests

st.title("OSS Power Analyzer (via Render Backend)")

urls_input = st.text_area("Enter one or more GitHub repo URLs (one per line)")

if st.button("Analyze"):
    repo_urls = [url.strip() for url in urls_input.strip().splitlines() if url.strip()]
    with st.spinner("Analyzing..."):
        try:
            response = requests.post(
                "https://oss-analyzer-api.onrender.com/analyze",
                json={"repo_urls": repo_urls}
            )
            response.raise_for_status()
            res = response.json()
            for log in res.get("logs", []):
                st.text(log)
            st.success("✅ Analysis complete. Check download folder for outputs.")
        except Exception as e:
            st.error(f"❌ Failed: {e}")
