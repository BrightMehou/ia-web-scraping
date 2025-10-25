from fastapi import FastAPI
from src.util import extract_job_info

app = FastAPI(
    title="Job Offer Information Extractor API",
    description="API simple pour extraire des informations d'offres d'emploi en utilisant un LLM.",
    version="0.1.0",
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"msg": "API de l'extracteur d'offres d'emploi est en cours d'exécution."}


@app.post("/extract")
async def extract(url: str, api_key: str) -> dict | None:
    return extract_job_info(url, api_key)
