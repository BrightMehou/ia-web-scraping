import os
from typing import Optional

import pandas as pd
import numpy as np
from langchain.agents import create_agent
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field


class Job(BaseModel):
    city: Optional[str] = Field(
        default=None, description="City name location of the job"
    )
    contract_type: Optional[str] = Field(
        default=None, description="Type of contract (CDI, CDD, alternance...)"
    )
    skills: Optional[list[str]] = Field(
        default=None, description="Required technical skills (languages, tools...)"
    )
    soft_skills: Optional[list[str]] = Field(
        default=None, description="Soft skills required for the job"
    )


cities: list[str] = [
    "Paris",
    "Marseille",
    "Lyon",
    "Toulouse",
    "Nice",
    "Nantes",
    "Strasbourg",
    "Montpellier",
    "Bordeaux",
    "Lille",
]
contract_types: list[str] = [
    "Internship",
    "Apprenticeship",
    "Full-Time",
    "Part-Time",
    "Freelance",
]

hard_skills: list[str] = [
    "Python",
    "SQL",
    "Docker",
    "Kubernetes",
    "AWS",
    "GCP",
    "Azure",
]

soft_skills: list[str] = [
    "communication",
    "travail d'équipe",
    "adaptabilité",
    "créativité",
    "autonomie",
    "leadership",
    "rigueur",
    "empathie",
    "organisation",
    "esprit critique",
    "curiosité",
    "gestion du stress",
    "esprit d'analyse",
    "écoute",
    "collaboration",
    "négociation",
    "gestion du temps",
    "polyvalence",
    "prise d'initiative",
    "persévérance",
    "motivation",
    "proactivité",
    "fiabilité",
    "éthique",
    "agilité",
    "flexibilité",
    "gestion de projet",
    "esprit de synthèse",
    "autodiscipline",
    "patience",
]


def generate_random_data() -> pd.DataFrame:
    df_cities = pd.DataFrame({"city": cities})
    df_contracts = pd.DataFrame({"contract_type": contract_types})
    df_hard = pd.DataFrame({
        "skill_type": "hard",
        "skill": hard_skills
    })
    df_soft = pd.DataFrame({
        "skill_type": "soft",
        "skill": soft_skills
    })

    df_skills = pd.concat([df_hard, df_soft], ignore_index=True)

    # Produit cartésien entre city, contract et skills
    df = (
        df_cities
        .merge(df_contracts, how="cross")
        .merge(df_skills, how="cross")
    )

    df["nb"] = np.random.randint(10, 101, size=len(df))

    return df


def load_pages(url: str):
    loader = WebBaseLoader(web_path=[url])
    return list(loader.lazy_load())


def extract_job_info(url: str, api_key: str) -> Optional[Job]:
    docs = load_pages(url)
    if not docs:
        return None
    os.environ["GOOGLE_API_KEY"] = api_key
    agent = create_agent(
        model="google_genai:gemini-2.5-pro",
        response_format=Job,
        system_prompt="Vous êtes un expert en ressources humaines spécialisé dans l'analyse des offres d'emploi.",
    )

    doc = docs[0]
    text = doc.page_content
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyse l'offre d'emploi suivante et extrait les informations demandées : {text}",
                }
            ]
        }
    )

    return result["structured_response"].model_dump()
