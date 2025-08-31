import os
import streamlit as st
from typing import Optional
from datetime import date
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




import asyncio  # <-- ajout

# ✅ Patch asyncio pour éviter "There is no current event loop..."
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
class Profil(BaseModel):
    title: Optional[str] = Field(default=None, description="Title of the job offer")
    company: Optional[str] = Field(default=None, description="Company offering the job")
    location: Optional[str] = Field(default=None, description="Location of the job")
    contract_type: Optional[str] = Field(default=None, description="Type of contract (CDI, CDD, alternance...)")
    publication_date: Optional[date] = Field(default=None, description="Date when the job was posted format YYYY-MM-DD")
    experience: Optional[str] = Field(default=None, description="Required experience level")
    skills: Optional[list[str]] = Field(default=None, description="Required technical skills (languages, tools...)")
    soft_skills: Optional[list[str]] = Field(default=None, description="Soft skills required for the job")
    salary: Optional[str] = Field(default=None, description="Salary range if mentioned")
    description: Optional[str] = Field(default=None, description="Full job description")


def load_pages(url: str):
    loader = WebBaseLoader(web_path=[url])
    return list(loader.lazy_load())


def compute_similarity(user_text: str, job_text: str, embedder) -> float:
    """Calcule la similarité cosinus entre deux textes via embeddings"""
    user_emb = np.array(embedder.embed_query(user_text))
    job_emb = np.array(embedder.embed_query(job_text))
    sim = cosine_similarity([user_emb], [job_emb])[0][0]
    return round(sim * 100, 2)


st.set_page_config(page_title="Extracteur d'offres d'emploi", layout="wide")
st.title("📝 Extracteur d'informations d'offres d'emploi avec Gemini")

api_key = st.text_input("🔑 Entre ta clé API Gemini", type="password")
url = st.text_input("🌍 Entre l'URL de l'offre d'emploi")

st.subheader("👤 Votre profil")
user_experience = st.text_area("Décrivez vos expériences (missions, responsabilités, projets...)")
user_skills = st.text_area("Listez vos compétences techniques et soft skills (séparées par des virgules)")

if st.button("Analyser l'offre"):
    if not api_key:
        st.error("Merci de renseigner une clé API Gemini.")
    elif not url:
        st.error("Merci de renseigner une URL.")
    else:
        

        with st.spinner("🔎 Analyse en cours..."):
            try:
                docs = load_pages(url)
                if not docs:
                    st.error("Impossible de charger la page.")
                else:
                    os.environ["GOOGLE_API_KEY"] = api_key
                    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    model = init_chat_model("gemini-2.5-flash", model_provider="google-genai")
                    structured_llm = model.with_structured_output(schema=Profil)

                    prompt_template = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are an expert extraction algorithm. "
                                "Only extract relevant information from the text. "
                                "If you do not know the value of an attribute asked to extract, "
                                "return null for the attribute's value.",
                            ),
                            ("human", "{text}"),
                        ]
                    )

                    doc = docs[0]
                    text = doc.page_content
                    prompt = prompt_template.invoke({"text": text})
                    res = structured_llm.invoke(prompt)

                    st.success("✅ Extraction réussie !")
                    st.subheader("Résultat brut (JSON)")
                    st.json(res.model_dump())

                    st.subheader("📋 Résumé formaté")
                    st.write(f"**Titre** : {res.title}")
                    st.write(f"**Entreprise** : {res.company}")
                    st.write(f"**Localisation** : {res.location}")
                    st.write(f"**Contrat** : {res.contract_type}")
                    st.write(f"**Date de publication** : {res.publication_date}")
                    st.write(f"**Expérience** : {res.experience}")
                    st.write(f"**Compétences techniques** : {', '.join(res.skills or [])}")
                    st.write(f"**Soft skills** : {', '.join(res.soft_skills or [])}")
                    st.write(f"**Salaire** : {res.salary}")
                    st.write("**Description complète** :")
                    st.write(res.description)

                    # --- Similarité avec le profil utilisateur ---
                    if user_experience or user_skills:
                        user_text = f"{user_experience} {user_skills}"
                        job_text = f"{res.title} {res.experience} {res.description} {' '.join(res.skills or [])} {' '.join(res.soft_skills or [])}"
                        score = compute_similarity(user_text, job_text, embedder)
                        st.subheader("📊 Score de similarité avec votre profil")
                        st.metric(label="Taux de correspondance (Gemini embeddings)", value=f"{score} %")
                    else:
                        st.info("👉 Ajoutez vos expériences et compétences pour obtenir un score de similarité.")

            except Exception as e:
                st.error(f"Erreur pendant l'analyse : {e}")
