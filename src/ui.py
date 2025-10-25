import pandas as pd
import requests
import streamlit as st

from util import generate_random_data

data = generate_random_data(2)
st.set_page_config(page_title="Extracteur d'offres d'emploi", layout="wide")
st.title("📝 Extracteur d'informations d'offres d'emploi avec Gemini")

api_key = st.text_input("🔑 Entre ta clé API Gemini", type="password")
url = st.text_input("🌍 Entre l'URL de l'offre d'emploi")

if st.button("Analyser l'offre"):
    if not api_key:
        st.error("Merci de renseigner une clé API Gemini.")
    elif not url:
        st.error("Merci de renseigner une URL.")
    else:

        with st.spinner("🔎 Analyse en cours..."):
            try:
                res = requests.post(
                    "http://localhost:8000/extract",
                    params={"url": url, "api_key": api_key},
                )
                res_json = res.json()
                if res_json is None:
                    st.error("Impossible de charger la page.")
                else:
                    st.success("✅ Extraction réussie !")
                    st.subheader("Résultat brut (JSON)")
                    st.json(res_json)
                    rows = []
                    # Hard skills
                    for skill in res_json["skills"]:
                        rows.append(
                            {
                                "city": res_json["city"],
                                "contract_type": res_json["contract_type"],
                                "skill_type": "hard",
                                "skill": skill,
                            }
                        )

                    # Soft skills
                    for skill in res_json["soft_skills"]:
                        rows.append(
                            {
                                "city": res_json["city"],
                                "contract_type": res_json["contract_type"],
                                "skill_type": "soft",
                                "skill": skill,
                            }
                        )
                    st.subheader("📋 Résumé formaté")
                    new_skills = pd.json_normalize(rows)
                    st.dataframe(new_skills)
                    new_skills["nb"] = 1
                    data = (
                        pd.concat([data, new_skills], ignore_index=True)
                        .groupby(
                            ["city", "contract_type", "skill_type", "skill"],
                            as_index=False,
                        )["nb"]
                        .sum()
                    )

            except Exception as e:
                st.error(f"Erreur pendant l'analyse : {e}")

st.markdown("---")
st.dataframe(data)
