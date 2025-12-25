import streamlit as st
import requests

API_URL = "http://localhost:8000/send-tweet"

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Sentiment Analysis (BERT)")
st.write("Analyse de sentiment en temps réel via une API FastAPI.")

text_input = st.text_area(
    "Entrez un texte à analyser",
    placeholder="I like to eat at Pizza Hut, this is fire !!!",
    height=120
)

if st.button("Analyser le tweet"):
    if not text_input.strip():
        st.warning("Veuillez entrer un texte.")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": text_input},
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()

                positive = result["positive"]
                negative = result["negative"]

                st.subheader("Résultat")
                st.metric("Positif ", f"{positive:.2%}")
                st.metric("Négatif ", f"{negative:.2%}")

                st.progress(positive)

                if positive > negative:
                    st.success("Sentiment global : POSITIF")
                else:
                    st.error("Sentiment global : NÉGATIF")

            except requests.exceptions.ConnectionError:
                st.error("Impossible de contacter l'API FastAPI.")
            except requests.exceptions.Timeout:
                st.error("L'API met trop de temps à répondre.")
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")
