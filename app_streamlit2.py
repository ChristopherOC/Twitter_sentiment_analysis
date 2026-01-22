import streamlit as st
import requests

API_URL = "http://localhost:8000/predict-sentiment"  # Endpoint adapté

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Analyse de Sentiment (Logistic Regression)")
st.write("Analyse de sentiment en temps réel via API FastAPI + LogReg.")

text_input = st.text_area(
    "Entrez un tweet à analyser",
    placeholder="I like to eat at Pizza Hut, this is fire !!!",
    height=120
)

if st.button("Analyser", type="primary"):
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

                # Parsing adapté à la réponse LogReg
                prob_positif = result["prob_positif"]
                prob_negatif = result["prob_negatif"]
                sentiment = result["sentiment"]
                confidence = result["confidence"]

                st.subheader("Résultats")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Positif", f"{prob_positif:.2%}")
                with col2:
                    st.metric("Négatif", f"{prob_negatif:.2%}")

                st.metric("Confiance", f"{confidence:.2%}")

                # Barre de progression (positif)
                st.progress(prob_positif)

                # Décision finale
                if prob_positif > 0.5:
                    st.success(f"Sentiment global : **{sentiment.upper()}** ✅")
                else:
                    st.error(f"Sentiment global : **{sentiment.upper()}** ❌")

            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de contacter l'API FastAPI (lancez `uvicorn api_logreg:app --reload`).")
            except requests.exceptions.Timeout:
                st.error("⏱️ L'API met trop de temps à répondre.")
            except KeyError as e:
                st.error(f"Réponse API inattendue : {e}. Vérifiez le format JSON.")
            except Exception as e:
                st.error(f"Erreur : {str(e)}")
