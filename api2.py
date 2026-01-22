from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = FastAPI()

MODEL_PATH = "models/model_log_reg.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"  # À sauvegarder lors de l'entraînement pour reproductibilité

# Chargement du modèle et du vectorizer (à entraîner une fois avec les mêmes données)
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Le modèle {MODEL_PATH} est manquant. Placez-le dans le répertoire courant.")

model = joblib.load(MODEL_PATH)
if not os.path.exists(VECTORIZER_PATH):
    # Vectorizer par défaut si non sauvegardé (à adapter selon votre entraînement)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
else:
    vectorizer = joblib.load(VECTORIZER_PATH)

class TextInput(BaseModel):
    text: str

@app.post("/predict-sentiment")
def predict_sentiment(data: TextInput):
    # Préprocessing du tweet
    tweet_vectorized = vectorizer.transform([data.text])
    # Prédiction (0: négatif, 1: positif - adaptez selon vos labels)
    prediction = model.predict(tweet_vectorized)[0]
    probability = model.predict_proba(tweet_vectorized)[0]
    
    sentiment = "négatif" if prediction == 0 else "positif"
    return {
        "sentiment": sentiment,
        "confidence": float(max(probability)),
        "prob_negatif": float(probability[0]),
        "prob_positif": float(probability[1])
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
