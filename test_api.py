import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from api2 import app  # Import du bon modèle LogReg

client = TestClient(app)

def test_predict_sentiment_positive():
    """Teste un texte positif avec le modèle LogReg réel."""
    response = client.post("/predict-sentiment", json={"text": "I love this app!"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "prob_positif" in data  # Note: prob_positif (underscore) comme dans api2.py
    assert "prob_negatif" in data
    assert "confidence" in data
    # Le modèle LogReg devrait prédire positif pour ce texte
    assert data["sentiment"] == "positif"
    assert data["prob_positif"] > 0.5

@patch('api2.model')
def test_predict_sentiment_mocked_model(mock_model):
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

    response = client.post("/predict-sentiment", json={"text": "I love this app!"})

    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positif"
    assert abs(data["prob_positif"] - 0.9) < 0.01
    assert abs(data["prob_negatif"] - 0.1) < 0.01
    assert abs(data["confidence"] - 0.9) < 0.01

def test_predict_sentiment_negative():
    """Teste un texte négatif."""
    response = client.post("/predict-sentiment", json={"text": "This is terrible!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "négatif"
    assert data["prob_negatif"] > 0.5

def test_empty_text():
    """Test avec texte vide."""
    response = client.post("/predict-sentiment", json={"text": ""})
    assert response.status_code == 200  # Pas d'erreur 422, FastAPI gère
