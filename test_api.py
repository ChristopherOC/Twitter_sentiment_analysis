import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import torch
import math
from api import app

client = TestClient(app)

class MockBatchEncoding:
    """Mock parfait du retour de tokenizer()"""
    def __init__(self):
        self.input_ids = torch.tensor([[1,2,3]])
        self.attention_mask = torch.tensor([[1,1,1]])
    
    def to(self, device):
        return {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask
        }

@pytest.mark.parametrize("text,logits", [
    ("I love this app!", torch.tensor([[0.1, 0.9]])),  # Positive → [neg, pos]
    ("This is terrible!", torch.tensor([[0.9, 0.1]])), # Negative → [neg, pos]
])
@patch('api.model')
@patch('api.tokenizer')
@patch('api.device')
def test_analyzesentiment(mock_device, mock_tokenizer, mock_model, text, logits):
    # Mock tokenizer
    mock_tokenizer.return_value = MockBatchEncoding()
    
    # Mock model avec logits spécifiques au texte
    mock_outputs = MagicMock()
    mock_outputs.logits = logits
    mock_model.return_value = mock_outputs
    
    # Mock device
    mock_device.__str__ = lambda self: 'cpu'
    
    response = client.post("/send-tweet", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    
    # Vérifie que softmax fonctionne: positive = exp(logits[1]) / sum(exp)
    positive_raw = math.exp(logits[0][1].item())
    total_raw = math.exp(logits[0][0].item()) + math.exp(logits[0][1].item())
    expected_positive = positive_raw / total_raw
    
    assert abs(data["positive"] - expected_positive) < 0.01
    assert abs(data["negative"] - (1 - expected_positive)) < 0.01

@patch('api.torch.load')
@patch('api.torch.cuda.is_available', return_value=False)
def test_model_loading(mock_cuda, mock_torch_load):
    mock_torch_load.return_value = MagicMock()
    response = client.post("/send-tweet", json={"text": "test"})
    assert response.status_code == 200

def test_valid_input():
    response = client.post("/send-tweet", json={"text": "I like it"})
    assert response.status_code == 200
    data = response.json()
    assert "positive" in data and "negative" in data

def test_empty_text():
    response = client.post("/send-tweet", json={"text": ""})
    assert response.status_code == 200
