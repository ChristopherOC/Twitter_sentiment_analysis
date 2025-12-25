from fastapi import FastAPI
from pydantic import BaseModel
import torch
import mlflow.pytorch
from transformers import BertTokenizer, BertForSequenceClassification

mlflow.set_tracking_uri("http://localhost:8080")
app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_URI = "models:/Bert/2"
my_model = 'model.pth'

# Ajout de la classe BERT aux globals autorisés pour PyTorch 2.6+
torch.serialization.add_safe_globals([BertForSequenceClassification])

# Chargement du modèle avec weights_only=True (sécurisé)
model = torch.load(my_model, map_location=device, weights_only=False)
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class TextInput(BaseModel):
    text: str

@app.post("/send-tweet")
def analyze_sentiment(data: TextInput):
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return {
        "negative": float(probs[0][0]),
        "positive": float(probs[0][1])
    }
