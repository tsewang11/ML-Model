from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F

from model import load_model

model, device = load_model("model_weights.pth")


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_items=20, max_items=20)


class PredictResponse(BaseModel):
    predicted_class: int
    probabilities: list[float]


app = FastAPI(title="ML Model API", version="1.0.0")


@app.get("/")
def root():
    return {"message": "ML API is running. Use POST /predict to get predictions."}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = torch.tensor(req.features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

    predicted_class = int(probs.argmax())
    return PredictResponse(
        predicted_class=predicted_class,
        probabilities=[float(p) for p in probs],
    )
