from fastapi import FastAPI
from pydantic import BaseModel, Field 
import torch
import torch.nn.functional as F

from model import load_model




model=load_model("model_weights.pth")
device = torch.device("mps")
app = FastAPI(title="ML Model API", version="1.0.0")

#Requet and response schemas
class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., 
        description ="Input features for the model. For now, expected length is 20.",
        min_items=20,
        max_items=20,
    )
    
class PredictResponse (BaseModel):
    predicted_class: int
    probabilities: list[float]
    
    


# Root endpoint : just to test the server
@app.get("/")
def read_root():
    return {"message": "ML API is running. Use POST /predict to get predictions."}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Convert list to tensor of shape (1, num_features)
    x = torch.tensor(request.features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

    predicted_class = int(probs.argmax())
    return PredictResponse(
        predicted_class=predicted_class,
        probabilities=[float(p) for p in probs],
    )