import os
import pickle
import random
from functools import lru_cache
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DEFAULT_MODEL_PATH = os.path.abspath(
    os.environ.get("CHATBOT_MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "model.pkl"))
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    tag: str
    probability: float


def load_artifact(model_path: str) -> Dict[str, object]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")

    with open(model_path, "rb") as fh:
        return pickle.load(fh)


@lru_cache()
def get_model_components() -> Dict[str, object]:
    artifact = load_artifact(DEFAULT_MODEL_PATH)
    pipeline = artifact.get("pipeline")
    intents = artifact.get("intents")

    if pipeline is None or intents is None:
        raise RuntimeError("Model artifact is missing required keys 'pipeline' and 'intents'.")

    intent_lookup = {intent["tag"]: intent["responses"] for intent in intents}

    return {"pipeline": pipeline, "intents": intents, "intent_lookup": intent_lookup}


app = FastAPI(title="Chatbot API", version="1.0.0")


@app.get("/health")
def read_health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    components = get_model_components()
    pipeline = components["pipeline"]
    intent_lookup = components["intent_lookup"]

    probabilities = pipeline.predict_proba([user_message])[0]
    classes: List[str] = list(pipeline.classes_)

    best_index = int(probabilities.argmax())
    predicted_tag = classes[best_index]
    predicted_probability = float(probabilities[best_index])

    responses = intent_lookup.get(predicted_tag)
    if not responses:
        raise HTTPException(status_code=500, detail="No responses configured for predicted tag.")

    response_text = random.choice(responses)

    return ChatResponse(response=response_text, tag=predicted_tag, probability=predicted_probability)
