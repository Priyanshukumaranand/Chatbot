import os
import json
import pickle
from typing import List, Tuple

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

INTENTS_PATH = os.environ.get("CHATBOT_INTENTS_PATH", "intents.json")
MODEL_PATH = os.environ.get("CHATBOT_MODEL_PATH", "model.pkl")


def load_dataset(path: str) -> Tuple[List[str], List[str], list]:
    with open(path, "r", encoding="utf-8") as fp:
        intents = json.load(fp)

    patterns: List[str] = []
    tags: List[str] = []

    for intent in intents:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            tags.append(tag)

    return patterns, tags, intents


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            (
                "clf",
                LogisticRegression(
                    random_state=0,
                    max_iter=10000,
                ),
            ),
        ]
    )


def train() -> None:
    patterns, tags, intents = load_dataset(INTENTS_PATH)
    pipeline = build_pipeline()
    pipeline.fit(patterns, tags)

    predictions = pipeline.predict(patterns)
    accuracy = accuracy_score(tags, predictions)

    print(f"Training accuracy: {accuracy:.3f}")
    print("\nClassification report:\n")
    print(classification_report(tags, predictions))

    artifact = {
        "pipeline": pipeline,
        "intents": intents,
    }

    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(artifact, fh)

    print(f"\nSaved model artifact to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    train()
