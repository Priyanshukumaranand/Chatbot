import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load the data
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

#create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state =0 , max_iter=10000)

#preprocess the data
import os
import csv
import datetime
from typing import Optional

import requests
import streamlit as st

API_URL = os.environ.get("CHATBOT_API_URL", "http://localhost:8000/chat")
LOG_HEADER = ["User Input", "Chatbot Response", "Tag", "Probability", "Timestamp"]


def ensure_log_file(path: str) -> None:
    if os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            existing_header = next(reader, [])
            rows = list(reader)

        if existing_header == LOG_HEADER:
            return

        padded_rows = []
        for row in rows:
            padded = row + [""] * max(0, len(LOG_HEADER) - len(row))
            padded_rows.append(padded[: len(LOG_HEADER)])

        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(LOG_HEADER)
            writer.writerows(padded_rows)
        return

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(LOG_HEADER)


def call_backend(message: str) -> Optional[dict]:
    try:
        response = requests.post(API_URL, json={"message": message}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Backend request failed: {exc}")
        return None


def log_interaction(path: str, user_message: str, bot_response: str, tag: Optional[str], probability: Optional[float]) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([user_message, bot_response, tag or "", f"{probability:.4f}" if probability is not None else "", timestamp])


def render_home(log_path: str) -> None:
    st.subheader("Chat")
    st.write("Send a message to the chatbot. The backend FastAPI service handles the intent classification.")

    user_input = st.text_input("You:", key="chat_input")

    if st.button("Send") and user_input:
        user_input_str = user_input.strip()
        result = call_backend(user_input_str)
        if result is None:
            return

        response_text = result.get("response", "I couldn't understand that.")
        tag = result.get("tag")
        probability = result.get("probability")

        tag_display = tag or "unknown"
        if isinstance(probability, (int, float)):
            confidence_display = f"{probability:.2%}"
        else:
            probability = None
            confidence_display = "n/a"

        st.text_area(
            "Chatbot:",
            value=f"{response_text}\n\n(Intent: {tag_display}, Confidence: {confidence_display})",
            height=150,
        )

        log_interaction(log_path, user_input_str, response_text, tag, probability)


def render_history(log_path: str) -> None:
    st.subheader("Conversation History")
    if not os.path.exists(log_path):
        st.info("No conversations logged yet.")
        return

    with open(log_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            user_msg, bot_resp, tag, probability, timestamp = row
            probability_display = probability or "n/a"
            tag_display = tag or "unknown"
            st.text(f"User: {user_msg}")
            st.text(f"Chatbot: {bot_resp}")
            st.text(f"Tag: {tag_display} | Confidence: {probability_display}")
            st.text(f"Timestamp: {timestamp}")
            st.markdown("---")


def main():
    st.title("Chatbot (Streamlit Frontend)")

    log_path = os.environ.get("CHATBOT_LOG_PATH", "chat_log.csv")
    ensure_log_file(log_path)

    menu = ["Home", "Conversation History"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        render_home(log_path)
    elif choice == "Conversation History":
        render_history(log_path)


if __name__ == "__main__":
    main()