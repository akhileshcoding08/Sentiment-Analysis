import re
import pickle
import os
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load the trained model artifact (vectorizer + classifier) once at startup
# ---------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

vectorizer = artifact["vectorizer"]
model = artifact["model"]


def clean_text(text: str) -> str:
    """Same cleaning function used during training — must match exactly
    so new input is preprocessed the same way as the training data."""
    text = re.sub(r"<.*?>", " ", str(text))          # strip HTML tags (e.g. <br />)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)          # keep only letters
    text = re.sub(r"\s+", " ", text).strip().lower()  # collapse whitespace, lowercase
    return text


def predict_sentiment(text: str):
    """Return (label, confidence_percent) for a given piece of text."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]           # [prob_negative, prob_positive]
    prediction = model.predict(vec)[0]              # 0 = Negative, 1 = Positive
    label = "Positive" if prediction == 1 else "Negative"
    confidence = round(max(proba) * 100, 2)
    return label, confidence


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review_text", "").strip()
        if review_text:
            result, confidence = predict_sentiment(review_text)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        review_text=review_text,
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    """JSON API endpoint: POST {"text": "..."} -> {"sentiment": "...", "confidence": ...}"""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return {"error": "No text provided"}, 400

    label, confidence = predict_sentiment(text)
    return {"sentiment": label, "confidence": confidence}


if __name__ == "__main__":
    app.run(debug=True)
