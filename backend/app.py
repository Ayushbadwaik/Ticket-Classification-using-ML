from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, os

app = Flask(__name__)
CORS(app)   # <--- enable CORS

# Load model & vectorizer
model = joblib.load(os.path.join(os.path.dirname(__file__), "model.pkl"))
vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), "vectorizer.pkl"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    ticket_text = data.get("ticket", "")
    if not ticket_text.strip():
        return jsonify({"error": "Empty ticket text"}), 400

    X = vectorizer.transform([ticket_text])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X).max()

    return jsonify({
        "ticket": ticket_text,
        "category": prediction,
        "confidence": round(float(proba) * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

