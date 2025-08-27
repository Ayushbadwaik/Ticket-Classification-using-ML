import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# ---- Example Training Data ----
data = {
    "ticket_text": [
        "I need help with my bill payment",
        "The product is not working",
        "How do I reset my password?",
        "I was overcharged on my last invoice",
        "App keeps crashing when I open it",
        "What are the product specifications?",
        "I cannot login to my account",
        "When will my order be delivered?",
        "My internet connection is very slow",
        "I want to upgrade my subscription"
    ],
    "category": [
        "Billing",
        "Technical Issue",
        "Technical Issue",
        "Billing",
        "Technical Issue",
        "Product Inquiry",
        "Technical Issue",
        "Product Inquiry",
        "Technical Issue",
        "Billing"
    ]
}

df = pd.DataFrame(data)

# Features and labels
X = df["ticket_text"]
y = df["category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved as model.pkl and vectorizer.pkl")

