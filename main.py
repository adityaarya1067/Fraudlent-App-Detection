import os
import json
import google.generativeai as genai
import numpy as np
import joblib
from google_play_scraper import app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv
from typing import Dict, List

# ✅ Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# ✅ File paths
genuine_apps_path = "genuine-apps.json"
fraud_apps_path = "fraud-apps.json"
model_path = "fraud_detection_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

# ✅ 1. Fetch App Details from Google Play Store
def fetch_app_details(app_id: str) -> Dict:
    """Fetches app metadata from Google Play Store"""
    try:
        result = app(app_id, lang="en", country="us")
        return {
            "app_id": app_id,
            "title": result["title"],
            "description": result["description"],
            "developer": result["developer"],
            "permissions": result.get("permissions", []),
            "reviews": result.get("comments", []),
            "ratings": result["score"],
            "installs": result["installs"]
        }
    except Exception as e:
        print(f"❌ Error fetching data for {app_id}: {e}")
        return None

# ✅ 2. Load JSON Data
def load_json(file_path: str) -> List[Dict]:
    """Loads JSON data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []

# ✅ 3. Train Fraud Detection Model (ML)
def train_fraud_detection_model(genuine_data: List[Dict], fraud_data: List[Dict]):
    """Trains a fraud detection model using app descriptions and evaluates it."""
    descriptions = [app.get("description", "") for app in genuine_data] + [app.get("description", "") for app in fraud_data]
    labels = [0] * len(genuine_data) + [1] * len(fraud_data)  # 0 = Genuine, 1 = Fraud
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(descriptions)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, labels)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("✅ Fraud detection model trained and saved.")

    # Model Evaluation
    predictions = model.predict(X)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=["Genuine", "Fraud"])

    print(f"📊 Model Accuracy: {accuracy:.4f}")
    print("📌 Classification Report:")
    print(report)

# ✅ 4. LLM Analysis (Google Gemini AI)
def analyze_fraud_with_llm(app_data: Dict) -> Dict:
    """Uses Gemini AI to classify an app as 'fraud', 'genuine', or 'suspected'."""
    prompt = f"""
    Analyze the following app details and classify it as 'fraud', 'genuine', or 'suspected'.
    Provide a short reason (max 300 chars) for the classification.

    App Title: {app_data.get('title', 'Unknown')}
    Description: {app_data.get('description', '')}
    Developer: {app_data.get('developer', 'Unknown')}
    Permissions: {', '.join(app_data.get('permissions', []))}
    Reviews: {', '.join(app_data.get('reviews', [])[:5])}
    Ratings: {app_data.get('ratings', 0)}
    Installs: {app_data.get('installs', '0')}
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return {"app_id": app_data.get("app_id", "Unknown"), "classification": response.text}

# ✅ 5. Predict Fraud Using ML Model
def predict_fraud_with_ml(app_data: Dict) -> str:
    """Predicts fraud probability using the trained ML model."""
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return "Model not trained yet."

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    X = vectorizer.transform([app_data.get("description", "")])
    prediction = model.predict(X)[0]
    return "genuine" if prediction == 0 else "fraud"

# ✅ 6. Generate Fraud Report
def generate_fraud_report(app_data: Dict) -> Dict:
    """Generates fraud classification using both ML and LLM models."""
    llm_result = analyze_fraud_with_llm(app_data)
    ml_prediction = predict_fraud_with_ml(app_data)

    return {
        "app_id": app_data["app_id"],
        "llm_classification": llm_result,
        "ml_classification": ml_prediction
    }

# ✅ 7. Model Testing & Validation
def evaluate_model(test_data: List[Dict], test_labels: List[int]):
    """Evaluates the trained model using accuracy and classification metrics."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    X_test = vectorizer.transform([app.get("description", "") for app in test_data])
    predictions = model.predict(X_test)

    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, target_names=["Genuine", "Fraud"])

    print(f"📊 Test Accuracy: {accuracy:.4f}")
    print("📌 Test Classification Report:")
    print(report)

# ✅ Main Execution
if __name__ == "__main__":
    # Load labeled datasets
    genuine_data = load_json(genuine_apps_path)
    fraud_data = load_json(fraud_apps_path)

    # Train model if not already trained
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("🚀 Training fraud detection model...")
        train_fraud_detection_model(genuine_data, fraud_data)

    # ✅ Keep asking for App ID until user quits
    while True:
        test_app_id = input("\n🔹 Enter the App ID (or type 'quit' to exit): ").strip()

        if test_app_id.lower() == "quit":
            print("👋 Exiting the program. Have a great day!")
            break

        test_app_data = fetch_app_details(test_app_id)

        if test_app_data:
            # Generate Fraud Report for the entered App
            fraud_report = generate_fraud_report(test_app_data)
            print("\n🔍 Fraud Detection Report:")
            print(json.dumps(fraud_report, indent=4))
        else:
            print("⚠️ App not found. Please check the App ID and try again.")

    # Run model validation with a subset of test data
    test_data = genuine_data[:10] + fraud_data[:10]  # Sample 10 genuine and 10 fraud apps
    test_labels = [0] * 10 + [1] * 10
    evaluate_model(test_data, test_labels)
