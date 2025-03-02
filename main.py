import json
import os
import google.generativeai as genai
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Define file paths
genuine_apps_path = "./genuine-apps.json"
fraud_apps_path = "./fraud-apps.json"
model_path = "./fraud_detection_model.pkl"
vectorizer_path = "./tfidf_vectorizer.pkl"

def load_json(file_path: str) -> List[Dict]:
    """Loads JSON data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def train_fraud_detection_model(genuine_data: List[Dict], fraud_data: List[Dict]):
    """Trains a fraud detection model using app descriptions and evaluates it."""
    descriptions = [app.get("description", "") for app in genuine_data] + [app.get("description", "") for app in fraud_data]
    labels = [0] * len(genuine_data) + [1] * len(fraud_data)  # 0 = genuine, 1 = fraud
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(descriptions)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, labels)
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Fraud detection model trained and saved.")
    
    # Model Evaluation
    predictions = model.predict(X)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=["Genuine", "Fraud"])
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

def analyze_fraud_with_llm(app_data: Dict) -> Dict:
    """Uses Gemini API to classify an app as fraudulent, suspected, or genuine."""
    prompt = f"""
    Analyze the following app details and classify it as 'fraud', 'genuine', or 'suspected'.
    Provide a short reason (max 300 chars) for the classification.
    
    App Title: {app_data.get('title', 'Unknown')}
    Description: {app_data.get('description', '')}
    Developer: {app_data.get('developer', 'Unknown')}
    Permissions: {', '.join(app_data.get('permissions', []))}
    Reviews: {', '.join(app_data.get('reviews', [])[:5])}  # Limit to 5 reviews
    Ratings: {app_data.get('ratings', 0)}
    Installs: {app_data.get('installs', '0')}
    """
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    analysis = response.text.strip()
    return {"app_id": app_data.get("app_id", "Unknown"), "classification": analysis}

def predict_fraud_with_ml(app_data: Dict) -> str:
    """Predicts fraud probability using the trained ML model."""
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return "Model not trained yet."
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    X = vectorizer.transform([app_data.get("description", "")])
    prediction = model.predict(X)[0]
    return "fraud" if prediction == 1 else "genuine"

if __name__ == "__main__":
    genuine_data = load_json(genuine_apps_path)
    fraud_data = load_json(fraud_apps_path)
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Training fraud detection model...")
        train_fraud_detection_model(genuine_data, fraud_data)
    
    # Testing & Validation
    print("\nTesting & Validation\n----------------------")
    test_data = genuine_data[:10] + fraud_data[:10]  # Sample 10 genuine and 10 fraud apps for testing
    test_labels = [0] * 10 + [1] * 10
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    X_test = vectorizer.transform([app.get("description", "") for app in test_data])
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, target_names=["Genuine", "Fraud"])
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Test Classification Report:")
    print(report)
    
    # LLM Output
    print("\nLLM Output\n-----------")
    for app in test_data[:5]:  # Sample 5 apps for LLM analysis
        llm_result = analyze_fraud_with_llm(app)
        print(f"App ID: {llm_result['app_id']}, Classification: {llm_result['classification']}")
