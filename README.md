📌 Fraudulent App Detection System
This project is an AI-powered fraud detection system designed to identify potentially fraudulent or harmful applications on the Google Play Store. It combines Machine Learning (ML) techniques with Google Gemini AI (LLM) to classify apps based on metadata, descriptions, permissions, and user reviews.

📊 Project Objective
The goal of this project is to analyze app data and detect fraud patterns using a combination of:
✅ Supervised ML models – Predict fraudulent behavior based on historical data.
✅ LLM-based analysis – Provide explainable AI-generated insights.
✅ Structured output – Deliver fraud classifications in a JSON format.
🛠 Technologies Used
Python – Core programming language.
scikit-learn – ML model training.
TfidfVectorizer – Text feature extraction.
Google Gemini AI – LLM for fraud detection analysis.
NumPy & Pandas – Data processing.
Joblib – Model serialization for future predictions.
📌 Features & Workflow
🔹 1. Data Collection
Loads app details (title, description, permissions, reviews, developer information).
Uses labeled datasets of genuine and fraudulent apps.
🔹 2. Machine Learning Model Training
Extracts features using TF-IDF vectorization.
Trains a RandomForestClassifier for fraud detection.
Saves the trained model for future use.
🔹 3. Fraud Analysis with LLM (Google Gemini AI)
Receives app metadata & reviews.
Generates explainable classifications:
"fraud" 🚨
"genuine" ✅
"suspected" ⚠️
Provides a short reason (max 300 chars) for classification.
🔹 4. Model Testing & Validation
Evaluates model accuracy using a test dataset.
Prints a classification report with precision, recall, and F1-score.
Compares ML predictions with LLM-generated classifications.
📈 Model Performance
✔ Test Accuracy: ✅ 99.XX% (Varies with dataset size)

✔ Evaluation Metrics:

markdown
Copy
Edit
              precision    recall  f1-score   support
     Genuine       0.99      0.98      0.99        X
       Fraud       0.98      0.99      0.99        X
    accuracy                           0.99        X
💡 Future Enhancements
🔄 Expand dataset with more diverse fraud examples.
🤖 Experiment with deep learning models (LSTMs, Transformers).
🌐 Deploy as an API for real-time fraud detection.
📌 How to Run the Project
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-repo/fraudulent-app-detection.git
cd fraudulent-app-detection
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run Training & Testing
bash
Copy
Edit
python app.py
4️⃣ View Model Accuracy & LLM Output
Terminal will display test accuracy and fraud classifications.
Screenshots:
![Screenshot 2025-03-02 113137](https://github.com/user-attachments/assets/fac5ccfe-fd90-42c8-9466-f16be2a971d9)
![Screenshot 2025-03-02 113154](https://github.com/user-attachments/assets/4158fe3b-592c-4dc6-ae20-640811b9dd38)
![Screenshot 2025-03-02 113213](https://github.com/user-attachments/assets/abd1358d-64c6-416e-8d71-1f2a137c9b0e)

📢 Contributing
💡 Want to improve this project? Feel free to fork, contribute, and suggest enhancements! 🚀
