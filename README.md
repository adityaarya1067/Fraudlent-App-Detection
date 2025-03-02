App Fraud Detection System
A machine learning and AI-powered system to detect potentially fraudulent applications on the Google Play Store. This tool combines traditional ML techniques with Google's Gemini AI to provide comprehensive fraud classification.
🌟 Features

Dual Analysis Approach: Combines machine learning model predictions with LLM-based analysis
Google Play Store Integration: Automatically fetches app metadata for analysis
Comprehensive Evaluation: Analyzes app descriptions, permissions, reviews, ratings, and install counts
Easy-to-Use Interface: Simple command-line interface for quick app analysis

📋 Requirements

Python 3.8+
Google Gemini API key

🔧 Installation

Clone this repository:
Copygit clone https://github.com/yourusername/app-fraud-detection.git
cd app-fraud-detection

Install required packages:
Copypip install -r requirements.txt

Create a .env file in the project root directory with your Gemini API key:
CopyGEMINI_API_KEY=your_gemini_api_key_here


📦 Dependencies

google-generativeai: For LLM-based classification using Gemini AI
google-play-scraper: For fetching app data from Google Play Store
scikit-learn: For machine learning model implementation
numpy: For numerical operations
joblib: For model serialization
python-dotenv: For environment variable management

🚀 Usage

Prepare your labeled datasets:

Create genuine-apps.json with details of known legitimate apps
Create fraud-apps.json with details of known fraudulent apps


Run the main script:
Copypython fraud_detection.py

Enter the App ID when prompted to analyze an app:
Copy🔹 Enter the App ID (or type 'quit' to exit): com.example.app

View the fraud detection report with both ML and LLM classifications.

🔄 Workflow

Data Collection: Fetches app metadata from Google Play Store
ML Analysis: Uses a Random Forest classifier trained on app descriptions
LLM Analysis: Uses Google's Gemini AI to analyze multiple app attributes
Combined Report: Generates a comprehensive fraud detection report

📊 Model Performance
The system evaluates model performance using accuracy and classification metrics:

Precision: How many identified frauds are actually fraudulent
Recall: How many actual fraudulent apps were correctly identified
F1-score: Harmonic mean of precision and recall

🛠️ Customization

Modify the train_fraud_detection_model function to use different ML algorithms
Adjust the Gemini AI prompt in analyze_fraud_with_llm to focus on specific fraud indicators
Add additional features to the ML model by modifying the feature extraction process

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgements

Google Generative AI for providing the Gemini API
The creators of google-play-scraper for making app data easily accessible
The scikit-learn team for their excellent machine learning library
