# Spam Email Classification Project

A machine learning project to classify emails as spam or ham (safe) using a Logistic Regression model with Python and Scikit-learn. This project includes training, evaluation, and a real-time Streamlit web app for predictions.

## Features

* High Accuracy spam detection model
* Real-time Prediction capability through app.py
* Confusion Matrix & Classification Report for evaluation
* Easy to Train & Deploy using Python and Scikit-learn

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/spam-email-classification.git
cd spam-email-classification
```

2. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

**Run the Streamlit Web App:**

```
streamlit run app.py
```

**Train the Model (if needed):**

```
python Spam\ Email\ Classification.ipynb
```

**Predict on New Emails (Python script):**

```
python predict_email.py
```

## Model Overview

* Algorithm Used: Logistic Regression
* Preprocessing: Text vectorization using TF-IDF or Count Vectorizer
* Evaluation Metrics: Accuracy, Confusion Matrix, Classification Report

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load saved vectorizer and model
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('spam_model.pkl')

# Example prediction
sample_email = ["Congratulations! You won a prize."]
sample_vec = vectorizer.transform(sample_email)
prediction = model.predict(sample_vec)
print("Prediction:", prediction[0])
```

## Project Structure

```
.
├── Spam Email Classification.ipynb  # Jupyter notebook for training & exploration
├── app.py                           # Streamlit web application
├── spam_model.pkl                   # Trained Logistic Regression model
├── vectorizer.pkl                   # Saved TF-IDF vectorizer
├── requirements.txt                 # Dependencies
└── README.md                         # This file
```

## Future Improvements

* Add Deep Learning models like LSTM or BERT for better accuracy
* Enhance web interface with additional features for users
* Deploy as a REST API for external integration

## Live Demo

Check out the live demo here: [Streamlit Spam Classifier](https://spam-email-classification-ap.streamlit.app/)

## Contact Me

Connect with me on LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/muhammad-hassaan-munir-79b5b2327/)


