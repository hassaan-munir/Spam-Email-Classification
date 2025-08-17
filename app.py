import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Email Spam Classifier")

# User input
user_input = st.text_area("Paste your email text here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Transform input
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.error("🚨 This email looks like SPAM!")
        else:
            st.success("✅ This email looks SAFE (Not Spam).")
