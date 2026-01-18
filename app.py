import streamlit as st
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Smart Text Classifier",
    layout="centered"
)

st.title("Smart Text Classifier")
st.write("Classifies business queries into **Technical**, **Billing**, or **General**")

# Input text
user_input = st.text_area(
    "Enter your business query:",
    placeholder="e.g. Payment deducted but order failed"
)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([user_input])[0]
        st.success(f"**Predicted Category:** {prediction}")
