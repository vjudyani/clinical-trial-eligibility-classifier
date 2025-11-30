import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('eligibility_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App title
st.title("🏥 Clinical Trial Eligibility Classifier")
st.write("Enter a clinical trial criterion to classify it as Inclusion or Exclusion.")

# Input box
text = st.text_area("Enter criterion here:", height=100)

# Predict button
if st.button("🔍 Classify"):
    if text.strip() == "":
        st.warning("Please enter a criterion!")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        label = "Exclusion" if pred == 1 else "Inclusion"
        confidence = round(max(proba) * 100, 2)

        if label == "Inclusion":
            st.success(f"✅ {label} (Confidence: {confidence}%)")
        else:
            st.error(f"❌ {label} (Confidence: {confidence}%)")