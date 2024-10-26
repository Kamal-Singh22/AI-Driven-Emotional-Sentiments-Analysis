# Import necessary libraries
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st

# Load and preprocess dataset
file_path = 'emotion_dataset.csv'
df = pd.read_csv(file_path)
df = df[['content', 'sentiment']]

# Text preprocessing function
def preprocess_text(content):
    content = content.lower()
    content = re.sub(r'http\S+', '', content)
    content = re.sub(r'[^\w\s]', '', content)
    content = re.sub(r'\d+', '', content)
    return content

df['content'] = df['content'].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['sentiment'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=100)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Model performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Save the model and vectorizer
joblib.dump(model, 'emotion_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Load the saved model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit UI with custom icon and layout
st.set_page_config(page_title="Emotion Detection App", page_icon="📊", layout="centered")

# Custom CSS for a minimalistic dark-themed UI
st.markdown("""
    <style>
        .main { background-color: #1e1e1e; color: #f0f0f0; }
        .header { text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.3em; color: #f0f0f0; }
        .subheader { text-align: center; font-size: 1.2em; margin-bottom: 1.5em; color: #cccccc; }
        .predict-btn { background-color: #333333; color: #ffffff; border-radius: 5px; padding: 8px 20px; margin-top: 20px; }
        .emotion { color: #d4d4d4; font-size: 1.2em; font-weight: bold; text-align: center; margin-top: 20px; }
        .sidebar-text { color: #ffffff; font-size: 1.1em; font-weight: bold; }
        .footer { text-align: center; color: #888888; margin-top: 50px; font-size: 0.9em; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for performance metrics
st.sidebar.title("Model Performance")
st.sidebar.markdown("<div class='sidebar-text'>Toggle between metrics:</div>", unsafe_allow_html=True)

# Add slider to toggle between classification report and confusion matrix
selected_metric = st.sidebar.radio("Choose a metric:", ('Classification Report', 'Confusion Matrix'))

# Display selected metric in the sidebar
if selected_metric == 'Classification Report':
    st.sidebar.text("Classification Report:")
    st.sidebar.text(class_report)
else:
    st.sidebar.text("Confusion Matrix:")
    st.sidebar.write(conf_matrix)

# Main title and description
st.markdown("<div class='header'>Emotion Detection Using AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Identify emotions in text such as joy, sadness, and anger.</div>", unsafe_allow_html=True)

# Text input area
user_input = st.text_area("Enter the text to analyze:", placeholder="Type your text here...")

# Prediction button
if st.button("Predict Emotion"):
    if user_input:
        preprocessed_text = preprocess_text(user_input)
        input_vec = vectorizer.transform([preprocessed_text])
        prediction = model.predict(input_vec)[0]
        st.markdown(f"<div class='emotion'>Predicted Emotion: {prediction}</div>", unsafe_allow_html=True)

# Display model accuracy in the sidebar
st.sidebar.write(f"**Model Accuracy:** {accuracy:.2%}")

# Footer with creator information
st.markdown("<div class='footer'>Created by Kamal Singh</div>", unsafe_allow_html=True)
