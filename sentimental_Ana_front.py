import streamlit as st
import pickle
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Load pre-trained model and vectorizer
model = pickle.load(open("sentiment_analysis_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))  # Load the fitted vectorizer

# Initialize the stemmer
port_stem = PorterStemmer()

st.title("Sentiment Analyser")

# Function for text preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]  # Stemming & stopword removal
    return ' '.join(text)

# User input
user_input = st.text_area("Enter the text or message you want to check")

if st.button("Check Sentiment"):
    try:
        processed_input = preprocess_text(user_input)
        input_vectorized = vectorizer.transform([processed_input])  # Use the fitted vectorizer
        result = model.predict(input_vectorized)

        # Display result
        if result[0] == 0:
            st.error("Bad Review")
        else:
            st.success("Good Review")

    except Exception as e:
        st.error(f"Error: {e}")