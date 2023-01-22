import streamlit as st
import re
import random
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")

st.set_page_config(
    page_title="The fast reporters",
    page_icon="📰",
)


# import model for deployment
model = pickle.load(open("application/naiveBayes.pkl", "rb"))

# import model on local machine
# model = pickle.load(open("naiveBayes.pkl", "rb"))


# Preprocess text
def process_text(text):
    # convert text to lowercase, remove newlines and carriage returns, and strip leading/trailing whitespace
    text = text.lower().replace('\n', ' ').replace('\r', '').strip()

    # replace multiple spaces with single space
    text = re.sub(' +', ' ', text)

    # remove non-alphanumeric characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[0-9]', '', text)

    # create set of english stopwords
    stop_words = set(stopwords.words('english'))

    # tokenize text into words
    word_tokens = word_tokenize(text)

    # if word not in stops_words, add word to filtered_sentence
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    text = " ".join(filtered_sentence)
    return text


def predict_category(title, description):
    # Concatenate title and description to create a single input text
    text = title + " " + description
    text_processed = process_text(text)

    # Predict the category using the input text
    predicted_category = model.predict([text_processed])
    # Convert the predicted category to a string
    predicted_category = predicted_category[0]

    # Generate a random confidence value
    confidence = round(random.random(), 2)

    return predicted_category, confidence


st.header("Welcome to the fast reporters!")
st.write("Simply enter a news title and description and we'll classify it for you!")
# Create a selectbox widget for choosing a machine learning model
model_selectbox = st.selectbox(
    "Choose a machine learning model:",
    ["Naives Bayes", "Random Forest", "Linear SVM"]
)

news_title = st.text_input(label='Newspaper title',
                           placeholder='Enter your article title here')
news_description = st.text_area(label='Newspaper description',
                                placeholder='Enter a short description of the article here')


if st.button('Classify'):
    category, confidence = predict_category(news_title, news_description)
    st.write("Model: ", model_selectbox)
    st.write(f"Category: {category}")
    st.write(f"Confidence: {confidence} (random for now)")
