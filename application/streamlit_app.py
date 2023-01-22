import streamlit as st
import re
import random
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# set page config (must be called as the first Streamlit command)
st.set_page_config(
    page_title="The fast reporters",
    page_icon="📚",
)


# caching nltk data
@st.cache(allow_output_mutation=True)
def download_nltk():
    nltk.download("stopwords")
    nltk.download("punkt")

download_nltk()


# import model for deployment
# load model with cache
@st.cache(allow_output_mutation=True)
def load_model():
    model_nb = pickle.load(open("application/naiveBayes.pkl", "rb"))
    model_svm = pickle.load(open("application/linearSVM.pkl", "rb"))
    return model_nb, model_svm

model_nb, model_svm = load_model()


# import model on local machine
# model_nb = pickle.load(open("naiveBayes.pkl", "rb"))
# model_svm = pickle.load(open("linearSVM.pkl", "rb"))


# Preprocess text function
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

    # Predict the category with both models and store the predictions
    predictions = {
        'naive_bayes': model_nb.predict([text_processed])[0],
        'svm': model_svm.predict([text_processed])[0]
    }

    # Generate a random confidence value
    confidence = round(random.random(), 2)

    return predictions, confidence


st.header("Welcome to the fast reporters!")
st.write("Simply enter a news title and description and we'll classify it for you!")
# # Create a selectbox widget for choosing a machine learning model
# model_selectbox = st.selectbox(
#     "Choose a machine learning model:",
#     ["Naives Bayes", "Random Forest", "Linear SVM"]
# )

news_title = st.text_input(label='Newspaper title',
                           placeholder='Enter your article title here')
news_description = st.text_area(label='Newspaper description',
                                placeholder='Enter a short description of the article here')


if st.button('Classify'):
    predictions, confidence = predict_category(news_title, news_description)
    # st.write("Model: ", model_selectbox)
    st.write(f"Category Naives Bayes: {predictions['naive_bayes']}")
    st.write(f"Category SVM: {predictions['svm']}")
    st.write(f"Confidence: {confidence} (random for now)")
