import streamlit as st
import re
import random
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.models import load_model

# set page config (must be called as the first Streamlit command)
st.set_page_config(
    page_title="The fast reporters",
    page_icon=":newspaper:",
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
def load_model_path():
    model_nb = pickle.load(open("application/naiveBayes.pkl", "rb"))
    model_svm = pickle.load(open("application/linearSVM.pkl", "rb"))
    return model_nb, model_svm


model_nb, model_svm = load_model_path()


# import model on local machine
# model_nb = pickle.load(open("naiveBayes.pkl", "rb"))
# model_svm = pickle.load(open("linearSVM.pkl", "rb"))

# Load the saved model nn and tokenizer
@st.cache(allow_output_mutation=True)
def load_saved_model_nn(model_path, tokenizer_path, encoder_path):
    # Load the model from the file
    model = load_model(model_path)

    # Load the tokenizer from the file
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)

    # Load the encoder from the file
    with open(encoder_path, "rb") as handle:
        encoder = pickle.load(handle)

    return model, tokenizer, encoder

# Load the model and tokenizer for deployment
model_nn, tokenizer_nn, encoder_nn = load_saved_model_nn("application/model_nn.h5", "application/tokenizer_nn.pickle", "application/encoder_nn.pickle")

# Load the model and tokenizer on local machine
# model_nn, tokenizer_nn, encoder_nn = load_saved_model_nn("model_nn.h5", "tokenizer_nn.pickle", "encoder_nn.pickle")



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

    # Predict the category with the neural network model
    text_matrix_nn = tokenizer_nn.texts_to_matrix([text_processed])
    predictions_nn = model_nn.predict(text_matrix_nn)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions_nn)
    # Get the corresponding class label from the label encoder
    predicted_class_label = encoder_nn.classes_[predicted_class_index]

    # Get the predicted class probability
    predicted_class_prob = predictions_nn[0][predicted_class_index]


    # Predict the category with both models and store the predictions
    predictions = {
        "naive_bayes": model_nb.predict([text_processed])[0],
        "svm": model_svm.predict([text_processed])[0],
        "nn": predicted_class_label,
        "nn_prob": predicted_class_prob
    }

    return predictions


st.header("Welcome to the fast reporters!")
st.write("Simply enter a news title and description and we'll classify it for you!")

news_title = st.text_input(label='Newspaper title',
                           placeholder='Enter your article title here')
news_description = st.text_area(label='Newspaper description',
                                placeholder='Enter a short description of the article here')


if st.button('Classify'):
    predictions = predict_category(news_title, news_description)
    st.write(f"Category Naives Bayes: {predictions['naive_bayes']}")
    st.write(f"Category SVM: {predictions['svm']}")
    st.write(f"Category NN: {predictions['nn']}, with probability: {predictions['nn_prob']:.2f}")
