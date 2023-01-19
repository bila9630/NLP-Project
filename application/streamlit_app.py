import streamlit as st
import random
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="The fast reporters",
    page_icon="📰",
)


# Load the tokenizer and model
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("model.pickle", "rb") as handle:
    model = pickle.load(handle)


def random_class(title, description):
    # Concatenate title and description to create a single input text
    text = title + " " + description

    # Convert the input text to numerical sequences
    input_string = [text]
    input_string = pad_sequences(
        tokenizer.texts_to_sequences(input_string), maxlen=140)

    # Predict the category using the input text
    predicted_category = model.predict(input_string)

    # Generate a random confidence value
    confidence = round(random.random(), 2)

    return predicted_category, confidence


st.header("Welcome to the fast reporters!")
st.write("Simply enter a news title and description and we'll classify it for you!")
# Create a selectbox widget for choosing a machine learning model
model_selectbox = st.selectbox(
    "Choose a machine learning model:",
    ["Decision Tree", "Random Forest", "SVM"]
)

news_title = st.text_input(label='Newspaper title',
                           placeholder='Enter your article title here')
news_description = st.text_area(label='Newspaper description',
                                placeholder='Enter a short description of the article here')


if st.button('Classify'):
    category, confidence = random_class(
        news_title, news_description)
    st.write("Model: ", model_selectbox)
    st.write(f"Category: {category}")
    st.write(f"Confidence: {confidence} (random for now)")
