import streamlit as st
import random
st.set_page_config(
    page_title="The fast reporters",
    page_icon="📰",
)


def random_class(title, description, model):
    # pass title and description to model
    # category = model.predict(title, description)
    category = random.choice(
        ["U.S. NEWS", "COMEDY", "WORLD NEWS", "TECH", "ENVIRONMENT"])
    confidence = round(random.random(), 2)
    return category, confidence


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
        news_title, news_description, model_selectbox)
    st.write("Model: ", model_selectbox)
    st.write(f"Category: {category}")
    st.write(f"Confidence: {confidence}")
