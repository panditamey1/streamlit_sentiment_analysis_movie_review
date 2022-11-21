import streamlit as st

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import numpy as np

# text preprocessing modules
from string import punctuation

# text preprocessing modules
from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import joblib

import warnings

warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)


stop_words = stopwords.words("english")

@st.cache
def text_cleaning(text, remove_stop_words= True, lemmatize_words=True):
    text = re.sub (r"[^A-Za-z0-9]"," ", text)
    text = re.sub(r"\'s"," ", text)
    text = re.sub(r"http\S+"," link ", text)
    text = re.sub(r"\b\d+(?:\.d+)?\s+", "", text)

    text = "".join([c for c in text if c not in punctuation])

    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word)  for word in text]
        text = " ".join(lemmatized_words)


    return text

@st.cache
def make_prediction(review):
    clean_review = text_cleaning(review)

    model = joblib.load("sentiment_model_pipeline.pkl")

    result = model.predict([clean_review])

    probas = model.predict_proba([clean_review])

    probability = "{:.2f}".format(float(probas[:, result]))

    return result, probability

st.title("Sentiment analysis app")
st.write("A simple machine learning app to predict the sentiment of a movie's review")

form = st.form(key = "my_form")
review = form.text_input(label="Enter the text of your movie review")
submit = form.form_submit_button(label="Make prediction")

if submit:
    result, probability = make_prediction(review)

    st.header("Results")

    if int(result) == 1:
        st.write("This is a positive review with a probability of ", probability)
    else:
        st.write("This is a negative review with a probability of ", probability)