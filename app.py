import streamlit as st
import pickle
import os
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
ps = PorterStemmer()


# Get the absolute path to tfidf.pkl
# pkl_path = os.path.join(os.path.dirname(__file__), 'tfidf.pkl')
# pkl_path2 = os.path.join(os.path.dirname(__file__), 'model.pkl')
# # Load the TF-IDF vectorizer from the absolute path
# with open(pkl_path, 'rb') as file:
#     tfidf = pickle.load(file)
# with open(pkl_path2, 'rb') as file:
#     model = pickle.load(file)

tfidf= pickle.load(open('tfidf.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))
def func(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
       if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
            y.append(ps.stem(i))
    return " ".join(y)


st.title('SMS Classifier')
sms=st.text_input('Enter  SMS')
# 1. preprocess
sms=func(sms)
# 2.vectorize
input_sms = tfidf.transform([sms])
# 3.predict
p=model.predict(input_sms)[0]

if st.button('Predict'):
    if p==1:
        st.header('Spam')
    else:
        st.header('Not Spam')
