import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()


model = pickle.load(open('spam_classifier.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

st.title('Spam Classifier')

def text_transform(text):
    x = []
    text = text.lower()
    for i in nltk.word_tokenize(text):
        if i.isalnum():
            x.append(i)
    text = x[:]
    x.clear()

    for i in text:
        if i not in stopwords.words('english'):
            x.append(i)
    text = x[:]
    x.clear()
    for i in text:
        x.append(ps.stem(i))
    text = x[:]
    x.clear()
    return ' '.join(text)

msg = st.text_area("Enter Your Message")


if st.button("predict"):

  tranform_msg = text_transform(msg)
  result = tfidf.transform([tranform_msg])
  predict = model.predict(result)[0]

  if predict == 1:
    st.header("Spam Message")
  else:
    st.header("Not Spam Message")

