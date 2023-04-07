from flask import Flask, request, jsonify
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
app = Flask(__name__)



model = pickle.load(open('spam_classifier.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))



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






@app.route('/predict', methods = ['POST'])
def predict():
    msg = request.form.get('msg')
    tranform_msg = text_transform(msg)
    result = tfidf.transform([tranform_msg])
    predict = model.predict(result)[0]

    if predict == 1:
        return jsonify('Spam Message')
    elif predict == 0:
        return jsonify('Not Spam')
    


    
    
if __name__ == '__main__':
    app.run(debug=True)

    



    

