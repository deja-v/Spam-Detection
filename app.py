from flask import Flask,request,jsonify,render_template


app = Flask(__name__)

import pickle
import string
import nltk
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get("a")
    # 1. preprocess
    transformed_sms = transform_text(data)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display

    if result == 1:
        return render_template('index.html', prediction_text='It is a Spam!')
    else:
        return render_template('index.html', prediction_text='Not Spam')


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')