from flask import Flask
from flask import Flask, render_template, request, redirect, url_for, session
from flask import render_template
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re

#################################################
# Flask Setup
#################################################
app = Flask(__name__)
#################################################
# Flask Routes
#################################################
stopwords =stopwords.words('english')

@app.route("/")
def index():
    return render_template('index.html')
	
@app.route('/homepage', methods =['GET', 'POST'])
def homepage():
    return render_template('index.html')

@app.route('/preprocessing', methods =['GET', 'POST'])
def preprocessing():
    return render_template('preprocessing.html')

	
@app.route('/classification', methods =['GET', 'POST'])
def classification():
    return render_template('classification.html')

@app.route('/predict', methods =['GET', 'POST'])
def predict():
    msg = ''
    if request.method == 'POST' and 'inputdata' in request.form :
        s1 = request.form['inputdata']
        df = pd.read_csv('BBC News Train.csv')
        labels=df.Category
        x_train,x_test,y_train,y_test=train_test_split(df['Text'], labels, test_size=0.2, random_state=7)
        tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
        tfidf_test=tfidf_vectorizer.transform(x_test)
        pac=PassiveAggressiveClassifier(max_iter=50)
        pac.fit(tfidf_train,y_train)
        tfidf_test1=tfidf_vectorizer.transform([s1])
        y_pred1=pac.predict(tfidf_test1)

        result5=str(y_pred1)
    return render_template('resultpage.html', msg = result5)

@app.route('/predict1', methods =['GET', 'POST'])
def predict1():
    msg = ''
    if request.method == 'POST' and 'inputdata' in request.form :
        s1 = request.form['inputdata']
        s2 = depure_data(s1)
    return render_template('resultpage1.html', msg = s1, msg1 = s2)

def depure_data(data):
    
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)
    
    data=re.sub("(\\d|\\W)+"," ",data)
    data = re.sub(r'[?|$|.|!]',r'',data)
    data = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", data)
    
    #data = re.sub('[^A-Za-z0-9]+', '', data)
        
    return data

if __name__ == "__main__":
    app.run(debug=True)