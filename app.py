from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
ps=PorterStemmer()

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punkt')
sw=nltk.corpus.stopwords.words('english')


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')
@app.route('/top')
def top():
    return render_template('top.html')

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    c=[]
    for i in text:
        if i.isalnum():
            c.append(i)
    text=c[:]
    c=[]
    c=[ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(c)

#Sentiment Analysis

tfidf=TfidfVectorizer(stop_words=sw,max_features=20)
def transform(txt):
    textnew=tfidf.fit_transform(txt)
    return textnew.toarray()
df=pd.read_csv("./new_file.csv")
df.columns=["Text","Label"]
x=transform(df["Text"])
y=df["Label"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)
model=LogisticRegression()
model.fit(x_train,y_train)





@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    transformed_txt=transform_text(text)
    #tfidf
    vtxt=tfidf.transform([transformed_txt])
    prediction=model.predict(vtxt)[0]

    return render_template('result.html', text=text, sentiment=prediction)



if __name__ == '__main__':
    app.run(debug=True)
