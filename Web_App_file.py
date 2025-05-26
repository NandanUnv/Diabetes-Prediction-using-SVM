import numpy as np
import pandas as pd

import pickle

data = pd.read_csv("diabetes.csv")

data.head()

x = data.iloc[:,:-1]

y = data.iloc[:,-1]


from sklearn.preprocessing import StandardScaler

scl = StandardScaler()

scl.fit(x)

x = scl.transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.80)

from sklearn import svm

clf = svm.SVC(kernel='linear')

clf.fit(x_train, y_train)


with open('diabetes.pkl' , 'wb') as f:
    pickle.dump(clf, f)


from flask import Flask, render_template, request
import pickle
from sklearn.svm import SVC


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("diabetes.html")


@app.route('/submit', methods=['POST'])
def sub():

    if request.method == "POST":
        preg = request.form['preg']
        gl = request.form['gl']
        bp = request.form['bp']
        st = request.form['st']
        ins = request.form['in']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']


        data = [[preg, gl, bp, st, ins, (float)(bmi), (float)(dpf), age]]
        std_data = scl.transform(data)
        model = pickle.load(open('diabetes.pkl', 'rb'))
        prediction = model.predict(std_data)[0]
        print(prediction)
    return render_template('diabetes.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
