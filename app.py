import json
import pickle
from flask import Flask, render_template,request, app,jsonify, url_for
import numpy as np
import pandas as pd

app=Flask(__name__)

# Load the model
bst_model=pickle.load(open("model.pkl",'rb'))

@app.route('/',methods=['GET'])

def home():
    return render_template('BAJAJ.html')

@app.route("/predict",methods=['POST'])

def predict():
    input_features=[int(x) for x in request.form.values()]
    data=[np.array(input_features)]
    output=bst_model.predict(data)
    def zen(x):
        if x==0:
            s='trad'
        elif x==1:
            s='ulip'
        else:
            s='term'
        return s

    Answer=zen(output)
    
    return render_template('BAJAJ.html',prediction_text="Suitable Insurance Product is {}".format(Answer))

if __name__=="__main__":
    app.run(debug=True)
