import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

app = Flask(__name__)
model_rdf = pickle.load(open('rdf.pkl','rb'))
model_xgb = pickle.load(open('xgb.pkl','rb'))
model_dtc = pickle.load(open('dtc.pkl','rb'))
model_grid = pickle.load(open('grid_search.pkl','rb'))
# scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction_text='Random Forest: {}',
                            xgBoost='xgBoost: {}',
                        desicion_tree='Desicion Tree: {}',
                        desicion_tree_grid='Desicion Tree + Grid Search: {}',  )

@app.route('/predict',methods=['POST'])
def predict():
    one = ['yes', 'present', 'good', 'normal', 'Yes', 'Present', 'Good', 'Normal', 'YES', 'PRESENT', 'GOOD', 'NORMAL']
    zero = ['no', 'notpresent', 'not present', 'poor', 'abnormal', 'No', 'Notpresent', 'NotPresent', 'Not Present', 'Poor', 'Abnormal', 'AbNormal', 'NO', 'NOTPRESENT', 'NOT PRESENT', 'POOR', 'ABNORMAL']
    int_features = []
    for i in request.form.values():
        if i in one:
            int_features.append(1.0)
        elif i in zero:
            int_features.append(0.0)
        else:
            int_features.append(float(i))
            
    final_features = [np.array(int_features)]

    # final_features = scaler.transform(final_features)
    # random forest
    prediction = model_rdf.predict(final_features)
    
    output1 = prediction
    output = ""

    if output1 == [0]:
        output = " Kidney Disease Not Detected"
    elif output1 == [1]:
        output = " Kidney Disease Detected"


    #xgBoost
    prediction = model_xgb.predict(final_features)
    
    output2 = prediction
    
    if output2 == [0]:
        output2 = "  Kidney Disease Not Detected"
    elif output2 == [1]:
        output2 = "  Kidney Disease Detected"

    #Desicion Tree
    prediction = model_dtc.predict(final_features)
    
    output3 = prediction
    
    if output3 == [0]:
        output3 = "  Kidney Disease Not Detected"
    elif output3 == [1]:
        output3 = "  Kidney Disease Detected"

    #Desicion Tree Grid Search

    prediction = model_grid.predict(final_features)
    
    output4 = prediction
    
    if output4 == [0]:
        output4 = "  Kidney Disease Not Detected"
    elif output4 == [1]:
        output4 = "  Kidney Disease Detected"


    

    
    return render_template('index.html', prediction_text='Random Forest: {}'.format(output),
                            xgBoost='xgBoost: {}'.format(output2),
                        desicion_tree='Desicion Tree: {}'.format(output3),
                        desicion_tree_grid='Desicion Tree + Grid Search: {}'.format(output4),  )

if __name__ == "__main__":
    app.run(debug=True)

    