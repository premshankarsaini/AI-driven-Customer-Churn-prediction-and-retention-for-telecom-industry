import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template('home.html')

@app.route("/", methods=['POST'])
def predict():
    Dependents = request.form['Dependents']
    tenure = float(request.form['tenure'])
    OnlineSecurity = request.form['OnlineSecurity']
    OnlineBackup = request.form['OnlineBackup']
    DeviceProtection = request.form['DeviceProtection']
    TechSupport = request.form['TechSupport']
    Contract = request.form['Contract']
    PaperlessBilling = request.form['PaperlessBilling']
    MonthlyCharges = float(request.form['MonthlyCharges'])
    TotalCharges = float(request.form['TotalCharges'])

    # Load trained model
    model = pickle.load(open('Model.sav', 'rb'))

    # Encode manually (same as model training)
    mapping = {'Yes': 1, 'No': 0, 
               'Month-to-month': 0, 'One year': 1, 'Two year': 2}

    data = [[Dependents, tenure, OnlineSecurity, OnlineBackup, DeviceProtection,
             TechSupport, Contract, PaperlessBilling, MonthlyCharges, TotalCharges]]
    
    df = pd.DataFrame(data, columns=['Dependents', 'tenure', 'OnlineSecurity',
                                     'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                     'Contract', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'])

    # Apply same encoding
    for col in ['Dependents', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'PaperlessBilling', 'Contract']:
        df[col] = df[col].map(mapping).fillna(0)

    single = model.predict(df)
    probability = model.predict_proba(df)[:, 1] * 100

    if single[0] == 1:
        op1 = "This Customer is likely to be Churned!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}%"
    else:
        op1 = "This Customer is likely to Continue!"
        op2 = f"Confidence level is {np.round(100 - probability[0], 2)}%"

    return render_template("home.html", op1=op1, op2=op2)

if __name__ == '__main__':
    app.run(debug=True)
