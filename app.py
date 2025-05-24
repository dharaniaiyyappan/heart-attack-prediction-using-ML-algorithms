from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the best trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Load dataset structure for preprocessing (Assuming 'Heart.csv' is available)
df = pd.read_csv('Heart.csv')
categorical_cols = df.select_dtypes(include=['object']).columns

# Function to preprocess input data
def preprocess_input(data):
    df_input = pd.DataFrame([data], columns=df.columns[:-1])  # Convert to DataFrame
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    return X_scaled

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('heart_attack.html')

@app.route('/heart-attack-prediction', methods=['POST'])
def predict():
    try:
        # Extract input values
        input_data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['chest-pain-type']),
            float(request.form['rbp']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['rest_ecg']),
            float(request.form['max-heart-rate']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slp']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        
        # Preprocess and predict
        features = preprocess_input(input_data)
        prediction = model.predict(features)[0]
        
        # Result message
        if prediction == 1:
            statement = "This model predicts that you WON'T suffer from a heart attack, but take care of your heart health."
        else:
            statement = "WARNING: This model predicts a HIGH risk of heart attack! Consult a doctor immediately."
        
        return render_template('heart_attack.html', statement=statement)
    
    except Exception as e:
        return render_template('heart_attack.html', statement=f"Error: {str(e)}")

# Run the Flask app
if __name__ == '__main__':
    app.run(port=4999, debug=True)
