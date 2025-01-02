from flask import Flask, render_template, request
import sqlite3
import pickle
import numpy as np

with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler_heart_disease_model.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

# Database 
DATABASE = 'heart_disease.db'

def create_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL,
            sex REAL,
            cp REAL,
            trestbps REAL,
            chol REAL,
            fbs REAL,
            restecg REAL,
            thalach REAL,
            exang REAL,
            oldpeak REAL,
            slope REAL,
            ca REAL,
            thal REAL,
            result TEXT,
            probability REAL
        )
    ''')
    conn.commit()
    conn.close()

create_table()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # data
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        input_data_scaled = scaler.transform(input_data)

        # prediction
        prediction = model.predict(input_data_scaled)
        prediction_prob = model.predict_proba(input_data_scaled)[:, 1]  # Probability for class 1 (Heart Disease)

        # Result 
        if prediction[0] == 1:
            result = 'Heart Disease'
            probability = prediction_prob[0]
        else:
            result = 'No Heart Disease'
            probability = 1 - prediction_prob[0]

        # Save to database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, result, probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, result, probability))
        conn.commit()
        conn.close()

        # Display the result
        final_result = f'Prediction: {result} (Probability: {probability:.2f})'
        return render_template('index.html', result=final_result)

    except Exception as e:
        return render_template('index.html', result=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
