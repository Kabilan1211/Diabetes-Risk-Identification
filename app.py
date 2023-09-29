from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('dia.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    Pregnancies = request.form['a']
    Glucose = request.form['b']
    BloodPressure = request.form['c']
    SkinThickness = request.form['d']
    Insulin = request.form['e']
    BMI = request.form['f']  # Corrected field name to 'f'
    DiabetesPedigreeFunction = request.form['g']
    Age = request.form['h']
    arr = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
    print("Flask application is running...")
