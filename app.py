from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load scaler and model
scaler = pickle.load(open("Model/standardScaler.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Collect input data from form
            Pregnancies = float(request.form.get('Pregnancies'))
            Glucose = float(request.form.get('Glucose'))
            BloodPressure = float(request.form.get('BloodPressure'))
            SkinThickness = float(request.form.get('SkinThickness'))
            Insulin = float(request.form.get('Insulin'))
            BMI = float(request.form.get('BMI'))
            DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
            Age = float(request.form.get('Age'))

            # Prepare and scale input
            new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                          Insulin, BMI, DiabetesPedigreeFunction, Age]])

            # Make prediction
            prediction = model.predict(new_data)[0]
            confidence = model.predict_proba(new_data)[0][1] * 100

            result = "Diabetic" if prediction == 1 else "Non-Diabetic"

            return render_template('single_prediction.html',
                                   result=result,
                                   confidence=round(confidence, 2))

        except Exception as e:
            return render_template('single_prediction.html',
                                   error="Invalid input. Please enter numeric values only.")

    return render_template('predict.html')


if __name__ == "__main__":
    app.run(debug=True)


