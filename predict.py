from flask import Flask, request, jsonify ,render_template
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import os, shutil


app = Flask(__name__)





@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract values from form
    TotalPregnanciestillnow = float(request.form['TotalPregnanciestillnow'])
    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    SkinThickness = float(request.form['SkinThickness'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])
    
    # Create the input vector
    input_data = [[TotalPregnanciestillnow, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
    
    # Make predictions using the loaded model
    model = mlflow.sklearn.load_model("saved_model")
    predictions = model.predict(input_data)
    print()
    # Return the predictions as a JSON response
    return jsonify({'prediction': predictions.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5014)




