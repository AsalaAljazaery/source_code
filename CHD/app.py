from flask import Flask, request, render_template, jsonify
import joblib
import os
from tensorflow import keras



app = Flask(__name__)
model = None

def load_model():
    global model
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_filename = "model.joblib"
    model_path = os.path.join(current_directory, model_filename)

    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model: {e}")

# Load the model when the app starts
load_model()

@app.route('/')
def home():
    return render_template('index.html')
  

@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    sex = int(float(request.form["sex"]))
    age = int(float(request.form["age"]))
    CurrentSmoker = int(float(request.form["CurrentSmoker"]))
    CigsPerDay = int(float(request.form["CigsPerDay"]))
    PrevalentHyp = int(float(request.form["PrevalentHyp"]))
    TotChol = int(float(request.form["TotChol"]))
    SysBP = int(float(request.form["SysBP"]))
    DiaBP = int(float(request.form["DiaBP"]))
    BMI = int(float(request.form["BMI"]))
    HeartRate = int(float(request.form["HeartRate"]))
    Glucose = int(float(request.form["Glucose"]))


    prediction = model.predict([[sex, age,CurrentSmoker,CigsPerDay,PrevalentHyp,TotChol,SysBP,DiaBP,BMI,HeartRate,Glucose]])
    

    return render_template('index.html', prediction_text=f'The chance of having CHD is {float(prediction)}')



if __name__ == "__main__":
    app.run(debug=True)
