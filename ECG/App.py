import os
import numpy as np
import base64
import cv2
from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
model = None

def load_model():
    global model
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_filename = "model2.h5"
    model_path = os.path.join(current_directory, model_filename)

    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model: {e}")

# Load the model when the app starts
load_model()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        global model

        data = request.form
        sex = data["sex"]
        height = float(data["height"])
        weight = float(data["weight"])
        age = float(data["age"])
        img = request.files["img"]

        # Map "F/f" to 1 (female) and "M/m" to 0 (male)
        sex = sex.lower()  # Convert to lowercase for case-insensitive comparison
        if sex == "f":
            sex = 1
        elif sex == "m":
            sex = 0
        else:
            return jsonify({'error': 'Invalid value for "sex". Please use "F" for female or "M" for male.'})

        # Load the ECG image
        image = Image.open(img)
        ecg_image = np.array(image)

        # Image processing steps
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(ecg_image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to segment the image (you can adjust the threshold value)
        _, segmented_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

        # Find contours in the segmented image
        contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a list to store segmented regions
        segmented_images = []

        # Iterate through the contours and extract segmented regions
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            segmented_region = ecg_image[y:y+h, x:x+w]
            segmented_images.append(segmented_region)

        # Initialize a list to store extracted signals for each lead
        lead_signals = []

        # Determine a common shape for all lead signals 
        common_shape = (1000, 12)  # Define the shape 

        # Iterate through segmented regions 
        for i, segmented_region in enumerate(segmented_images):
            # Perform signal extraction (you may use peak detection or other methods)
            # Here, we simply take the grayscale values as the signal
            signal = segmented_region[:, :, 0]

            # Perform signal processing
            # For example, you can apply Gaussian blur as a simple smoothing filter
            smoothed_signal = cv2.GaussianBlur(signal, (5, 5), 0)

            # Ensure all lead signals have the same shape
            resized_signal = cv2.resize(smoothed_signal, common_shape)  # Reverse the shape for the resize

            # Normalize pixel values to the range [0, 1]
            resized_signal = resized_signal / 255.0

            # Append the resized signal to the list of lead signals
            lead_signals.append(resized_signal)

        # Convert the list of lead signals into a NumPy array
        ecg_signals = np.array(lead_signals)

        # Ensure that 'input1' has the same number of samples as 'ecg_signals'
        num_samples = ecg_signals.shape[0]

        # Create input1 as an array with the same shape as 'ecg_signals'
        input1 = np.array([[sex, age, height, weight]] * num_samples, dtype=np.float32)

        # Transpose the 'ecg_signals' array to match the expected input shape of 'model02'
        ecg_signals = ecg_signals.transpose(0, 2, 1)  # Shape: (num_samples, 12, 1000)

        # Prepare the input data for prediction
        input_data = [input1, ecg_signals]

        # Make predictions using the model
        predictions = model.predict(input_data)

        # Assuming 'predictions' is an array of probabilities
        predicted_class_index = np.argmax(predictions)

        # Replace this with the class labels
        class_labels = ["NORM", "MI", "STTC", "CD", "HYP"]

        # Get the corresponding class label
        predicted_class_label = class_labels[predicted_class_index]

        # Render the "results.html" template with the prediction result
        return render_template('results.html', result=predicted_class_label)

if __name__ == '__main__':
    app.run(debug=True)
