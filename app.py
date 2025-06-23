from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = load_model('models/fruit_classifier.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Define class names based on your 16-class dataset
class_names = [
    'freshapples', 'freshbanana', 'freshcapsicum', 'freshcucumber',
    'freshokra', 'freshoranges', 'freshpotato', 'freshtomato',
    'rottenapples', 'rottenbanana', 'rottencapsicum', 'rottencucumber',
    'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato'
]

# Ensure the static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Save the uploaded file to the static folder
            image_filename = os.path.join('static', file.filename)
            file.save(image_filename)
            print(f"Image saved as {image_filename}")

            # Preprocess the image
            img = image.load_img(image_filename, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            print("Image preprocessed successfully")

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_index]
            confidence = predictions[0][predicted_class_index]
            print(f"Prediction: {predicted_class}, Confidence: {confidence}")

            result = {
                'predicted_class': predicted_class.replace('fresh', 'Fresh ').replace('rotten', 'Rotten '),
                'confidence': float(confidence)
            }
            return jsonify(result)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
