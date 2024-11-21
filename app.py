import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import io

# Set TensorFlow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Optimize TensorFlow threading
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the pre-trained model once during app startup
MODEL_PATH = "cat_dog_classifier_model.h5"
model = load_model(MODEL_PATH)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        file_stream = io.BytesIO(file.read())
        img = load_img(file_stream)
        image_array = preprocess_image(img)

        prediction = model.predict(image_array)
        result = "Dog" if prediction[0] > 0.5 else "Cat"
        return jsonify({"prediction": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "API is running!", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
