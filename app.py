from flask import Flask, request, jsonify

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

#
# from keras.src.saving import load_model
# from keras.src.utils import img_to_array, load_img

import os
port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "cat_dog_classifier_model.h5"
model = load_model(MODEL_PATH)


# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to the input size
    img_array = img_to_array(image)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array


import io  # Import the io module

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file is included in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']

        # Convert the file stream to a BytesIO object
        file_stream = io.BytesIO(file.read())

        # Open the image using PIL
        img = load_img(file_stream)

        # Preprocess the image
        image_array = preprocess_image(img)

        # Make a prediction
        prediction = model.predict(image_array)
        result = "Dog" if prediction[0] > 0.5 else "Cat"

        return jsonify({"prediction": result}), 200
    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
