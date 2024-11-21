from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS
import numpy as np
import io  # Import the io module
import os  # For reading environment variables

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load the pre-trained model
MODEL_PATH = "cat_dog_classifier_model.h5"
model = load_model(MODEL_PATH)


# Function to preprocess the image
def preprocess_image(image):
    """
    Resize and normalize the input image to match the model's requirements.
    """
    image = image.resize((150, 150))  # Resize to the input size
    img_array = img_to_array(image)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to handle image prediction requests.
    Expects a file with the key 'file' in the request.
    """
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


@app.route('/')
def home():
    """
    Root endpoint for health checks.
    """
    return "API is running!", 200


if __name__ == "__main__":
    # Use the PORT environment variable for deployment, default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
