from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load TensorFlow model
tf_model = load_model("turkeysaved_model.pb")

# Load Keras model
keras_model = load_model("adam_model.h5")

# Define prediction function
def predict(tf_model, keras_model, input_data):
    # Use TensorFlow model
    tf_prediction = tf_model(input_data)

    # Use Keras model
    keras_prediction = keras_model.predict(input_data)

    return tf_prediction, keras_prediction

@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.json  # Assuming JSON data is sent in the request

    # Preprocess input data as required by the models
    # Example: Convert JSON data to NumPy array
    input_data = np.array(data['input'])

    # Make predictions
    tf_prediction, keras_prediction = predict(tf_model, keras_model, input_data)

    # Return predictions
    return jsonify({
        'tf_prediction': tf_prediction.tolist(),
        'keras_prediction': keras_prediction.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0',debug=True)