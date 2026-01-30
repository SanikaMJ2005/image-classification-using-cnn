"""
Flask Application for CNN Image Classification
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model (if available)
try:
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    MODEL_LOADED = True
except:
    MODEL_LOADED = False
    print("Warning: Model not found. Please train the model first.")

# Define class labels for MNIST
CLASS_LABELS = {i: str(i) for i in range(10)}


@app.route('/')
def home():
    """Render the landing page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image_array = np.array(image) / 255.0  # Normalize
        image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model
        
        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class]) * 100
        
        # Get the image as base64 for display
        image.seek(0) if hasattr(image, 'seek') else None
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'prediction': int(predicted_class),
            'confidence': round(confidence, 2),
            'image': f'data:image/png;base64,{img_base64}',
            'all_predictions': {str(i): round(float(p) * 100, 2) for i, p in enumerate(prediction[0])}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')


@app.route('/api/model-info')
def model_info():
    """Get model information"""
    return jsonify({
        'status': 'loaded' if MODEL_LOADED else 'not_loaded',
        'model_type': 'CNN (Convolutional Neural Network)',
        'dataset': 'MNIST',
        'input_size': '28x28 pixels',
        'classes': 10,
        'class_labels': CLASS_LABELS
    })


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
