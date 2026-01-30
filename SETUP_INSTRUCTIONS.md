# CNN Image Classification Web Application - Setup Instructions

This guide will help you set up and run the CNN Image Classification web application.

## Project Structure

```
image-classification-using-cnn/
├── app.py                          # Flask web application
├── cnn_image_classifier.py          # Model training script
├── mnist_cnn_model.h5              # Trained model (generated after training)
├── templates/
│   ├── index.html                  # Home/landing page
│   └── about.html                  # About page
├── static/
│   ├── css/
│   │   └── style.css               # Styling for all pages
│   └── js/
│       └── script.js               # Client-side JavaScript functionality
├── uploads/                         # Directory for uploaded images
├── requirements.txt                # Python dependencies
└── README.md                        # Project documentation
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Installation Steps

### 1. Install Dependencies

Open a terminal/command prompt in the project directory and install required packages:

```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)

If you want to train the model from scratch, run:

```bash
python cnn_image_classifier.py
```

This will:
- Load the MNIST dataset
- Train a CNN model
- Evaluate the model
- Save the model as `mnist_cnn_model.h5`

**Note:** Training may take 10-15 minutes depending on your hardware.

### 3. Run the Flask Application

Start the web application:

```bash
python app.py
```

You should see output similar to:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### 4. Access the Web Application

Open your web browser and navigate to:
```
http://localhost:5000
```

## Features

### Landing Page (Home)
- Hero section with project introduction
- Features section highlighting key capabilities
- Live demo area for image classification
- How it works section with step-by-step guide
- Responsive design for all devices

### Image Classification Demo
1. **Upload Image**: Drag and drop or click to browse for handwritten digit images
2. **Preview**: See the uploaded image preview
3. **Prediction**: Get instant classification result
4. **Confidence Score**: View the confidence percentage
5. **All Probabilities**: See classification scores for all digits (0-9)

### About Page
- Detailed project overview
- Dataset information
- Model architecture explanation
- Technologies used
- Performance metrics
- Getting started guide
- Future enhancements

## Usage Tips

### For Best Results:

1. **Image Requirements**:
   - Use clear, high-contrast images of handwritten digits
   - Single digit per image (the model classifies one digit at a time)
   - Similar to the MNIST dataset style

2. **Image Formats**:
   - Supported: PNG, JPG, JPEG, GIF, BMP, etc.
   - Maximum file size: 16MB

3. **Improving Predictions**:
   - Ensure the digit is centered in the image
   - Use a white or light background
   - Write the digit in black or dark ink

## API Endpoints

### POST /predict
Upload an image for classification

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Parameter: `file` (image file)

**Response:**
```json
{
  "prediction": 5,
  "confidence": 99.87,
  "image": "data:image/png;base64,...",
  "all_predictions": {
    "0": 0.01,
    "1": 0.02,
    "2": 0.05,
    "3": 0.01,
    "4": 0.02,
    "5": 99.87,
    "6": 0.01,
    "7": 0.01,
    "8": 0.00,
    "9": 0.00
  }
}
```

### GET /api/model-info
Get information about the loaded model

**Response:**
```json
{
  "status": "loaded",
  "model_type": "CNN (Convolutional Neural Network)",
  "dataset": "MNIST",
  "input_size": "28x28 pixels",
  "classes": 10,
  "class_labels": {
    "0": "0",
    "1": "1",
    ...
    "9": "9"
  }
}
```

## Troubleshooting

### Model Not Loading
**Error:** "Model not found. Please train the model first."
**Solution:** Run `python cnn_image_classifier.py` to train the model

### Port Already in Use
**Error:** "Address already in use"
**Solution:** 
- Change the port in `app.py` (default is 5000)
- Or stop the existing process using that port

### Image Upload Issues
**Error:** "Failed to process image"
**Solution:**
- Ensure the image is a valid image file
- Check that the model is loaded (see Model Not Loading)
- Try a different image

### Dependencies Not Installed
**Error:** "ModuleNotFoundError"
**Solution:** Run `pip install -r requirements.txt` again

## Configuration

### Modify Flask Settings (in app.py)

```python
# Maximum upload size (default: 16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Debug mode (change to False for production)
app.run(debug=True)

# Host and port
app.run(host='127.0.0.1', port=5000)
```

## Model Information

### Architecture
- Input: 28×28 grayscale images
- Conv2D layers with ReLU activation
- MaxPooling for dimension reduction
- Dropout layers for regularization
- Dense layers for classification
- Output: Softmax with 10 classes (digits 0-9)

### Performance
- Training Accuracy: >98%
- Test Accuracy: >95%
- Inference Time: <100ms per image

## Development

### Modifying the Frontend
- Edit HTML in `templates/` directory
- Modify CSS in `static/css/style.css`
- Update JavaScript in `static/js/script.js`

### Modifying the Backend
- Edit `app.py` to change Flask logic
- Update model training in `cnn_image_classifier.py`

### Adding New Features
1. Create new routes in `app.py`
2. Add corresponding HTML templates in `templates/`
3. Update CSS/JS as needed

## Deployment

For production deployment:

1. Set `debug=False` in `app.py`
2. Use a production WSGI server (Gunicorn, uWSGI)
3. Set up proper error logging
4. Use environment variables for configuration

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Performance Optimization

- Model predictions are cached when possible
- Image preprocessing is optimized
- Frontend uses efficient DOM manipulation
- CSS is minified for production

## Supported Browsers

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

This project is open source and available for educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the error messages in browser console (F12)
3. Check Flask logs in terminal for server errors

## Future Enhancements

- [ ] Multi-digit recognition
- [ ] Webcam input
- [ ] Model visualization
- [ ] Custom dataset support
- [ ] Docker containerization
- [ ] Mobile app

---

Enjoy classifying handwritten digits with our CNN Image Classifier!
