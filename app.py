from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "/media/rohith/windows/vscode/Parkinson_Disease/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure directory exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model_path = '/media/rohith/windows/vscode/parkinson.h5'  # Ensure correct path
model = load_model(model_path)

# Function to predict Parkinson's
def predict_parkinson(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Affected by Parkinson's Disease" if prediction > 0.5 else "Not Affected by Parkinson's Disease"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # Save file correctly
        prediction = predict_parkinson(file_path)
        return render_template('result.html', image_filename=file.filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
