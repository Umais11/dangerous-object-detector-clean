from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'dangerous_object_detection_secret_key_2024'

# ================= CONFIG =================
UPLOAD_FOLDER = '../static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = '../model/saved_models/best_model.h5'
IMG_SIZE = 128

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ================= LOAD MODEL =================
print("Loading CNN model...")
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# ================= HELPERS =================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(img_path):
    if model is None:
        return None, None, None, "Model not loaded"

    try:
        processed_img = preprocess_image(img_path)

        # 🔥 SIGMOID OUTPUT (binary model)
        prediction = float(model.predict(processed_img, verbose=0)[0][0])

        if prediction < 0.85:
            predicted_class = "dangerous"
            confidence = (1 - prediction) * 100
            is_dangerous = True
        else:
            predicted_class = "safe"
            confidence = prediction * 100
            is_dangerous = False

        return predicted_class, round(confidence, 2), is_dangerous, None

    except Exception as e:
        return None, None, None, str(e)


# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file uploaded!', 'error')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No file selected!', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_class, confidence, is_dangerous, error = predict_image(filepath)

        if error:
            flash(f'Prediction error: {error}', 'error')
            return redirect(url_for('index'))

        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_dangerous': is_dangerous,
            'image_path': url_for('static', filename=f'uploads/{filename}')
        }

        return render_template('index.html', result=result)

    else:
        flash('Invalid file type! Please upload PNG, JPG, or JPEG.', 'error')
        return redirect(url_for('index'))


# ================= MAIN =================
if __name__ == '__main__':
    print("=" * 50)
    print("DANGEROUS OBJECT DETECTION - FLASK SERVER")
    print("=" * 50)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Upload Folder: {UPLOAD_FOLDER}")
    print("=" * 50)
    print("🚀 Server running at: http://127.0.0.1:5000")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)
