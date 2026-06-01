from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import os
import uuid

try:
    from utils import load_trained_model, predict_single_image
except ImportError:
    from .utils import load_trained_model, predict_single_image

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Path to saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'pneumonia_model.h5')

# Try to load model at startup; if not found, app will run in demo/mock mode
model = None
_model_mtime = None

def ensure_model_loaded():
    """Load or reload the model if a newer file exists."""
    global model, _model_mtime
    try:
        if os.path.exists(MODEL_PATH):
            mtime = os.path.getmtime(MODEL_PATH)
            if model is None or _model_mtime is None or mtime > _model_mtime:
                print(f"Loading model from {MODEL_PATH}")
                model = load_trained_model(MODEL_PATH)
                _model_mtime = mtime
                return True
            return True
        else:
            print('Model not found; web demo will run in mock mode. Place pneumonia_model.h5 at project root to enable real predictions.')
            model = None
            _model_mtime = None
            return False
    except Exception as e:
        print('Error loading model:', e)
        model = None
        _model_mtime = None
        return False

# Try initial load
ensure_model_loaded()

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB


@app.route('/', methods=['GET', 'POST'])
def index():
    # Ensure model is up-to-date on each page visit
    model_loaded = ensure_model_loaded()

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Save uploaded file
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Prepare a relative URL for the uploaded image
            image_url = url_for('uploaded_file', filename=filename)

            # If model loaded, use it; otherwise generate mock prediction
            if model is not None:
                try:
                    result = predict_single_image(file_path, model)
                    # return image_url instead of filesystem path
                    result['image_url'] = image_url
                    return render_template('result.html', result=result)
                except Exception as e:
                    flash(f'Prediction error: {e}')
                    return redirect(request.url)
            else:
                # Mock prediction for demo purposes
                import random
                pred = random.choice(['NORMAL', 'PNEUMONIA'])
                confidence = random.random() * 0.5 + 0.5
                mock_result = {
                    'image_path': file_path,
                    'image_url': image_url,
                    'class': pred,
                    'confidence': confidence,
                    'raw_prediction': 1.0 if pred == 'PNEUMONIA' else 0.0
                }
                return render_template('result.html', result=mock_result)
    return render_template('index.html', model_loaded=model_loaded)


if __name__ == '__main__':
    # Flask development server
    app.run(host='0.0.0.0', port=8000, debug=True)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/reload')
def reload_model():
    ok = ensure_model_loaded()
    if ok:
        flash('Model loaded/reloaded successfully.')
    else:
        flash('No model found or error loading model.')
    return redirect(url_for('index'))
