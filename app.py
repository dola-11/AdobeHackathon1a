import os
import json
import joblib
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from src.feature_extractor import extract_features
from src.train_model import train_model
from src.pdf_processor import process_pdf
import tempfile
import zipfile
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['OUTPUT_FOLDER'] = 'output'

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['MODEL_FOLDER'], app.config['OUTPUT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    try:
        if 'pdf_file' not in request.files or 'json_file' not in request.files:
            return jsonify({'error': 'Both PDF and JSON files are required'}), 400
        
        pdf_file = request.files['pdf_file']
        json_file = request.files['json_file']
        
        if pdf_file.filename == '' or json_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if not (allowed_file(pdf_file.filename) and allowed_file(json_file.filename)):
            return jsonify({'error': 'Invalid file types'}), 400
        
        # Save files
        pdf_filename = secure_filename(pdf_file.filename)
        json_filename = secure_filename(json_file.filename)
        
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
        
        pdf_file.save(pdf_path)
        json_file.save(json_path)
        
        return jsonify({
            'message': 'Training files uploaded successfully',
            'pdf_file': pdf_filename,
            'json_file': json_filename
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    try:
        data = request.get_json()
        pdf_filename = data.get('pdf_file')
        json_filename = data.get('json_file')
        
        if not pdf_filename or not json_filename:
            return jsonify({'error': 'PDF and JSON filenames are required'}), 400
        
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
        
        if not os.path.exists(pdf_path) or not os.path.exists(json_path):
            return jsonify({'error': 'Training files not found'}), 404
        
        # Train the model
        model_info = train_model(pdf_path, json_path, app.config['MODEL_FOLDER'])
        
        return jsonify({
            'message': 'Model trained successfully',
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/process_pdf', methods=['POST'])
def process_pdf_endpoint():
    try:
        if 'pdf_file' not in request.files:
            return jsonify({'error': 'PDF file is required'}), 400
        
        pdf_file = request.files['pdf_file']
        
        if pdf_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(pdf_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Check if model exists
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'heading_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': 'No trained model found. Please train a model first.'}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_file.save(tmp_file.name)
            
            # Process the PDF
            result = process_pdf(tmp_file.name, app.config['MODEL_FOLDER'])
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/batch_process', methods=['POST'])
def batch_process():
    try:
        if 'pdf_files' not in request.files:
            return jsonify({'error': 'PDF files are required'}), 400
        
        pdf_files = request.files.getlist('pdf_files')
        
        if not pdf_files or all(f.filename == '' for f in pdf_files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Check if model exists
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'heading_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': 'No trained model found. Please train a model first.'}), 400
        
        results = []
        
        for pdf_file in pdf_files:
            if pdf_file.filename and allowed_file(pdf_file.filename):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        pdf_file.save(tmp_file.name)
                        
                        # Process the PDF
                        result = process_pdf(tmp_file.name, app.config['MODEL_FOLDER'])
                        result['filename'] = pdf_file.filename
                        results.append(result)
                        
                        # Clean up temporary file
                        os.unlink(tmp_file.name)
                        
                except Exception as e:
                    results.append({
                        'filename': pdf_file.filename,
                        'error': str(e)
                    })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500

@app.route('/download_results', methods=['POST'])
def download_results():
    try:
        data = request.get_json()
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'No results to download'}), 400
        
        # Create a zip file with all results
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for result in results:
                filename = result.get('filename', 'unknown.pdf')
                json_filename = filename.replace('.pdf', '.json')
                
                # Remove filename from result before saving
                clean_result = {k: v for k, v in result.items() if k != 'filename'}
                
                json_content = json.dumps(clean_result, indent=4, ensure_ascii=False)
                zf.writestr(json_filename, json_content)
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='pdf_outline_results.zip'
        )
        
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/model_status')
def model_status():
    try:
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'heading_model.pkl')
        encoder_path = os.path.join(app.config['MODEL_FOLDER'], 'label_encoder.pkl')
        features_path = os.path.join(app.config['MODEL_FOLDER'], 'feature_keys.pkl')
        
        model_exists = all(os.path.exists(path) for path in [model_path, encoder_path, features_path])
        
        status = {
            'model_trained': model_exists,
            'model_files': {
                'heading_model.pkl': os.path.exists(model_path),
                'label_encoder.pkl': os.path.exists(encoder_path),
                'feature_keys.pkl': os.path.exists(features_path)
            }
        }
        
        if model_exists:
            # Try to load model info
            try:
                feature_keys = joblib.load(features_path)
                encoder = joblib.load(encoder_path)
                status['feature_count'] = len(feature_keys)
                status['classes'] = list(encoder.classes_)
            except Exception:
                pass
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
