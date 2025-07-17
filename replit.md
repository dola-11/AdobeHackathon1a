# PDF Outline Extractor

## Overview

This is a Flask-based web application that extracts structured outlines (Title, H1, H2 headings) from PDF documents using machine learning. The system allows users to train a custom model on sample PDF and JSON ground truth data, then use that model to process new PDFs and extract their outline structure.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Technology**: HTML, CSS, JavaScript (Vanilla JS)
- **Framework**: Bootstrap 5 for responsive UI
- **Structure**: Single-page application with multiple sections for training and processing
- **Features**: File upload, real-time status updates, batch processing, results download

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Pattern**: REST API with server-side rendering
- **Structure**: Modular design with separate modules for feature extraction, model training, and PDF processing
- **File Handling**: Secure file uploads with validation and temporary file management

### Machine Learning Pipeline
- **Feature Extraction**: Custom feature engineering from PDF text properties (font size, position, styling)
- **Model**: LightGBM classifier for text element classification
- **Training**: Supervised learning using PDF-JSON pairs as ground truth
- **Inference**: Real-time classification of PDF text elements into outline categories

## Key Components

### Core Modules
1. **Feature Extractor** (`src/feature_extractor.py`)
   - Extracts text features from PDF using PyMuPDF
   - Generates feature vectors based on font size, position, formatting
   - Handles document-level statistics for normalization

2. **Model Training** (`src/train_model.py`)
   - Trains LightGBM classifier on PDF-JSON pairs
   - Handles label encoding and feature preparation
   - Saves trained model artifacts for inference

3. **PDF Processor** (`src/pdf_processor.py`)
   - Loads trained model and processes new PDFs
   - Applies feature extraction and classification
   - Generates structured outline output

### Web Interface
- **Training Section**: Upload PDF and JSON files, train custom models
- **Processing Section**: Single PDF and batch processing capabilities
- **Results Display**: Structured outline visualization with download options

## Data Flow

1. **Training Phase**:
   - User uploads PDF + JSON training data
   - Features extracted from PDF text elements
   - Ground truth labels mapped from JSON structure
   - LightGBM model trained and saved

2. **Processing Phase**:
   - User uploads PDF(s) for processing
   - Features extracted using same pipeline
   - Trained model classifies text elements
   - Structured outline generated and returned

3. **Output Generation**:
   - Results formatted as JSON with title and hierarchical outline
   - Batch processing creates ZIP archive of results
   - Web interface displays structured outline

## External Dependencies

### Core Libraries
- **PyMuPDF (fitz)**: PDF parsing and text extraction
- **LightGBM**: Machine learning model for classification
- **scikit-learn**: Data preprocessing and evaluation
- **Flask**: Web framework and API
- **joblib**: Model serialization and persistence

### Frontend Dependencies
- **Bootstrap 5**: UI framework and responsive design
- **Font Awesome**: Icons and visual elements
- **Vanilla JavaScript**: Client-side interactivity

## Deployment Strategy

### Current Setup
- **Environment**: Flask development server
- **File Storage**: Local filesystem with organized folder structure
- **Model Persistence**: Local model files saved with joblib
- **Configuration**: Environment-based settings for upload limits and paths

### Folder Structure
```
/uploads - Temporary file storage for user uploads
/models - Trained model artifacts and encoders
/output - Generated results and processed files
/static - CSS, JavaScript, and static assets
/templates - HTML templates for web interface
/src - Core Python modules for ML pipeline
```

### Production Considerations
- File size limits (16MB default)
- Secure filename handling
- Model versioning and management
- Error handling and user feedback
- Batch processing capabilities

The application is designed to work entirely offline without external API calls, making it suitable for environments with restricted internet access or privacy requirements.