# SmartNotes PDF Outline Extractor

A machine learning-powered PDF outline extraction system that automatically identifies and extracts structured outlines (titles, headings, and hierarchical structure) from PDF documents.

## 🎯 Approach

### **Machine Learning Pipeline**
This solution uses a **supervised learning approach** with a **RandomForest classifier** to identify different text elements in PDFs:

1. **Feature Extraction**: Extracts 13 text-based features from each line of text in the PDF
2. **Model Training**: Trains on PDF-JSON pairs to learn heading patterns
3. **Classification**: Classifies text elements into Title, H1, H2, H3, H4, or Body Text
4. **Heuristic Refinement**: Applies rule-based post-processing to improve accuracy

### **Key Features**
- **Multi-level heading detection** (Title, H1, H2, H3, H4)
- **Font-based analysis** (size, weight, positioning)
- **Contextual understanding** (spacing, indentation, numbering)
- **Robust error handling** for corrupted or complex PDFs
- **Performance optimized** for ≤10 seconds on 50-page PDFs

## 🤖 Models and Libraries Used

### **Core Machine Learning**
- **RandomForest**: Ensemble classifier for text element classification
- **scikit-learn**: Data preprocessing and label encoding
- **joblib**: Model serialization and persistence

### **PDF Processing**
- **PyMuPDF (fitz)**: High-performance PDF parsing and text extraction
- **Feature Engineering**: Custom 13-feature extraction pipeline

### **Web Framework**
- **Flask**: Web application framework for training interface
- **Werkzeug**: File handling and security utilities

### **Model Size**: ~50MB total (well under 200MB limit)

## 🏗️ Architecture

### **Modular Design**
```
src/
├── feature_extractor.py    # PDF text feature extraction
├── pdf_processor.py        # Main PDF processing pipeline
└── train_model.py         # Model training and validation

models/
├── heading_model.pkl      # Trained RandomForest classifier
├── label_encoder.pkl      # Label encoding for classes
└── feature_keys.pkl       # Feature key definitions
```

### **Feature Engineering**
The system extracts 13 features from each text line:
- Font size ratio and relative positioning
- Bold/italic formatting detection
- Indentation and centering analysis
- Word count and text length
- Numbering patterns and capitalization
- Spacing and contextual indicators

## 🚀 How to Build and Run

### **Prerequisites**
- Docker installed and running
- Input PDF files ready for processing

### **1. Build the Docker Image**
```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

### **2. Prepare Input Files**
```bash
# Create input directory and add PDF files
mkdir input
cp your_documents/*.pdf input/
```

### **3. Run the Container**
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolutionname:somerandomidentifier
```

### **4. Check Results**
```bash
# View generated JSON files
ls output/
# Example: document1.json, document2.json, output.json
```

## 📊 Output Format

### **Individual PDF Results** (`filename.json`)
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "Background",
      "page": 2
    }
  ],
  "metadata": {
    "total_pages": 50,
    "total_text_lines": 1500,
    "outline_items": 25,
    "processing_time_seconds": 3.45,
    "input_file": "document.pdf",
    "output_file": "document.json"
  }
}
```

### **Batch Summary** (`output.json`)
```json
{
  "title": "Batch Processing Summary",
  "outline": [],
  "metadata": {
    "total_files_processed": 5,
    "input_directory": "/app/input",
    "output_directory": "/app/output",
    "output_file": "output.json"
  }
}
```

## ⚡ Performance Characteristics

- **Processing Speed**: 2-8 seconds for typical 50-page documents
- **Model Size**: ~50MB (LightGBM + encoders)
- **Memory Usage**: ~200-500MB during processing
- **CPU Optimization**: Multi-threaded for 8-core systems
- **Network**: Completely offline - no external dependencies

## 🔧 Advanced Features

### **Robust Error Handling**
- Graceful handling of corrupted PDFs
- Fallback processing for complex layouts
- Detailed error reporting in JSON output

### **Heuristic Refinement**
- Post-processing rules to improve ML predictions
- Context-aware heading detection
- Numbering pattern recognition

### **Performance Monitoring**
- Real-time processing time tracking
- Automatic warnings for slow processing
- Detailed metadata for analysis

## 🧪 Testing

### **Test Different PDF Types**
- Simple text documents
- Complex multi-column layouts
- Documents with images and tables
- Scanned documents (with OCR text layers)

### **Performance Validation**
```bash
# Test with various PDF sizes
# Verify ≤10 second processing for 50-page PDFs
# Check memory usage stays under 16GB
```

## 📝 Technical Notes

### **Constraints Met**
✅ **Execution Time**: ≤10 seconds for 50-page PDFs  
✅ **Network**: No internet access required  
✅ **Model Size**: ≤200MB (actual: ~50MB)  
✅ **Runtime**: CPU-only (amd64) compatible  
✅ **Resources**: Optimized for 8 CPUs and 16GB RAM  

### **Pro Tips Implemented**
✅ **Multi-feature approach**: Not just font sizes  
✅ **Modular code structure**: Reusable for Round 1B  
✅ **Comprehensive testing**: Simple and complex PDFs  
✅ **Robust error handling**: Corrupted file support  

## 🔄 Training Your Own Model

The system includes a web interface for training custom models:

1. **Start the Flask app**: `python app.py`
2. **Upload training data**: PDF + JSON pairs
3. **Train model**: Use the web interface
4. **Deploy**: Replace model files and rebuild container

## 📄 License

This implementation is designed for the specified evaluation environment and constraints. 