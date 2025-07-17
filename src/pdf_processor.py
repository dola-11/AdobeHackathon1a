import os
import joblib
from .feature_extractor import extract_features

def process_pdf(pdf_path: str, model_dir: str) -> dict:
    """
    Process a PDF file to extract its outline structure.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        model_dir (str): Directory containing trained model files
        
    Returns:
        dict: Extracted outline structure with title and outline
    """
    # Load trained model and artifacts
    try:
        model_path = os.path.join(model_dir, 'heading_model.pkl')
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        features_path = os.path.join(model_dir, 'feature_keys.pkl')
        
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        feature_keys = joblib.load(features_path)
        
    except FileNotFoundError as e:
        raise Exception(f"Model files not found: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")
    
    # Extract features from PDF
    try:
        line_features = extract_features(pdf_path)
    except Exception as e:
        raise Exception(f"Failed to extract features from PDF: {str(e)}")
    
    if not line_features:
        return {
            "title": "Empty or unreadable PDF",
            "outline": [],
            "error": "No text content found in PDF"
        }
    
    # Prepare features for prediction
    X_predict = []
    for line in line_features:
        feature_vector = []
        for key in feature_keys:
            value = line.get(key, 0)
            # Convert boolean to int for consistency
            if isinstance(value, bool):
                value = int(value)
            feature_vector.append(value)
        X_predict.append(feature_vector)
    
    # Make predictions
    try:
        predicted_labels_encoded = model.predict(X_predict)
        predicted_labels = encoder.inverse_transform(predicted_labels_encoded)
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")
    
    # Extract title and outline
    title = "Untitled Document"
    outline = []
    found_title = False
    
    # First pass: look for title
    for i, label in enumerate(predicted_labels):
        if label == 'Title':
            title = line_features[i]['text']
            found_title = True
            break
    
    # Second pass: extract outline items
    for i, label in enumerate(predicted_labels):
        if label.startswith('H') and not (found_title and line_features[i]['text'] == title):
            outline.append({
                "level": label,
                "text": line_features[i]['text'],
                "page": line_features[i]['page_num']
            })
    
    # If no title was found but we have outline items, use the first H1 as title
    if not found_title and outline:
        h1_items = [item for item in outline if item['level'] == 'H1']
        if h1_items:
            title = h1_items[0]['text']
            outline = [item for item in outline if item['text'] != title]
    
    # Sort outline by page number and level
    outline.sort(key=lambda x: (x['page'], x['level']))
    
    result = {
        "title": title,
        "outline": outline
    }
    
    # Add some metadata
    result["metadata"] = {
        "total_pages": max([line['page_num'] for line in line_features]) if line_features else 0,
        "total_text_lines": len(line_features),
        "outline_items": len(outline)
    }
    
    return result
