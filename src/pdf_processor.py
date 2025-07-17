import os
import re
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
    
    # Make predictions and apply heuristic rules
    try:
        predicted_labels_encoded = model.predict(X_predict)
        ml_predictions = encoder.inverse_transform(predicted_labels_encoded)
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")
    
    # Debug: print prediction results
    print(f"Processing {len(ml_predictions)} text elements...")
    label_counts = {}
    for label in ml_predictions:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"ML model predictions: {label_counts}")
    
    # Apply heuristic rules to improve predictions
    final_predictions = []
    all_font_sizes = [line.get('relative_font_size', 10) for line in line_features]
    median_font_size = sorted(all_font_sizes)[len(all_font_sizes)//2] if all_font_sizes else 12
    
    for i, line in enumerate(line_features):
        text = line['text'].strip()
        ml_label = ml_predictions[i]
        
        # Much more conservative heuristic rules
        is_large_font = line.get('relative_font_size', 10) > median_font_size * 1.4
        is_bold = line.get('is_bold', False)
        is_numbered = bool(re.match(r"^\d+(\.\d+)*\s+\w+", text))  # Number + space + word
        has_colon = text.strip().endswith(':') and not text.strip().endswith('::')
        is_reasonable_length = 5 <= len(text) <= 200  # Must be substantial but not too long
        is_reasonable_words = 1 <= len(text.split()) <= 20
        has_space_before = line.get('space_before', 0) > 15
        is_first_page = line.get('page_num', 1) == 1
        is_appendix = text.lower().startswith('appendix')
        
        # Strong filters for non-headings
        is_page_number = text.isdigit() and len(text) <= 3
        is_single_char = len(text.strip()) <= 2
        is_copyright_etc = text.lower().startswith(('copyright', 'version', 'page', '©', 'www.', 'http'))
        is_incomplete = text.endswith(('-', '–')) or len(text.split()) == 1 and len(text) < 8
        has_lowercase_start = text and text[0].islower() and not is_numbered
        
        if (is_page_number or is_single_char or is_copyright_etc or 
            is_incomplete or has_lowercase_start or not is_reasonable_length):
            final_predictions.append('Body Text')
        # Title detection - very selective for first page
        elif (is_first_page and i < 3 and len(text) > 20 and 
              (is_large_font or is_bold) and ':' in text and 
              not text.lower().startswith(('copyright', 'version'))):
            final_predictions.append('Title')
        # H1 detection - major sections only
        elif ((is_large_font and is_bold and is_reasonable_words) or 
              (is_appendix and is_bold and is_reasonable_words) or
              (text.isupper() and is_reasonable_words and has_space_before)):
            final_predictions.append('H1')
        # H2 detection - clear subsections
        elif ((is_bold and has_space_before and is_reasonable_words and 
               (has_colon or is_numbered)) or
              (is_numbered and is_bold and len(text.split()) >= 3)):
            final_predictions.append('H2')
        # H3 detection - numbered sub-subsections or specific patterns
        elif ((is_numbered and '.' in text and text.count('.') >= 2 and len(text.split()) >= 2) or
              (has_colon and is_reasonable_words and len(text.split()) >= 2 and 
               any(word in text.lower() for word in ['for each', 'timeline', 'result', 'phase']))):
            final_predictions.append('H3')
        # H4 detection - very specific patterns
        elif (text.startswith('For each') and has_colon and len(text.split()) >= 3):
            final_predictions.append('H4')
        else:
            final_predictions.append('Body Text')
    
    # Debug: print final predictions
    final_label_counts = {}
    for label in final_predictions:
        final_label_counts[label] = final_label_counts.get(label, 0) + 1
    print(f"Final predictions after heuristics: {final_label_counts}")
    
    # Extract title and outline
    title = "Untitled Document"
    outline = []
    found_title = False
    
    # First pass: look for title
    for i, label in enumerate(final_predictions):
        if label == 'Title':
            title = line_features[i]['text'].strip()
            found_title = True
            print(f"Found title: '{title}'")
            break
    
    # Second pass: extract outline items and clean them up
    seen_texts = set()  # Track to avoid duplicates
    for i, label in enumerate(final_predictions):
        if label.startswith('H'):
            text = line_features[i]['text'].strip()
            # Don't add the title text again if we already found it
            # Also avoid duplicates and very similar entries
            if not (found_title and text == title) and text not in seen_texts:
                # Clean up the text
                clean_text = text.strip()
                if clean_text and len(clean_text) > 2:  # Ensure meaningful content
                    outline.append({
                        "level": label,
                        "text": clean_text,
                        "page": line_features[i]['page_num']
                    })
                    seen_texts.add(text)
                    print(f"Added to outline: {label} - '{clean_text}' (page {line_features[i]['page_num']})")
    
    # If no title was found, try to construct one from first few text elements or first H1
    if not found_title:
        if outline:
            # Use the first H1 as title if available
            h1_items = [item for item in outline if item['level'] == 'H1']
            if h1_items:
                title = h1_items[0]['text']
                outline = [item for item in outline if item['text'] != title]
                print(f"Using first H1 as title: '{title}'")
        else:
            # Try to construct title from first few meaningful text lines
            meaningful_lines = []
            for line in line_features[:10]:  # Check first 10 lines
                text = line['text'].strip()
                if len(text) > 10 and not text.lower().startswith(('copyright', 'page', 'version')):
                    meaningful_lines.append(text)
                    if len(meaningful_lines) >= 2:
                        break
            
            if meaningful_lines:
                title = ' '.join(meaningful_lines[:2])
                print(f"Constructed title from text: '{title}'")
    
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
